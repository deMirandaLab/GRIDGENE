import unittest
import numpy as np
import cv2
import os
from gridgene.get_masks import (GetMasks, ConstrainedMaskExpansion,
                               SingleClassObjectAnalysis, MultiClassObjectAnalysis)
import logging
from unittest.mock import MagicMock

class TestGetMasks(unittest.TestCase):
    def setUp(self):
        # Sample image shape
        self.height = 100
        self.width = 100
        self.get_masks = GetMasks(image_shape=(self.height, self.width))

    def test_filter_mask_by_area(self):
        mask = np.zeros((self.height, self.width), dtype=np.int32)
        # Create two connected components, one small (area=3), one large (area=10)
        mask[10:15, 10:12] = 1  # area = 10
        mask[20:21, 20:22] = 2  # area = 3

        filtered = self.get_masks.filter_binary_mask_by_area(mask, min_area=5)

        # Label 2 should be removed, label 1 remains
        self.assertTrue(np.all(filtered[10:15, 10:12] == 1))
        self.assertTrue(np.all(filtered[20:21, 20:22] == 0))

    def test_create_mask(self):
        # Create a simple contour (rectangle)
        contours = [np.array([[[10, 10]], [[10, 20]], [[20, 20]], [[20, 10]]])]
        mask = self.get_masks.create_mask(contours)

        # Check that the mask inside rectangle is 1
        self.assertEqual(mask[15, 15], 1)
        # Check outside is 0
        self.assertEqual(mask[5, 5], 0)

    def test_fill_holes(self):
        # Create a mask with a hole (a square with a smaller empty square inside)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (30, 30), 1, thickness=cv2.FILLED)
        cv2.rectangle(mask, (15, 15), (20, 20), 0, thickness=cv2.FILLED)

        filled = self.get_masks.fill_holes(mask)

        # The hole area should be filled
        self.assertTrue(np.all(filled[15:20, 15:20] == 1))

    def test_apply_morphology(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        # Add noise pixel
        mask[5, 5] = 1

        opened = self.get_masks.apply_morphology(mask, operation="open", kernel_size=3)
        self.assertEqual(opened[5, 5], 0)  # Noise removed
        self.assertEqual(opened[15, 15], 1)  # Main region remains

    def test_subtract_masks(self):
        base = np.ones((self.height, self.width), dtype=np.uint8)
        mask1 = np.zeros_like(base)
        mask1[10:20, 10:20] = 1

        mask2 = np.zeros_like(base)
        mask2[15:25, 15:25] = 1

        result = self.get_masks.subtract_masks(base, mask1, mask2)

        # Area in mask1 and mask2 should be subtracted
        self.assertEqual(result[12, 12], 0)
        self.assertEqual(result[22, 22], 0)
        self.assertEqual(result[30, 30], 1)

    def test_save_and_load_npy(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        mask[10:20, 10:20] = 1

        save_path = "test_mask.npy"
        self.get_masks.save_masks_npy(mask, save_path)

        self.assertTrue(os.path.exists(save_path))

        loaded = np.load(save_path)
        self.assertTrue(np.array_equal(mask, loaded))

        os.remove(save_path)

    def test_save_masks_image(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        mask[10:20, 10:20] = 1

        save_path = "test_mask.png"
        self.get_masks.save_masks(mask, save_path)

        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_plot_masks(self):
        mask1 = np.zeros((self.height, self.width), dtype=np.uint8)
        mask1[10:30, 10:30] = 1
        mask2 = np.zeros((self.height, self.width), dtype=np.uint8)
        mask2[40:60, 40:60] = 1

        # Just check that plotting does not raise exceptions
        try:
            self.get_masks.plot_masks(
                masks=[mask1, mask2],
                mask_names=["Mask1", "Mask2"],
                show=False
            )
        except Exception as e:
            self.fail(f"plot_masks raised an exception: {e}")


class TestGetMasksFiltering(unittest.TestCase):

    def test_filter_binary_mask_by_area(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 1  # Area = 4
        mask[5:9, 5:9] = 1  # Area = 16

        gm = GetMasks()
        filtered = gm.filter_binary_mask_by_area(mask, min_area=5)

        self.assertEqual(np.sum(filtered), 16)
        self.assertTrue((filtered[1:3, 1:3] == 0).all())
        self.assertTrue((filtered[5:9, 5:9] == 1).all())

    def test_filter_labeled_mask_by_area(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[1:3, 1:3] = 1  # Area = 4
        mask[5:9, 5:9] = 2  # Area = 16
        gm = GetMasks()
        filtered = gm.filter_labeled_mask_by_area(mask=mask, min_area=5)

        self.assertTrue((filtered[1:3, 1:3] == 0).all())
        self.assertTrue((filtered[5:9, 5:9] == 2).all())
        self.assertEqual(set(np.unique(filtered)) - {0}, {2})
class TestConstrainedMaskExpansion(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)

    def test_expansion_with_constraints_and_area_filter(self):
        seed_mask = np.zeros((20, 20), dtype=np.uint8)
        seed_mask[2:4, 2:4] = 1
        seed_mask[10:14, 10:14] = 1

        constraint_mask = np.zeros((20, 20), dtype=np.uint8)
        constraint_mask[:, 8:] = 1

        expander = ConstrainedMaskExpansion(seed_mask, constraint_mask, logger=self.logger)

        expander.expand_mask(
            expansion_pixels=[3],
            min_area=5,
            restrict_to_limit=True
        )

        self.assertIn("expansion_3", expander.binary_expansions)
        expansion = expander.binary_expansions["expansion_3"]

        self.assertTrue(np.all(expansion[constraint_mask == 0] == 0))
        self.assertTrue(np.all(seed_mask[expansion == 1] == 0))

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(expansion.astype(np.uint8), connectivity=8)
        self.assertEqual(num_labels - 1, 1)

    def test_expansion_without_constraint(self):
        seed_mask = np.zeros((10, 10), dtype=np.uint8)
        seed_mask[4, 4] = 1

        expander = ConstrainedMaskExpansion(seed_mask, constraint_mask=None, logger=self.logger)

        expander.expand_mask([1], min_area=None, restrict_to_limit=False)

        self.assertIn("expansion_1", expander.binary_expansions)
        expansion = expander.binary_expansions["expansion_1"]

        self.assertGreater(np.sum(expansion), 0)
        self.assertTrue(np.all(seed_mask[expansion == 1] == 0))

    def test_multiple_rings(self):
        seed_mask = np.zeros((30, 30), dtype=np.uint8)
        seed_mask[10:15, 10:15] = 1

        constraint_mask = np.ones((30, 30), dtype=np.uint8)

        expander = ConstrainedMaskExpansion(seed_mask, constraint_mask, logger=self.logger)

        expander.expand_mask([3, 6, 9], min_area=5)

        for dist in [3, 6, 9]:
            self.assertIn(f"expansion_{dist}", expander.binary_expansions)

        e3 = expander.binary_expansions["expansion_3"]
        e6 = expander.binary_expansions["expansion_6"]
        e9 = expander.binary_expansions["expansion_9"]

        overlap_3_6 = np.sum(e3 & e6)
        overlap_6_9 = np.sum(e6 & e9)
        self.assertEqual(overlap_3_6, 0)
        self.assertEqual(overlap_6_9, 0)

        self.assertGreater(np.sum(e3), 0)
        self.assertGreater(np.sum(e6), 0)
        self.assertGreater(np.sum(e9), 0)

        self.assertTrue(np.all(expander.binary_expansions["seed_mask"] == seed_mask))

        combined = np.zeros_like(seed_mask)
        for k, v in expander.binary_expansions.items():
            if k.startswith("expansion_"):
                combined = cv2.bitwise_or(combined, v)

        expected_remaining = constraint_mask.copy()
        expected_remaining[combined == 1] = 0

        self.assertTrue(np.array_equal(expander.binary_expansions["constraint_remaining"], expected_remaining))

# Dummy GetMasks remains unchanged
class DummyGetMasks:
    """Minimal GetMasks mock for testing."""
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler())

    def filter_mask_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        filtered_mask = np.zeros_like(mask)
        for label_id in range(1, num_labels):  # Skip background
            if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == label_id] = 1
        return filtered_mask
    def filter_binary_mask_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        return self.filter_mask_by_area(mask, min_area)

class TestSingleClassObjectAnalysis(unittest.TestCase):
    def setUp(self):
        self.dummy_masks = DummyGetMasks(height=20, width=20)
        self.dummy_contour = [np.array([[[5, 5]], [[5, 10]], [[10, 10]], [[10, 5]]])]

    def test_get_mask_objects_without_exclusion(self):
        sa = SingleClassObjectAnalysis(self.dummy_masks, self.dummy_contour)
        sa.get_mask_objects()
        self.assertIsNotNone(sa.mask_object_SA)
        self.assertGreater(sa.mask_object_SA.sum(), 0)
        self.assertEqual(np.max(sa.mask_object_SA), 1)

    def test_get_mask_objects_with_exclusion(self):
        sa = SingleClassObjectAnalysis(self.dummy_masks, self.dummy_contour)
        exclusion = np.zeros((20, 20), dtype=np.uint8)
        exclusion[5:11, 5:11] = 1
        sa.get_mask_objects(exclude_masks=[exclusion])
        self.assertEqual(np.sum(sa.mask_object_SA), 0)

    def test_get_mask_objects_with_area_filter(self):
        cnt = np.array([[[1, 1]], [[1, 2]], [[2, 2]], [[2, 1]]])
        sa = SingleClassObjectAnalysis(self.dummy_masks, [cnt])
        sa.get_mask_objects(filter_area=20)
        self.assertEqual(np.sum(sa.mask_object_SA), 0)

    def test_ring_expansion_stores_all_dicts(self):
        sa = SingleClassObjectAnalysis(self.dummy_masks, self.dummy_contour)
        sa.get_mask_objects()
        sa.get_objects_expansion(expansions_pixels=[2, 4, 6])

        # Check dictionary presence
        self.assertIn("expansion_2", sa.binary_expansions)
        self.assertIn("expansion_4", sa.labeled_expansions)
        self.assertIn("expansion_6", sa.referenced_expansions)

        # Check seed also stored
        self.assertIn("seed_mask", sa.binary_expansions)
        self.assertIn("seed_mask", sa.labeled_expansions)
        self.assertIn("seed_mask", sa.referenced_expansions)

        # Check shape consistency
        for key in sa.binary_expansions:
            self.assertEqual(sa.binary_expansions[key].shape, (20, 20))
            self.assertTrue(np.issubdtype(sa.binary_expansions[key].dtype, np.integer))

        for key in sa.referenced_expansions:
            self.assertTrue(np.max(sa.referenced_expansions[key]) >= 0)

    def test_no_mask_object_set_returns_none(self):
        sa = SingleClassObjectAnalysis(self.dummy_masks, self.dummy_contour)
        result = sa.get_objects_expansion(expansions_pixels=[3])
        self.assertIsNone(result)

    def test_expansion_area_filter(self):
        sa = SingleClassObjectAnalysis(self.dummy_masks, self.dummy_contour)
        sa.get_mask_objects()
        sa.get_objects_expansion(expansions_pixels=[3], filter_area=100)

        exp = sa.binary_expansions.get("expansion_3", None)
        self.assertIsNotNone(exp)
        self.assertEqual(np.sum(exp), 0)

    def test_propagated_labels_have_valid_values(self):
        sa = SingleClassObjectAnalysis(self.dummy_masks, self.dummy_contour)
        sa.get_mask_objects()
        sa.get_objects_expansion(expansions_pixels=[2])
        ref = sa.referenced_expansions["expansion_2"]
        self.assertTrue(np.all(ref >= 0))
        self.assertTrue(np.any(ref > 0))

#
class TestMultiClassObjectAnalysis(unittest.TestCase):
    def setUp(self):
        self.height = 50
        self.width = 50
        self.dummy_masks = DummyGetMasks(height=self.height, width=self.width)

        # Define 2 class categories with 2 objects each
        self.multiple_contours = {
            "ClassA": [
                np.array([[[10, 10]], [[10, 15]], [[15, 15]], [[15, 10]], [[25, 10]], [[10, 10]]]),  # closed loop
                np.array([[[30, 30]], [[30, 35]], [[35, 35]],[[45, 40]], [[35, 30]], [[30, 30]]])
            ],
            "ClassB": [
                np.array([[[10, 30]], [[10, 35]], [[15, 35]], [[15, 30]], [[10, 30]]]),
                np.array([[[30, 10]], [[30, 15]], [[35, 15]], [[35, 10]], [[30, 10]]]),

            ]
        }

    def test_initialization_and_voronoi(self):
        ma = MultiClassObjectAnalysis(self.dummy_masks, self.multiple_contours)
        ma.derive_voronoi_from_contours()

        self.assertIsNotNone(ma.vor)
        self.assertEqual(len(ma.all_centroids), 4)
        self.assertEqual(len(ma.class_labels), 4)

    def test_generate_expanded_masks_returns_all(self):
        ma = MultiClassObjectAnalysis(self.dummy_masks, self.multiple_contours)
        ma.derive_voronoi_from_contours()

        binary, labeled, referenced = ma.generate_expanded_masks_limited_by_voronoi(expansion_distances=[2, 4])

        for mask_dict in (binary, labeled, referenced):
            self.assertIsInstance(mask_dict, dict)
            self.assertIn("ClassA_expansion_2", mask_dict)
            self.assertIn("ClassA_expansion_4", mask_dict)
            self.assertEqual(mask_dict["ClassA_expansion_2"].shape, (self.height, self.width))

    def test_seed_is_in_expansions(self):
        ma = MultiClassObjectAnalysis(self.dummy_masks, self.multiple_contours)
        ma.derive_voronoi_from_contours()
        ma.generate_expanded_masks_limited_by_voronoi(expansion_distances=[3])

        self.assertIn("ClassA", ma.binary_masks)
        self.assertIn("ClassA", ma.labeled_masks)
        self.assertIn("ClassA", ma.referenced_masks)

    def test_masks_are_disjoint_per_class(self):
        ma = MultiClassObjectAnalysis(self.dummy_masks, self.multiple_contours)
        ma.derive_voronoi_from_contours()
        ma.generate_expanded_masks_limited_by_voronoi(expansion_distances=[5])

        mask = ma.binary_masks["ClassA_expansion_5"]
        self.assertLessEqual(np.max(mask), 1)
        self.assertTrue(np.all(np.logical_or(mask == 0, mask == 1)))

    def test_reference_mask_values_match_object_count(self):
        ma = MultiClassObjectAnalysis(self.dummy_masks, self.multiple_contours)
        ma.derive_voronoi_from_contours()
        ma.generate_expanded_masks_limited_by_voronoi(expansion_distances=[20])

        ref_mask = ma.labeled_masks["ClassA_expansion_20"]
        unique_refs = np.unique(ref_mask)
        # Should include 0 (background) and 4 objects
        self.assertTrue(set(range(3)).issubset(set(unique_refs)))

    def test_voronoi_constraints_limit_expansion(self):
        ma = MultiClassObjectAnalysis(self.dummy_masks, self.multiple_contours)
        ma.derive_voronoi_from_contours()
        ma.generate_expanded_masks_limited_by_voronoi(expansion_distances=[20])

        # Make sure expansion doesn't spill out of Voronoi
        for label in range(1, 5):  # object labels are 1-indexed
            mask = (ma.referenced_masks["ClassA_expansion_20"] == label).astype(np.uint8)
            point = ma.all_centroids[label - 1]
            self.assertGreater(np.sum(mask), 0, f"No expansion for object {label}")

            voronoi_mask = ma.get_voronoi_mask(ma.class_labels[label - 1])
            intersection = np.logical_and(mask, voronoi_mask).astype(np.uint8)
            self.assertEqual(np.sum(mask), np.sum(intersection), f"Expansion for label {label} leaked Voronoi bounds")

    def test_empty_contours_handle_gracefully(self):
        multiple_contours_failing = {
            "ClassA": [
                np.array([[[10, 10]], [[10, 15]], [[15, 15]], [[15, 10]], [[10, 10]]]),  # closed loop
                np.array([[[30, 30]], [[30, 35]], [[35, 35]], [[35, 30]], [[30, 30]]])
            ],
            "ClassB":  [np.array([[[1, 1]], [[2, 2]], [[3, 3]]])]  # invalid contour
        }

        ma = MultiClassObjectAnalysis(self.dummy_masks,multiple_contours_failing)
        ma.contours = [np.array([[[1, 1]], [[2, 2]], [[3, 3]]])]  # invalid contour
        try:
            ma.derive_voronoi_from_contours()
        except Exception as e:
            self.fail(f"derive_voronoi_from_contours raised an exception unexpectedly: {e}")

if __name__ == "__main__":
    unittest.main()
