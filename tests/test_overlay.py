import unittest
import numpy as np
from gridgene.overlay import Overlay  # Assuming your class is in a file named overlay.py
from PIL import Image

class TestOverlay(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 100)

        # Create dummy masks
        self.mask_dict = {
            "Mask1": np.zeros(self.shape, dtype=np.uint8),
            "Mask2": np.zeros(self.shape, dtype=np.uint8),
        }
        self.mask_dict["Mask1"][20:40, 20:40] = 1
        self.mask_dict["Mask2"][60:80, 60:80] = 1

        # Dummy polygon segmentation
        self.segmentation = {
            "geometries": [
                {
                    "cell": "1",
                    "coordinates": [[(22, 22), (22, 38), (38, 38), (38, 22), (22, 22)]],
                },
                {
                    "cell": "2",
                    "coordinates": [[(62, 62), (62, 78), (78, 78), (78, 62), (62, 62)]],
                },
            ]
        }

    def test_initialization_and_type_detection(self):
        overlay = Overlay(self.mask_dict, self.segmentation)
        self.assertEqual(overlay.segmentation_type, "polygons")

    def test_polygon_shifting(self):
        overlay = Overlay(self.mask_dict, self.segmentation, min_x=10, min_y=10)
        shifted_coords = overlay.segmentation['geometries'][0]['coordinates'][0][0]
        self.assertEqual(shifted_coords, (12, 12))  # originally 22,22

    def test_mask_extraction_from_polygons(self):
        overlay = Overlay(self.mask_dict, self.segmentation)
        masks = overlay._extract_segmentation_masks()
        self.assertEqual(len(masks), 2)
        for m in masks.values():
            self.assertEqual(m.shape, self.shape)

    def test_compute_overlap(self):
        overlay = Overlay(self.mask_dict, self.segmentation)
        results = overlay.compute_overlap()
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)  # Two cells
        for cell_id in results:
            self.assertIn("Mask1", results[cell_id])
            self.assertIn("Mask2", results[cell_id])

    def test_plot_masks_overlay_segmentation(self):
        overlay = Overlay(self.mask_dict, self.segmentation)
        try:
            overlay.plot_masks_overlay_segmentation(
                titles=["Mask1", "Mask2"],
                colors=["red", "blue"],
                show=False
            )
        except Exception as e:
            self.fail(f"plot_masks_overlay_segmentation raised an exception: {e}")

    def test_plot_colored_by_mask_overlap(self):
        overlay = Overlay(self.mask_dict, self.segmentation)
        overlay.compute_overlap()
        try:
            overlay.plot_colored_by_mask_overlap(
                mask_to_color=["Mask1"],
                color_map='Reds',
                show=False
            )
        except Exception as e:
            self.fail(f"plot_colored_by_mask_overlap raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
