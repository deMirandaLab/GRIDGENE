import unittest
import numpy as np
from gridgen.mask_properties import MaskDefinition, MaskAnalysisPipeline
from skimage.measure import label, regionprops_table


class TestMaskAnalysisPipeline(unittest.TestCase):

    def setUp(self):
        self.simple_mask = np.zeros((10, 10), dtype=np.uint8)
        self.simple_mask[2:4, 2:4] = 1
        self.simple_mask[6:8, 6:8] = 1

        self.bulk_mask = np.ones((10, 10), dtype=np.uint8)

        self.grid_mask = np.zeros((10, 10), dtype=np.uint8)
        self.grid_mask[0:5, 0:5] = 1
        self.grid_mask[5:10, 5:10] = 1

        self.parent_mask = np.zeros((10, 10), dtype=np.uint8)
        self.parent_mask[2:6, 2:6] = 1

        self.child_mask = np.zeros((10, 10), dtype=np.uint8)
        self.child_mask[3:5, 3:5] = 1

        self.array_counts = np.zeros((10, 10, 2), dtype=np.int64)
        self.array_counts[3:5, 3:5, 0] = 5  # gene1 counts in child region
        self.array_counts[3:5, 3:5, 1] = 10 # gene2 counts in child region

        self.target_dict = {'gene1': 0, 'gene2': 1}

    def test_per_object_analysis(self):
        defs = [MaskDefinition(mask=self.simple_mask, mask_name="test_obj", analysis_type="per_object")]
        pipeline = MaskAnalysisPipeline(defs, self.array_counts, self.target_dict)
        results = pipeline.run()
        self.assertEqual(results[0].mask_name, "test_obj")
        self.assertEqual(results[0].analysis_type, "per_object")
        self.assertEqual(len(results[0].features), 2)  # two objects detected

    def test_bulk_analysis(self):
        defs = [MaskDefinition(mask=self.bulk_mask, mask_name="test_bulk", analysis_type="bulk")]
        pipeline = MaskAnalysisPipeline(defs, self.array_counts, self.target_dict)
        results = pipeline.run()
        feature = results[0].features[0]
        self.assertEqual(feature['area'], 100)
        self.assertEqual(feature['object_id'], 'bulk')

    # def test_grid_analysis(self):
    #     defs = [MaskDefinition(mask=self.grid_mask, mask_name="test_grid", analysis_type="grid", grid_size=5)]
    #     pipeline = MaskAnalysisPipeline(defs, self.array_counts, self.target_dict)
    #     results = pipeline.run()
    #     areas = sorted([f['area'] for f in results[0].features])
    #     self.assertEqual(areas, [0, 0, 25, 25])
    def test_mask_analysis_pipeline_grid(self):
        from skimage.measure import label

        labeled_grid_mask = label(self.grid_mask)
        print("Unique labels in grid mask:", np.unique(labeled_grid_mask))

        array_counts = np.zeros_like(self.array_counts)
        array_counts[0:5, 0:5, 0] = 5
        array_counts[0:5, 0:5, 1] = 10
        array_counts[5:10, 5:10, 0] = 7
        array_counts[5:10, 5:10, 1] = 14

        mask_def = MaskDefinition(
            mask=labeled_grid_mask,
            mask_name="test_mask",
            analysis_type="grid",
            grid_size=5
        )

        pipeline = MaskAnalysisPipeline(
            mask_definitions=[mask_def],
            array_counts=array_counts,
            target_dict=self.target_dict
        )

        results = pipeline.run()
        df = pipeline.get_results_df()

        print(df[['gene1', 'gene2']])
        print("Sum of gene counts:", df[['gene1', 'gene2']].sum().sum())

        self.assertFalse(df.empty)

        expected_cols = {"object_id", "x", "y", "area", "centroid_x", "centroid_y"}
        gene_cols = set(self.target_dict.keys())

        self.assertTrue(expected_cols.issubset(df.columns))
        self.assertTrue(gene_cols.issubset(df.columns))

        self.assertGreater(df[list(gene_cols)].sum().sum(), 0)


    def test_hierarchy_mapping(self):
        defs = [
            MaskDefinition(mask=self.parent_mask, mask_name="parent", analysis_type="per_object"),
            MaskDefinition(mask=self.child_mask, mask_name="child", analysis_type="per_object")
        ]
        pipeline = MaskAnalysisPipeline(defs, self.array_counts, self.target_dict)
        pipeline.run()

        hierarchy_definitions = {
            "child": {
                "labels": label(self.child_mask),
                "level_hierarchy": "parent"
            }
        }

        df = pipeline.map_hierarchies(hierarchy_definitions)

        child_result = next(r for r in pipeline.results if r.mask_name == "child")
        for feature in child_result.features:
            self.assertIn('parent_ids', feature)
            self.assertIsInstance(feature['parent_ids'], list)

        self.assertIn('mask_name', df.columns)
        self.assertIn('object_id', df.columns)
        self.assertIn('parent_mask', df.columns)
        self.assertIn('parent_ids', df.columns)
        self.assertEqual(df.iloc[0]['mask_name'], 'child')
        self.assertEqual(df.iloc[0]['parent_mask'], 'parent')

if __name__ == '__main__':
    unittest.main()