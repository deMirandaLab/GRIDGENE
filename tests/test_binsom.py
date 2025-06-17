import unittest
import numpy as np
import pandas as pd
import logging
from anndata import AnnData
from gridgen.binsom import GetBins, GetContour  # Assuming your classes are in `binsom.py`

class TestBinSOM(unittest.TestCase):

    def setUp(self):
        # Set up dummy spatial transcriptomics data
        np.random.seed(0)
        self.df = pd.DataFrame({
            'X': np.random.randint(0, 100, size=100),
            'Y': np.random.randint(0, 100, size=100),
            'target': np.random.choice(['GeneA', 'GeneB', 'GeneC'], size=100)
        })

        self.unique_targets = ['GeneA', 'GeneB', 'GeneC']
        self.bin_size = 10
        self.get_bins = GetBins(bin_size=self.bin_size, unique_targets=self.unique_targets, logger=logging.getLogger("test"))
        self.adata = self.get_bins.get_bin_df(self.df, df_name="Sample1")
        self.get_contour = GetContour(self.adata, logger=logging.getLogger("test"))

    def test_get_bin_df_output(self):
        self.assertIsInstance(self.adata, AnnData)
        self.assertEqual(self.adata.shape[1], len(self.unique_targets))
        self.assertTrue(all(name in self.adata.var_names for name in self.unique_targets))

    def test_get_bin_cohort(self):
        df_list = [self.df, self.df.copy()]
        df_name_list = ["Sample1", "Sample2"]
        self.get_bins.get_bin_cohort(df_list, df_name_list, cohort_name="Cohort1")
        adata = self.get_bins.adata
        self.assertEqual(len(adata.obs['cohort'].unique()), 1)
        self.assertEqual(len(adata.obs['name'].unique()), 2)

    def test_preprocess_bin(self):
        pre_bin = self.adata.copy()
        self.get_bins.preprocess_bin(min_counts=1, adata=pre_bin)
        self.assertIn("counts", pre_bin.layers)
        self.assertTrue(np.count_nonzero(pre_bin.X > 0) > 0)

    def test_run_som(self):
        self.get_contour.run_som(som_shape=(2, 1), n_iter=10)
        self.assertIn("cluster_som", self.get_contour.adata.obs.columns)
        self.assertGreaterEqual(self.get_contour.adata.obs["cluster_som"].nunique(), 1)

    def test_eval_som_statistical(self):
        self.get_contour.run_som(som_shape=(2, 1), n_iter=10)
        self.get_contour.eval_som_statistical(top_n=2)
        df = self.get_contour.eval_som_statistical_df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("names", df.columns)
        self.assertLessEqual(len(df), 2 * self.get_contour.adata.obs["cluster_som"].nunique())

    def test_create_cluster_image(self):
        self.get_contour.run_som(som_shape=(2, 1), n_iter=10)
        img = self.get_contour.create_cluster_image(self.adata, grid_size=self.bin_size)
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.sum(), 0)  # SOM clusters added to image

    def test_plot_som(self):
        self.get_contour.run_som(som_shape=(2, 1), n_iter=10)
        img = self.get_contour.create_cluster_image(self.adata, grid_size=self.bin_size)
        try:
            ax = self.get_contour.plot_som(som_image=img, show=False)
            self.assertIsNotNone(ax)
        except Exception as e:
            self.fail(f"plot_som() raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
