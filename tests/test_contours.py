import unittest
import numpy as np
import cv2
import os
import tempfile
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from gridgen.contours import GetContour
from gridgen.contours import ConvolutionContours


def make_dummy_contour(center, radius, points=8):
    """Creates a dummy circular contour."""
    angle = np.linspace(0, 2 * np.pi, points, endpoint=False)
    contour = np.stack([
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle)
    ], axis=1).astype(np.int32)
    return contour.reshape((-1, 1, 2))

class TestGetContour(unittest.TestCase):

    def setUp(self):
        self.height = 100
        self.width = 100
        self.genes = 3
        self.array = np.zeros((self.height, self.width, self.genes), dtype=np.float32)

        self.array[30:40, 30:40, 0] = 5
        self.array[60:70, 60:70, 1] = 5

        self.gc = GetContour(array_to_contour=self.array)
        self.gc.contours = [
            make_dummy_contour((35, 35), 5).squeeze(1),
            make_dummy_contour((65, 65), 5).squeeze(1),
            make_dummy_contour((10, 10), 3).squeeze(1),
        ]
        self.gc.local_sum_image = np.sum(self.array, axis=2)
        self.gc.contour_name = "testplot"

    def test_filter_area(self):
        self.gc.filter_contours_area(min_area_threshold=50)
        self.assertTrue(all(cv2.contourArea(c) >= 50 for c in self.gc.contours))

    def test_filter_no_counts(self):
        filtered = self.gc.filter_contours_no_counts()
        self.assertEqual(len(filtered), 2)

    def test_filter_gene_threshold(self):
        gene_array = np.sum(self.array, axis=2)
        self.gc.filter_contours_by_gene_threshold(gene_array, threshold=100, gene_name="TestGene")
        self.assertEqual(len(self.gc.contours), 2)

    def test_filter_gene_comparison(self):
        gene1 = self.array[..., 0]
        gene2 = self.array[..., 1]
        self.gc.filter_contours_by_gene_comparison(gene1, gene2, gene_name1="G1", gene_name2="G2")
        self.assertEqual(len(self.gc.contours), 1)

    def test_plot_contours_scatter_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ax = self.gc.plot_contours_scatter(path=tmpdir, show=False)
            self.assertIsInstance(ax, Axes)
            expected_path = os.path.join(tmpdir, f'Scatter_contours_{self.gc.contour_name}.png')
            self.assertTrue(os.path.exists(expected_path))

    def test_plot_conv_sum_runs(self):
        ax = self.gc.plot_conv_sum(show=False)
        self.assertIsInstance(ax, Axes)

class TestConvolutionContours(unittest.TestCase):

    def setUp(self):
        self.height, self.width, self.genes = 100, 100, 3
        self.array = np.zeros((self.height, self.width, self.genes), dtype=np.float32)

        # Add synthetic expression in two spots
        self.array[30:40, 30:40, 0] = 1
        self.array[60:70, 60:70, 1] = 1

        self.cc = ConvolutionContours(self.array)

    def test_get_conv_sum_square(self):
        self.cc.get_conv_sum(kernel_size=5, kernel_shape='square')
        self.assertIsNotNone(self.cc.local_sum_image)
        self.assertEqual(self.cc.local_sum_image.shape, (self.height, self.width))

    def test_get_conv_sum_circle(self):
        self.cc.get_conv_sum(kernel_size=5, kernel_shape='circle')
        self.assertIsNotNone(self.cc.local_sum_image)

    def test_contours_from_sum(self):
        self.cc.get_conv_sum(kernel_size=5)
        self.cc.contours_from_sum(density_threshold=1.0, min_area_threshold=10)
        self.assertTrue(len(self.cc.contours) > 0)

    def test_convolution_preserves_shape(self):
        self.cc.get_conv_sum(kernel_size=10)
        self.assertEqual(self.cc.local_sum_image.shape, self.array.shape[:2])

    def test_convolution_border_values(self):
        self.cc.get_conv_sum(kernel_size=10)
        # Just check that the shape is preserved and values are finite
        self.assertEqual(self.cc.local_sum_image.shape, self.array.shape[:2])
        self.assertTrue(np.isfinite(self.cc.local_sum_image).all())


if __name__ == '__main__':
    unittest.main()


# (GRIDGEN) martinha@gaia:~/PycharmProjects/phd/spatial_transcriptomics/GRIDGEN$ python -m unittest tests/test_contours.py
