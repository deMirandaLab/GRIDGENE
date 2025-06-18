import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import Polygon
import time
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from gridgen.logger import get_logger
from typing import Optional, Tuple, List, Dict, Any, Union
from matplotlib.figure import Figure

from matplotlib.axes import Axes
from functools import wraps
from sklearn.neighbors import BallTree

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start

        # Try to log using `self.logger` if available
        instance = args[0]  # self, assuming it's a method
        logger = getattr(instance, 'logger', None)
        if logger:
            logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
        else:
            print(f"{func.__name__} took {elapsed:.4f} seconds")

        return result
    return wrapper

class GetContour():
    """
    Parent class for contour handling and filtering.
    A class to handle contour extraction and filtering from a 3D array.

    Attributes
    ----------
    array : np.ndarray
        The 3D array from which contours are to be extracted. xand y positions are the first two dimensions.
        each gene is expected in the z dimension.
    local_sum_image : np.ndarray
        The 2D array representing the local sum of the input array.
    contours : list
        List of contours extracted from the array.
    contour_name : str
        Name of the contour for identification.
    total_valid_contours : int
        Total number of valid contours after filtering.
    contours_filtered_area : int
        Number of contours remaining after area filtering.
    logger : logging.Logger
        Logger for logging information and errors.
    points_x_y: np.ndarray
        Optional 2D array of points (x, y) for plotting or further processing.
    Methods
    -------
    __init__(array_to_contour, logger=None, contour_name=None):
        Initializes the GetContour class with the given array and optional logger and contour name.
    get_conv_sum(kernel_size, kernel_shape='square'):
        Computes the convolution sum of the array with a specified kernel.
    check_contours():
        Checks and processes the contours to ensure they are valid.
    filter_contours_area(min_area_threshold):
        Filters contours based on a minimum area threshold.
    contours_from_sum(density_threshold, min_area_threshold, directionality='higher'):
        Extracts contours from the local sum image based on a density threshold and filters them by area.
    filter_contours_no_counts():
        Filters contours that have no counts in the given array.
    filter_contours_by_gene_threshold(gene_array, threshold, gene_name=""):
        Filters contours based on a gene count threshold.
    filter_contours_by_gene_comparison(gene_array1, gene_array2, gene_name1="", gene_name2=""):
        Filters contours based on the comparison of gene counts between two gene arrays.
    plot_contours_scatter(path=None, show=False, s=0.1, alpha=0.5, linewidth=1, c_points='blue', c_contours='red', figsize=(10, 10), ax=None, **kwargs):
        Plot scatter plot with contours.
    plot_conv_sum(cmap='plasma', c_countour='white', path=None, show=False, figsize=(10, 10), ax=None):
        Plot the convolution sum image with contours.

    """

    def __init__(self, array_to_contour, logger=None, contour_name=None, points_x_y:np.ndarray = None):
        """
        Initializes the GetContour class with the given array and optional logger and contour name.

        Parameters
        ----------
        array_to_contour : np.ndarray
            The 3D array from which contours are to be extracted.
        logger : logging.Logger, optional
            Logger for logging information and errors (default is None, which configures a default logger).
        contour_name : str, optional
            Name of the contour for identification (default is None).
        points_x_y : np.ndarray, optional
            Optional 2D array of points (x, y) for plotting or further processing (default is None).
        """
        self.array = array_to_contour
        self.contours = None
        self.contour_name = contour_name
        self.total_valid_contours = 0
        self.contours_filtered_area = 0
        self.logger = logger or get_logger(f'{__name__}.{contour_name or "GetContour"}')
        self.logger.info("Initialized GetContour")
        self.points_x_y = points_x_y

    #############################
    # Filtering of contours

    def check_contours(self) -> None:
        """
        Validate and process contours.

        - Excludes contours with fewer than 3 points.
        - Ensures contours are closed (first point == last point).
        - Converts contours to np.ndarray of dtype int32 for OpenCV compatibility.

        Returns
        -------
        None
        """

        if not self.contours:
            self.logger.info("No contours to check.")
            return

        filtered_and_closed_contours = []
        for contour in self.contours:
            squeezed = np.squeeze(contour)
            if squeezed.ndim != 2 or squeezed.shape[0] < 3 or squeezed.shape[1] < 2:
                continue
            if not np.array_equal(squeezed[0], squeezed[-1]):
                squeezed = np.vstack([squeezed, squeezed[0]])
            filtered_and_closed_contours.append(squeezed.astype(np.int32))

        self.contours = filtered_and_closed_contours

    def filter_contours_area(self, min_area_threshold: float) -> None:
        """
        Filters contours based on a minimum area threshold.

        Parameters
        ----------
        min_area_threshold : float
            Minimum area threshold for filtering contours.
        """
        self.contours = [contour for contour in self.contours if cv2.contourArea(contour) >= min_area_threshold]
        # self.contours_filtered_area = len(self.contours)
        self.logger.info(f'Number of contours after area filtering: {len(self.contours)}')

    def filter_contours_no_counts(self) -> List[np.ndarray]:
        """
        Filters contours that have no counts in the given array.

        This method iterates through each contour, creates a mask for the contour, and checks if the sum of the
        masked array within the contour is greater than 0. Contours are kept if they have counts.

        Returns
        -------
        list
            The list of valid contours that have counts.
        """
        # todo check what is more efficient
        array2d = np.sum(self.array, axis=2)
        valid_contours = []

        mask_all = np.zeros_like(array2d, dtype=np.uint8)
        cv2.drawContours(mask_all, self.contours, -1, 1, thickness=cv2.FILLED)
        # Multiply once to get the masked array
        masked_array2d = array2d * mask_all

        # valid_contours = [
        #     contour for contour in self.contours
        #     if np.sum(cv2.drawContours(
        #         np.zeros_like(array2d, dtype=np.uint8), [contour], -1, 1, thickness=cv2.FILLED) * masked_array2d) > 0
        # ]
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)  # bounding box of contour
            roi = array2d[y:y + h, x:x + w]  # region of interest in array2d

            mask = np.zeros((h, w), dtype=np.uint8)
            # Shift contour points to roi coords
            shifted_contour = contour - [x, y]
            cv2.drawContours(mask, [shifted_contour], -1, 1, thickness=cv2.FILLED)

            if np.sum(roi * mask) > 0:
                valid_contours.append(contour)

        self.contours = valid_contours
        self.logger.info(f'Number of contours after filtering no counts: {len(self.contours)}')
        return self.contours

    def filter_contours_no_counts_and_area(self, min_area_threshold: float) -> List[np.ndarray]:
        """
        Filters contours that have no counts in the given array and are smaller than the minimum area threshold.

        Returns
        -------
        List[np.ndarray]
            The list of contours with counts and meeting area threshold.
        """
        if self.array.ndim == 3:
            array2d = np.sum(self.array, axis=2)
        elif self.array.ndim == 2:
            array2d = self.array
        else:
            raise ValueError(f"Unexpected array shape: {array.shape}")
        valid_contours = []

        for contour in self.contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            roi = array2d[y:y + h, x:x + w]

            mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour - [x, y]
            cv2.drawContours(mask, [shifted_contour], -1, 1, thickness=cv2.FILLED)

            if np.sum(roi * mask) > 0:
                valid_contours.append(contour)

        self.contours = valid_contours
        # self.contours_filtered_area = len(self.contours)
        self.logger.info(f'Number of contours after filtering no counts: {len(self.contours)}')
        return self.contours

    def filter_contours_by_gene_threshold(
            self,
            gene_array: np.ndarray,
            threshold: float,
            gene_name: Optional[str] = ""
    ) -> None:
        """
        Filters contours based on a gene count threshold.

        This method iterates through each contour, creates a mask for the contour, and calculates the gene count
        within the masked region. Contours are kept if the gene count is greater than or equal to the threshold.

        Parameters
        ----------
        gene_array : np.ndarray
            The gene array to be checked. Can be a 2D or 3D array.
        threshold : float
            The gene count threshold for filtering contours.
        gene_name : str, optional
            Name of the gene for logging purposes (default is an empty string).

        Returns
        -------
        None
            The method updates the `self.contours` attribute with the valid contours.
        """
        valid_contours = []
        for i, contour in enumerate(self.contours):
            mask_ = np.zeros((gene_array.shape[0], gene_array.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
            gene_count = np.sum(gene_array * mask_)
            if gene_count >= threshold:
                valid_contours.append(contour)
            else:
                self.logger.info(
                    f'Excluding contour {i}. Gene {gene_name} count  {gene_count} is below threshold {threshold}')
        self.contours = valid_contours
        self.logger.info(f'Number of contours remaining: {len(valid_contours)}')

    def filter_contours_by_gene_comparison(
            self,
            gene_array1: np.ndarray,
            gene_array2: np.ndarray,
            gene_name1: Optional[str] = "",
            gene_name2: Optional[str] = ""
    ) -> None:
        """
        Filters contours based on the comparison of gene counts between two gene arrays.

        This method iterates through each contour, creates a mask for the contour, and calculates the gene counts
        for the given gene arrays within the masked region. Contours are kept if the gene count in `gene_array1`
        is greater than the gene count in `gene_array2`.

        Parameters
        ----------
        gene_array1 : np.ndarray
            The first gene array to be compared. Can be a 2D or 3D array.
        gene_array2 : np.ndarray
            The second gene array to be compared. Can be a 2D or 3D array.
        gene_name1 : str, optional
            Name of the first gene for logging purposes (default is an empty string).
        gene_name2 : str, optional
            Name of the second gene for logging purposes (default is an empty string).

        Returns
        -------
        None
            The method updates the `self.contours` attribute with the valid contours.
        """
        # Ensure arrays are 2D by summing if needed
        if gene_array1.ndim == 3:
            gene_array1 = np.sum(gene_array1, axis=-1)
        if gene_array2.ndim == 3:
            gene_array2 = np.sum(gene_array2, axis=-1)

        height, width = gene_array1.shape
        valid_contours = []
        for i, contour in enumerate(self.contours):
            mask_ = np.zeros((gene_array1.shape[0], gene_array1.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
            gene_count1 = np.sum(gene_array1 * mask_)
            gene_count2 = np.sum(gene_array2 * mask_)
            if gene_count1 > gene_count2:
                valid_contours.append(contour)
            else:
                self.logger.info(
                    f'Excluding contour {i}. '
                    f'{gene_name1 or "Gene1"} count {gene_count1:.2f} '
                    f'≤ {gene_name2 or "Gene2"} count {gene_count2:.2f}'
                )
        self.contours = valid_contours
        self.logger.info(f'Contours remaining after gene comparison: {len(valid_contours)}')


    # Plotting
    def plot_contours_scatter(
            self,
            path: Optional[str] = None,
            show: bool = False,
            s: float = 0.1,
            alpha: float = 0.5,
            linewidth: float = 1,
            c_points: str = 'blue',
            c_contours: str = 'red',
            figsize: Tuple[int, int] = (10, 10),
            ax: Optional[Axes] = None,
            **kwargs: Dict[str, Any]
    ) -> Axes:
        """
        Plot scatter plot with contours.

        :param path: Path to save the plot
        :param show: Whether to display the plot
        :param s: Size of scatter points
        :param alpha: Alpha transparency of scatter points
        :param linewidth: Line width for contours
        :param c_points: Color of scatter points
        :param c_contours: Color of contours
        :param ax: Axes object to draw the plot on (default is None, plot is drawn on the current axes)
        :param kwargs: Additional keyword arguments for scatter and plot
        """
        if self.points_x_y is not None:
            x = self.points_x_y[:, 0].astype(int)  # X column
            y = self.points_x_y[:, 1].astype(int)  # Y column
        else:
            x, y = np.where(np.sum(self.array, axis=2) > 0)

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        # Extract specific kwargs for scatter and plot if provided
        scatter_kwargs = kwargs.get('scatter_kwargs', {})
        plot_kwargs = kwargs.get('plot_kwargs', {})

        # Scatter plot with original coordinates
        ax.scatter(x, y, c=c_points, marker='.', s=s, alpha=alpha, **scatter_kwargs)

        # Rescale and plot the contours
        for contour in self.contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=c_contours, **plot_kwargs)

        ax.set_title(f'Scatter with contours and genes {self.contour_name}')

        if path is not None:
            save_path = os.path.join(path, f'Scatter_contours_{self.contour_name}.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        if show:
            plt.show()

        return ax

    def plot_conv_sum(
            self,
            cmap: str = 'plasma',
            c_countour: str = 'white',
            path: Optional[str] = None,
            show: bool = False,
            figsize: Tuple[int, int] = (10, 10),
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot the convolution sum image with contours.

        :param cmap: Colormap for the convolution sum image (default is 'plasma')
        :param c_countour: Color for the contours (default is 'white')
        :param path: Path to save the plot (default is None, plot is not saved)
        :param show: Whether to display the plot (default is False)
        :param ax: Axes object to draw the plot on (default is None, plot is drawn on the current axes)
        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        im = ax.imshow(self.local_sum_image.T, cmap=cmap, interpolation='none', origin='lower')
        ax.set_title(f'Count distribution with contours for {self.contour_name}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Rescale and plot the contours
        for contour in self.contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=c_countour)

        # Add a colorbar for the colormap
        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label('Color scale', rotation=270)
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax, with 5% width of ax
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create colorbar in the appended axes
        # `cax` argument places the colorbar in the cax axes
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Local Transcript Sum', rotation=270, labelpad=20)

        if path is not None:
            save_path = os.path.join(path, f'count_dist_contours_{self.contour_name}.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')

        if show:
            plt.show()

        return ax

class ConvolutionContours(GetContour):
    """
    Subclass for convolution-based contour generation.
    """

    def __init__(self, array_to_contour: np.ndarray, logger=None, contour_name: Optional[str] = None):
        # Initialize parent class attributes
        super().__init__(array_to_contour, logger, contour_name)
        # Initialize subclass-specific attributes
        self.local_sum_image: Optional[np.ndarray] = None

    @timeit
    def get_conv_sum(self, kernel_size: int, kernel_shape: str = 'square') -> None:
        """
        Computes the convolution sum of the array with a specified kernel.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel to be used for convolution.
        kernel_shape : str, optional
            Shape of the kernel ('square' or 'circle'), by default 'square'.
        """
        if kernel_shape not in {'square', 'circle'}:
            raise ValueError("kernel_shape must be either 'square' or 'circle'.")

        kernel = (np.ones((kernel_size, kernel_size), dtype=np.float32)
                  if kernel_shape == 'square'
                  else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))

        array_sum = np.sum(self.array, axis=2)
        # self.local_sum_image = cv2.filter2D(array_sum, -1, kernel)
        self.local_sum_image = cv2.filter2D(array_sum, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

        del array_sum

    @timeit
    def contours_from_sum(
            self, density_threshold: float, min_area_threshold: float,
            directionality: str = 'higher') -> None:
        """
       Extracts contours from the local sum image based on a density threshold and filters them by area.

       Parameters
       ----------
       density_threshold : float
           Density threshold for extracting contours.
       min_area_threshold : float
           Minimum area threshold for filtering contours.
       directionality : str, optional
           Directionality for finding contours ('higher' or 'lower'), by default 'higher'.
       """
        if self.local_sum_image is None:
            raise RuntimeError("local_sum_image is not computed. Run get_conv_sum() first.")

        if directionality == 'higher':
            binary_mask = (self.local_sum_image > density_threshold).astype(np.uint8)
        elif directionality == 'lower':
            binary_mask = (self.local_sum_image < density_threshold).astype(np.uint8)
        else:
            raise ValueError("directionality must be either 'higher' or 'lower'.")

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

        self.check_contours()
        self.filter_contours_no_counts_and_area(min_area_threshold)
        # self.logger.info(f'Contours extracted from sum after checks: {len(self.contours)}')


class KDTreeContours(GetContour):
    """
    KDTreeContours extends GetContour to analyze spatial point data using KD-tree,
    neighbors count, and derive contours via DBSCAN clustering and various hulls.
    """

    def __init__(
            self,
            kd_tree_data: Union[pd.DataFrame, np.ndarray],
            logger=None,
            contour_name: Optional[str] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
    ):
        """
        :param kd_tree_data: DataFrame or ndarray containing X, Y coordinate data
                             (must contain columns or shape accordingly)
        :param logger: optional logger for information
        :param contour_name: name of contour, used in neighbor-count column
        :param height: image height (y-max) for array creation
        :param width: image width (x-max)
        """
        # Coerce kd_tree_data to DataFrame if ndarray:
        self.radius = None
        if isinstance(kd_tree_data, np.ndarray):
            kd_tree_data = pd.DataFrame(kd_tree_data, columns=["X", "Y"])
        assert isinstance(kd_tree_data, pd.DataFrame), "kd_tree_data must be DataFrame"

        # Set attributes
        self.kd_tree_data = kd_tree_data.copy()
        self.points_x_y = self.kd_tree_data[["X", "Y"]].to_numpy()
        self.contour_name = contour_name or "contour"
        self.height = height or int(self.kd_tree_data["Y"].max())
        self.width = width or int(self.kd_tree_data["X"].max())
        self.image_size = (self.height + 1, self.width + 1)

        super().__init__(
            kd_tree_data,
            logger,
            self.contour_name,
            self.points_x_y,
        )

    @timeit
    def get_kdt_dist(self, radius:int) -> None:
        """
        Compute neighbor counts using BallTree and add column to kd_tree_data.

        This method uses BallTree to find neighbors within a specified radius
        and adds a new column to the kd_tree_data DataFrame with the count of neighbors.

        Parameters
        ----------
        radius : int
            The radius within which to count neighbors for each point.
            This is the maximum distance to consider a point as a neighbor.

        Returns
        -------
        None
        """

        self.radius = radius # max_dist
        # # Query neighbors within the radiu
        ball_tree = BallTree(self.points_x_y)
        neighbor_counts = ball_tree.query_radius(self.points_x_y, self.radius)

        self.kd_tree_data[f'{self.contour_name}_neighbor_count'] = np.array(
            [len(neighbors) for neighbors in neighbor_counts]
        )

    @timeit
    def get_neighbour_array(self) -> np.ndarray:
        """
        Construct a 2D array of neighbor counts indexed by rounded integer coordinates.
        """
        self.array_total_nei = np.zeros((self.height + 1, self.width + 1))

        # Get rounded integer indices as NumPy arrays
        x_indices = np.round(self.kd_tree_data['X']).astype(int).to_numpy()
        y_indices = np.round(self.kd_tree_data['Y']).astype(int).to_numpy()
        values = self.kd_tree_data[f'{self.contour_name}_neighbor_count'].to_numpy()
        # Assign values directly using advanced indexing
        self.array_total_nei[x_indices, y_indices] = values

        return self.array_total_nei
    def interpolate_array(self) -> np.ndarray:
        """
        Fill zeros in array_total_nei via OpenCV inpainting.
        """
        assert hasattr(self, "array_total_nei"), "Call get_neighbour_array first"

        # Convert zeros to NaN to create a mask
        mask = (self.array_total_nei == 0).astype(np.uint8)  # Mask of missing values
        self.array_total_nei = cv2.inpaint(self.array_total_nei.astype(np.float32), mask, inpaintRadius=3,
                                                 flags=cv2.INPAINT_TELEA)
        return self.array_total_nei


    # same as covcontours_from_sum. change this future version
    @timeit
    def contours_from_neighbors(
            self, density_threshold: float, min_area_threshold: float,
            directionality: str = 'higher') -> None:
        """
       Extracts contours from the local sum image based on a density threshold and filters them by area.

       Parameters
       ----------
       density_threshold : float
           Density threshold for extracting contours.
       min_area_threshold : float
           Minimum area threshold for filtering contours.
       directionality : str, optional
           Directionality for finding contours ('higher' or 'lower'), by default 'higher'.
       """
        if self.array_total_nei is None:
            raise RuntimeError("local_sum_image is not computed. Run get_conv_sum() first.")

        # check give the same name
        self.local_sum_image = self.array_total_nei.copy()
        self.array = self.array_total_nei.copy()[...,np.newaxis]  # Add a new axis to make it 3D

        if directionality == 'higher':
            binary_mask = (self.array_total_nei > density_threshold).astype(np.uint8)
        elif directionality == 'lower':
            binary_mask = (self.array_total_nei < density_threshold).astype(np.uint8)
        else:
            raise ValueError("directionality must be either 'higher' or 'lower'.")

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

        self.check_contours()
        self.filter_contours_no_counts_and_area(min_area_threshold)
        self.logger.info(f'Contours extracted from neighboor counts after checks: {len(self.contours)}')

    def find_points_with_neighoors(self, radius: float, min_neighbours: int) -> None:
        """
        Find points in the array with neighbors within a given radius.
        This method uses KDTree to efficiently find points that have a specified number of neighbors
        within a given radius.
        Parameters
        ----------
        radius : float
            The radius within which to search for neighbors.
        min_neighbours : int
            Minimum number of neighbors required for a point to be considered valid.
        Returns
        -------
        None
        """

        self.radius = radius # todo add check for array_points
        self.min_neighbours = min_neighbours

        # Initialize KDTree with the array of points
        kd_tree = KDTree(self.points_x_y)

        # Count points within the radius for each point
        counts = [len(kd_tree.query_ball_point(p, radius)) for p in self.points_x_y]
        # Filter points with more than 2 neighbors
        filtered_points = self.points_x_y[np.array(counts) > min_neighbours]
        self.logger.info("Points with more than %d neighbors: %d", min_neighbours, len(filtered_points))

        self.points_w_neig = filtered_points
        if self.points_w_neig.ndim == 3:
            self.points_w_neig = self.points_w_neig[:, :2]

        if len(filtered_points) == 0:
            self.logger.WARNING("No points with neighbors found within the given radius.")

    def label_points_with_neigbors(self) -> None:
        """
        Run DBSCAN clustering on points_w_neig.
        This method labels points with neighbors using DBSCAN clustering.
        It uses the radius and minimum number of neighbors to form clusters.
        The resulting labels are stored in self.dbscan_labels.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        # eps = 60  # Search radius (similar to the one used in KDTree)
        min_samples = max(self.min_neighbours, 2)  # Minimum number of points in a cluster
        # is going to be min_neighboors, or in case is 1, 2

        # Extract only the first two dimensions (x, y) if the points are in 3D
        # Initialize DBSCAN with the given parameters
        db = DBSCAN(eps=self.radius, min_samples=min_samples)
        self.dbscan_labels = db.fit_predict(self.points_w_neig)
        self.logger.info("Points w/ neig agglomerated in DBSCAN labels: %d", len(self.dbscan_labels))

    def contours_from_kd_tree_simple_circle(self) -> None:
        """
        Draw circular hull around each DBSCAN cluster.
        This method generates contours by creating circles around each cluster of points identified by DBSCAN.
        Each circle is centered at the centroid of the cluster and has a radius that encompasses all points in the cluster.
        """
        contours_list = []
        unique_labels = set(self.dbscan_labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # Select points in this cluster
            cluster_points = self.points_w_neig[self.dbscan_labels == label]

            # Compute the centroid of the cluster
            centroid = np.mean(cluster_points, axis=0)

            # Compute the maximum distance from the centroid to any point in the cluster
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            radius = np.max(distances)  # This ensures all points are within the circle

            # Ensure the minimum radius is 20 pixels (minimum diameter of 40)
            radius = max(radius, self.radius // 2)  # todo problem here!

            # Create an image to draw the circle and convert it to a contour
            circle_image = np.zeros(self.image_size, dtype=np.uint8)
            center = (int(centroid[1]), int(centroid[0]))  # Convert to (x, y) format for OpenCV
            cv2.circle(circle_image, center, int(radius), (255), thickness=-1)  # Fill the circle

            # Find contours from the circle mask
            contours, _ = cv2.findContours(circle_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_list.append(contours[0])  # Only need the external contour

        self.contours = contours_list
        self.logger.info("N contours: %d", len(self.contours))
        del contours_list

    def contours_from_kd_tree_concave_hull(self, alpha: float = 0.1) -> None:
        """
        Generate contours using concave hulls (alpha shapes).
        Still in development, not working properly.

        Parameters
        ----------
        alpha : float
            Alpha parameter for the alpha shape. Smaller values make the hull tighter, while larger
            values make it looser (closer to a convex hull).

        Returns
        -------
        None
        """
        alpha = max(0.05, max(1.0, len(self.points_w_neig) / 1000))
        alpha = 10000
        print(alpha)

        contours_list = []
        unique_labels = set(self.dbscan_labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # Select points in this cluster
            cluster_points = self.points_w_neig[self.dbscan_labels == label]

            if len(cluster_points) < 3:
                continue  # Skip clusters with fewer than 3 points

            # Compute the concave hull using alphashape
            concave_hull = alphashape.alphashape(cluster_points, alpha)

            # Ensure the concave hull is valid
            if isinstance(concave_hull, Polygon):
                # Extract exterior coordinates of the hull as a contour
                contour = np.array(concave_hull.exterior.coords)
                contours_list.append(contour)

        # Store the contours and log
        self.contours = contours_list
        self.logger.info("Generated %d concave hull contours", len(self.contours))

    def contours_from_kd_tree_complex_hull(self) -> None:
        """
        Draw convex hulls around clusters using scipy.ConvexHull.
        Still in development, not working properly.
        """
        # todo not working properly
        contours_list = []
        unique_labels = set(self.dbscan_labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # Select points in this cluster
            cluster_points = self.points_w_neig[self.dbscan_labels == label]

            ## Handle clusters with fewer than 3 points
            if len(cluster_points) < 3:
                self.logger.warning("Cluster with label %d has less than 3 points; skipping.", label)
                continue

            try:
                # Compute the convex hull for the points
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]  # Get the points on the hull

                # Ensure the contour is closed by adding the first point to the end
                hull_points = np.vstack([hull_points, hull_points[0]])

                # Convert the points into an OpenCV-compatible format
                contour = hull_points.astype(np.int32)

                # Add the contour to the list
                contours_list.append(contour)

            except Exception as e:
                self.logger.error("Error processing cluster %d: %s", label, str(e))
                continue

        self.contours = contours_list
        self.logger.info("N contours: %d", len(self.contours))

    # def refine_contours(self):
    #     pass

    def get_contours_around_points_with_neighboors(self, type_contouring : str ='simple_circle') -> None:
        """
            Get contours around points with neighbors using KDTree and DBSCAN.

            This method performs the following steps:
            1. Label close points using DBSCAN clustering.
            2. Create contours based on the labeled points using different contouring methods.
            3. Check the validity of the contours and store the count of valid contours.

            Parameters
            ----------
            type_contouring : str, default 'simple_circle'
                Type of contouring to use. Options are:
                - 'simple_circle': draw circles around clusters.
                - 'complex_hull': draw convex hulls around clusters.
                - 'concave_hull': draw concave hulls (alpha shapes) around clusters.

            Returns
            -------
            None
            """

        self.label_points_with_neigbors()         # 2. Label close points with DBSCAN

        # 3. Create contours from the labeled points
        if type_contouring == 'simple_circle':
            self.contours_from_kd_tree_simple_circle()
        elif type_contouring == 'complex_hull':
            self.contours_from_kd_tree_complex_hull()
        elif type_contouring == 'concave_hull':
            self.contours_from_kd_tree_concave_hull()
        # 4. Check contours
        self.check_contours()
        self.total_valid_contours = len(self.contours)

    def plot_point_clusters_with_contours(self,
                                          show: bool = False,
                                          figsize: Tuple = (10, 10)) -> plt.Figure:
        """
        Plot DBSCAN-derived clusters and their contour boundaries.

        This method plots each contour in `self.contours` and overlays the clustered points
        (from `self.points_w_neig` and `self.dbscan_labels`). Contours are expected to be arrays
        of shape (N, 1, 2) or (N, 2); the singleton middle dimension is squeezed out automatically.

        Parameters
        ----------
        show : bool, default=False
            If True, the plot is immediately displayed via `plt.show()`. If False, the Figure
            object is returned for further manipulation or testing, and no immediate `show()` is called.
        figsize : tuple of int (width, height), default=(10, 10)
            Size of the figure in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The created Figure object containing the cluster and contour plot.
            If `show=True`, the figure is still returned after display.

        Raises
        ------
        AttributeError
            If `self.contours`, `self.points_w_neig`, or `self.dbscan_labels` are not set prior to calling.
        ValueError
            If any contour cannot be interpreted as a sequence of 2D points after squeezing.

        Notes
        -----
        - Expects:
          - `self.contours`: sequence of NumPy arrays, each representing a contour of shape (N, 1, 2) or (N, 2).
          - `self.points_w_neig`: NumPy array of shape (M, 2) containing clustered point coordinates.
          - `self.dbscan_labels`: 1D array of length M containing integer labels from DBSCAN.
        - Points with label `-1` (noise) are skipped.
        - Coordinate convention: plots use x = arr[:, 0], y = arr[:, 1]. Adjust if your data uses reversed axes.

        """

        if not hasattr(self, "contours"):         # Validate that required attributes exist
            raise AttributeError("`self.contours` is not defined. Run contour-generation first.")
        if not hasattr(self, "points_w_neig") or not hasattr(self, "dbscan_labels"):
            raise AttributeError("`self.points_w_neig` or `self.dbscan_labels` not defined. Run DBSCAN steps first.")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each contour
        for idx, contour in enumerate(self.contours):
            # contour may have shape (N, 1, 2) or (N, 2). Squeeze the singleton dim if present.
            arr = contour
            try:
                # If shape is (N, 1, 2), squeeze middle dimension
                if arr.ndim == 3 and arr.shape[1] == 1:
                    arr = arr.squeeze(1)  # now (N, 2)
            except Exception:
                raise ValueError(f"Contour at index {idx} has unexpected shape {contour.shape}")

            if arr.ndim != 2 or arr.shape[1] != 2:
                # Skip or raise error; here we choose to skip with a warning
                # You may replace with: raise ValueError(...)
                import warnings
                warnings.warn(f"Skipping contour at index {idx}: expected shape (N,2) after squeeze, got {arr.shape}")
                continue

            # Plot: x = arr[:, 0], y = arr[:, 1]
            ax.plot(arr[:, 0], arr[:, 1], color='blue', linestyle='-', alpha=0.7, label="_nolegend_")

        # Plot cluster points (skip noise label = -1)
        labels = self.dbscan_labels
        pts = self.points_w_neig
        # Optionally, you could collect unique labels except -1:
        unique_labels = sorted(set(labels) - {-1})
        for label in unique_labels:
            mask = (labels == label)
            cluster_points = pts[mask]
            if cluster_points.size == 0:
                continue
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       label=f"Cluster {label}", alpha=0.7, s=10)

            # If desired, plot centroids:
            # centroid = cluster_points.mean(axis=0)
            # ax.scatter(centroid[0], centroid[1], color='red', marker='x',
            #            label=f"Centroid {label}")

        ax.set_title("Clusters with Boundaries (Contours)")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        # You can enable legend if desired:
        # ax.legend(loc='best', fontsize='small')

        if show:
            plt.show()

        return plt

    def plot_dbscan_labels(
            self,
            show: bool = True,
            figsize: Tuple[int, int] = (8, 8)
    ) -> Figure:
        """
        Plot points colored by DBSCAN labels (clusters vs noise).

        This method creates a scatter plot of the points in `self.points_w_neig`, coloring each
        point according to its label in `self.dbscan_labels`. Noise points (label = -1) are plotted
        with a distinct marker/style.

        Parameters
        ----------
        show : bool, default=True
            If True, display the plot immediately with `plt.show()`.
            If False, return the Figure object for further manipulation or testing without showing it.
        figsize : tuple of int (width, height), default=(8, 8)
            Size of the figure in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The created Figure containing the DBSCAN label plot. Returned even if `show=True`.

        Raises
        ------
        AttributeError
            If `self.points_w_neig` or `self.dbscan_labels` are not present.
        ValueError
            If `self.points_w_neig` and `self.dbscan_labels` have incompatible lengths or invalid shapes.

        Notes
        -----
        - Expects:
            - `self.points_w_neig`: NumPy array of shape (N, 2), the points used in DBSCAN.
            - `self.dbscan_labels`: 1D array of length N, integer labels from DBSCAN.
        - Points labeled `-1` are treated as noise and plotted differently (marker 'x').
        - Uses the `tab20` colormap to assign distinct colors to each cluster label.
        - Adds a legend, grid, and axis labels.
        - Returns the Figure to facilitate testing (e.g., `isinstance(fig, Figure)`) and saving (e.g., `fig.savefig(...)`).
        """
        # Validate attributes exist
        if not hasattr(self, "points_w_neig") or not hasattr(self, "dbscan_labels"):
            raise AttributeError(
                "`self.points_w_neig` and `self.dbscan_labels` must be set before calling plot_dbscan_labels.")

        points = self.points_w_neig
        labels = self.dbscan_labels

        # Basic shape/length check
        if not hasattr(points, "shape") or points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(
                f"`self.points_w_neig` must be an array of shape (N, 2); got shape {getattr(points, 'shape', None)}")
        if labels.shape[0] != points.shape[0]:
            raise ValueError(f"Length mismatch: points length {points.shape[0]}, labels length {labels.shape[0]}")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Determine unique labels
        unique_labels = sorted(set(labels))
        # Use tab20 colormap to get distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = (labels == label)
            pts = points[mask]
            if pts.size == 0:
                continue

            if label == -1:
                # Noise points: marker 'x'
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    c=[color], label="Noise", marker="x", s=20, alpha=0.6
                )
            else:
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    c=[color], label=f"Cluster {label}", marker="o", s=20, alpha=0.6
                )

        ax.set_title("DBSCAN Clusters and Noise Points")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True)

        if show:
            plt.show()

        return plt