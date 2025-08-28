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
from gridgene.logger import get_logger
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

class GetContour:
    """
    Parent class for contour handling and filtering from a 3D array.

    This class handles extraction and filtering of contours from a 3D array where the first two
    dimensions represent spatial coordinates (x and y), and the third dimension contains gene-specific values.

    Attributes
    ----------
    array : np.ndarray
        The 3D array from which contours are to be extracted. The first two dimensions are x and y
        spatial positions; the third dimension corresponds to genes.
    local_sum_image : np.ndarray
        The 2D array representing the local sum of the input array.
    contours : list of np.ndarray
        List of contours extracted from the array.
    contour_name : str
        Name of the contour for identification.
    total_valid_contours : int
        Total number of valid contours after filtering.
    contours_filtered_area : int
        Number of contours remaining after area filtering.
    logger : logging.Logger
        Logger instance for logging information and errors.
    points_x_y : np.ndarray, optional
        Optional 2D array of shape (N, 2) containing (x, y) points for plotting or further analysis.

    Methods
    -------
    __init__(array_to_contour, logger=None, contour_name=None, points_x_y=None)
        Initializes the GetContour object.
    check_contours()
        Validates and closes contours.
    filter_contours_area(min_area_threshold)
        Filters contours based on a minimum area threshold.
    filter_contours_no_counts()
        Filters out contours with no transcript counts in the array.
    filter_contours_by_gene_threshold(gene_array, threshold, gene_name="")
        Filters contours based on gene-specific thresholds.
    filter_contours_by_gene_comparison(gene_array1, gene_array2, gene_name1="", gene_name2="")
        Filters contours where gene_array1 has higher signal than gene_array2.
    plot_contours_scatter(...)
        Creates a scatter plot with overlaid contours.
    plot_conv_sum(...)
        Plots the convolution sum image with contours.
    """

    def __init__(self, array_to_contour, logger=None, contour_name=None, points_x_y: np.ndarray = None):
        """
        Initialize the contour handler with a 3D input array.

        Parameters
        ----------
        array_to_contour : np.ndarray
            3D array from which contours are to be derived.
        logger : logging.Logger, optional
            Optional logger instance; a default logger will be used if None.
        contour_name : str, optional
            Optional name for this set of contours.
        points_x_y : np.ndarray, optional
            Optional array of (x, y) positions for scatter plotting.
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
        Validate and clean contour data.

        Ensures each contour has at least 3 points, closes open contours,
        and converts them to integer format for OpenCV compatibility.
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
        Filter contours by minimum area.

        Parameters
        ----------
        min_area_threshold : float
            Contours with area below this threshold are discarded.
        """
        self.contours = [contour for contour in self.contours if cv2.contourArea(contour) >= min_area_threshold]
        # self.contours_filtered_area = len(self.contours)
        self.logger.info(f'Number of contours after area filtering: {len(self.contours)}')

    def filter_contours_no_counts(self) -> List[np.ndarray]:
        """
        Remove contours that do not contain any non-zero values in the array.

        Returns
        -------
        List[np.ndarray]
            Contours that contain non-zero signal in the original array.
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
        Filter contours based on both signal presence and minimum area.

        Parameters
        ----------
        min_area_threshold : float
            Minimum area a contour must have to be retained.

        Returns
        -------
        List[np.ndarray]
            Valid contours satisfying both count and area criteria.
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
        Retain contours where the gene signal meets a minimum threshold.

        Parameters
        ----------
        gene_array : np.ndarray
            2D or 3D array of gene expression values.
        threshold : float
            Minimum gene signal required inside the contour.
        gene_name : str, optional
            Optional name of the gene, used for logging.
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
        Filter contours by comparing signal from two gene arrays.

        Only retain contours where the first gene has higher counts than the second.

        Parameters
        ----------
        gene_array1 : np.ndarray
            First gene array.
        gene_array2 : np.ndarray
            Second gene array.
        gene_name1 : str, optional
            Name for the first gene (used in logging).
        gene_name2 : str, optional
            Name for the second gene (used in logging).
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
        Plot a scatter plot of spatial points overlaid with contours.

        Parameters
        ----------
        path : str, optional
            Directory where the plot will be saved (if specified).
        show : bool
            Whether to display the plot interactively.
        s : float
            Size of scatter points.
        alpha : float
            Transparency of scatter points.
        linewidth : float
            Width of contour lines.
        c_points : str
            Color for scatter points.
        c_contours : str
            Color for contour lines.
        figsize : Tuple[int, int]
            Size of the figure.
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot; if None, a new figure is created.
        **kwargs : dict
            Additional keyword arguments for customizing scatter and line plots.

        Returns
        -------
        Axes
            The matplotlib Axes object used for plotting.
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
        Plot the local sum (convolution) image with contour overlays.

        Parameters
        ----------
        cmap : str
            Colormap for the local sum image.
        c_countour : str
            Color for overlaying contours.
        path : str, optional
            Path to save the figure (if specified).
        show : bool
            Whether to display the plot interactively.
        figsize : Tuple[int, int]
            Size of the figure.
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot; if None, a new figure is created.

        Returns
        -------
        Axes
            The matplotlib Axes object used for plotting.
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
    A subclass of GetContour for generating contours based on a convolution
    of the input array. This class provides methods for computing a local
    density map and extracting contours based on intensity thresholds.

    Attributes
    ----------
    local_sum_image : np.ndarray or None
        2D array containing the result of convolution on the input data,
        used as a basis for contour detection.
    """

    def __init__(self, array_to_contour: np.ndarray, logger=None, contour_name: Optional[str] = None):
        """
        Initialize the ConvolutionContours instance.

        Parameters
        ----------
        array_to_contour : np.ndarray
            3D input array used to compute the convolution-based contours.
        logger : logging.Logger, optional
            Optional logger instance for debugging and logging purposes.
        contour_name : str, optional
            Optional identifier for this contour set.
        """
        # Initialize parent class attributes
        super().__init__(array_to_contour, logger, contour_name)
        # Initialize subclass-specific attributes
        self.local_sum_image: Optional[np.ndarray] = None

    @timeit
    def get_conv_sum(self, kernel_size: int, kernel_shape: str = 'square') -> None:
        """
        Compute a 2D convolution sum across the 3D input array to create a density map.

        Parameters
        ----------
        kernel_size : int
            The size of the convolution kernel.
        kernel_shape : str, optional
            Shape of the kernel: either 'square' or 'circle'. Defaults to 'square'.

        Raises
        ------
        ValueError
            If `kernel_shape` is not one of {'square', 'circle'}.
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
            self,
            density_threshold: float,
            min_area_threshold: float,
            directionality: str = 'higher'
    ) -> None:
        """
        Generate contours from the convolution sum using a threshold, and filter by area.

        Parameters
        ----------
        density_threshold : float
            The threshold applied to the convolution image to create a binary mask.
        min_area_threshold : float
            Minimum area that a contour must have to be retained.
        directionality : str, optional
            Direction of thresholding:
            - 'higher': select pixels greater than the threshold
            - 'lower': select pixels less than the threshold
            Default is 'higher'.

        Raises
        ------
        RuntimeError
            If the convolution sum image (`local_sum_image`) has not been computed.
        ValueError
            If `directionality` is not 'higher' or 'lower'.
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
    A subclass of GetContour for generating contours from spatial point data
    using KD-tree or BallTree-based neighbor counts, clustering, and geometric hulls.

    This class supports estimating local point densities via neighbor search,
    extracting density-based arrays, applying clustering (e.g., DBSCAN),
    and generating contours from those clusters using circle or concave hulls.

    Attributes
    ----------
    kd_tree_data : pd.DataFrame
        Input coordinate data, with 'X' and 'Y' columns.
    points_x_y : np.ndarray
        Array of shape (n_samples, 2) containing the input (X, Y) coordinates.
    height : int
        Height of the image space (used to define output array size).
    width : int
        Width of the image space.
    image_size : tuple
        Tuple of (height+1, width+1) defining the output array dimensions.
    radius : float
        Radius used for neighborhood calculations and clustering.
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
        Initialize the KDTreeContours instance with spatial point data.

        Parameters
        ----------
        kd_tree_data : Union[pd.DataFrame, np.ndarray]
            Input data with spatial X, Y coordinates. If ndarray, it must be of shape (n, 2).
        logger : logging.Logger, optional
            Optional logger for debugging or information output.
        contour_name : str, optional
            Optional name for labeling neighbor count columns and logs.
        height : int, optional
            Optional height (Y-extent) of the output image grid. Defaults to max Y in data.
        width : int, optional
            Optional width (X-extent) of the output image grid. Defaults to max X in data.
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
    def get_kdt_dist(self, radius: int) -> None:
        """
        Compute the number of neighbors for each point using BallTree within a given radius.

        The neighbor count is stored in a new column of `kd_tree_data` named
        '{contour_name}_neighbor_count'.

        Parameters
        ----------
        radius : int
            Search radius used to define neighborhood around each point.
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
        Create a 2D array where each pixel value corresponds to the neighbor count
        at the rounded integer (X, Y) location of the points.

        Returns
        -------
        np.ndarray
            2D array with shape (height+1, width+1) where each pixel represents
            the number of neighbors for that spatial location.
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
        Fill zero values in the neighbor count array using OpenCV inpainting.

        This helps in smoothing sparse or missing regions in the data grid.

        Returns
        -------
        np.ndarray
            Inpainted version of `array_total_nei`.
        """
        assert hasattr(self, "array_total_nei"), "Call get_neighbour_array first"

        # Convert zeros to NaN to create a mask
        mask = (self.array_total_nei == 0).astype(np.uint8)  # Mask of missing values
        self.array_total_nei = cv2.inpaint(self.array_total_nei.astype(np.float32), mask, inpaintRadius=3,
                                           flags=cv2.INPAINT_TELEA)
        return self.array_total_nei

    # same as covcontours_from_sum. change this future version
    @timeit
    def contours_from_neighbors(self, density_threshold: float, min_area_threshold: float,
                                directionality: str = 'higher') -> None:
        """
        Extract contours from a local sum image using a density threshold.

        Parameters
        ----------
        density_threshold : float
            Density threshold for extracting contours.
        min_area_threshold : float
            Minimum area threshold to keep a contour.
        directionality : str, optional
            Direction to threshold ('higher' or 'lower'), by default 'higher'.

        Returns
        -------
        None
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
        Find points with neighbors within a given radius using KDTree.

        Parameters
        ----------
        radius : float
            Search radius to find neighbors around each point.
        min_neighbours : int
            Minimum number of neighbors for a point to be considered valid.

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
        Label clustered points with DBSCAN based on neighbor relationships.

        Uses the radius and minimum number of neighbors to identify point clusters.

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
        Create contours using circular masks centered on DBSCAN clusters.

        A circle is drawn around each cluster, with a radius covering all points.

        Returns
        -------
        None
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
        Create contours using concave hulls (alpha shapes) on DBSCAN clusters.

        Parameters
        ----------
        alpha : float
            Alpha parameter controlling the shape tightness. Smaller values yield tighter shapes.

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
        Create contours using convex hulls from DBSCAN clusters.

        Uses scipy.spatial.ConvexHull to wrap cluster points.

        Returns
        -------
        None
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

    def get_contours_around_points_with_neighboors(self, type_contouring: str = 'simple_circle') -> None:
        """
        Generate contours around points with neighbors using KDTree + DBSCAN.

        Parameters
        ----------
        type_contouring : str, default='simple_circle'
            Method for generating contours:
            - 'simple_circle' : Circles around clusters.
            - 'complex_hull'  : Convex hulls around clusters.
            - 'concave_hull'  : Concave hulls (alpha shapes) around clusters.

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

    def plot_point_clusters_with_contours(self, show: bool = False, figsize: Tuple = (10, 10)) -> plt.Figure:
        """
        Plot DBSCAN clusters with overlayed contour boundaries.

        Parameters
        ----------
        show : bool, default=False
            Whether to call `plt.show()` immediately.
        figsize : tuple, default=(10, 10)
            Size of the plot in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object for the plot.
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

    def plot_dbscan_labels(self, show: bool = True, figsize: Tuple[int, int] = (8, 8)) -> Figure:
        """
        Plot points colored by DBSCAN cluster labels.

        Parameters
        ----------
        show : bool, default=True
            Whether to call `plt.show()` immediately.
        figsize : tuple of int, default=(8, 8)
            Size of the figure in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object for the plot.
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