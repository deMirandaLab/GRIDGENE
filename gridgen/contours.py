import logging
import cv2
import numpy as np
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
from typing import Optional, Tuple, Dict, Any, List
from matplotlib.axes import Axes
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
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

    def __init__(self, array_to_contour, logger=None, contour_name=None):
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
        """
        self.array = array_to_contour
        self.contours = None
        self.contour_name = contour_name
        self.total_valid_contours = 0
        self.contours_filtered_area = 0
        self.logger = logger or get_logger(f'{__name__}.{contour_name or "GetContour"}')
        self.logger.info("Initialized GetContour")

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
        array2d = np.sum(self.array, axis=2)

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


######## TODO testing and cleaning here.
class KDTreeContours(GetContour):

    def __init__(self, kd_tree_data, logger=None, contour_name=None):

        # Initialize parent class attributes
        super().__init__(kd_tree_data, logger, contour_name)
        # Initialize subclass-specific attributes
        self.kd_tree_data = kd_tree_data
        self.radius = None

        # todo KD tree data is an array. but it could be a df. change this
        self.image_size = (kd_tree_data.shape[0], kd_tree_data.shape[1])
        self.points = np.argwhere(kd_tree_data == 1)  # Extract indices where value is 1

    def find_points_with_neighoors(self, radius, min_neighbours):
        """
        Find points in the array with neighbors within a given radius.

        Parameters
        ----------
        array_points : np.ndarray
            Array of points to search for neighbors.
        radius : float
            Radius in pixels within which to search for neighbors.

        Returns
        -------
        np.ndarray
            Array of points with neighbors within the given radius.
        """
        # todo add check for array_points

        # Initialize KDTree with the array of points
        kd_tree = KDTree(self.points)

        # Count points within the radius for each point
        counts = [len(kd_tree.query_ball_point(p, radius)) for p in self.points]
        # Filter points with more than 2 neighbors
        filtered_points = self.points[np.array(counts) > min_neighbours]
        self.logger.info("Points with more than %d neighbors: %d", min_neighbours, len(filtered_points))

        self.points_w_neig = filtered_points
        if self.points_w_neig.ndim == 3:
            self.points_w_neig = self.points_w_neig[:, :2]

        if len(filtered_points) == 0:
            self.logger.WARNING("No points with neighbors found within the given radius.")

    def label_points_with_neigbors(self):
        # eps = 60  # Search radius (similar to the one used in KDTree)
        min_samples = max(self.min_neighbours, 2)  # Minimum number of points in a cluster
        # is going to be min_neighboors, or in case is 1, 2

        # Extract only the first two dimensions (x, y) if the points are in 3D
        # Initialize DBSCAN with the given parameters
        db = DBSCAN(eps=self.radius, min_samples=min_samples)
        self.dbscan_labels = db.fit_predict(self.points_w_neig)
        self.logger.info("Points w/ neig agglomerated in DBSCAN labels: %d", len(self.dbscan_labels))

    def contours_from_kd_tree_simple_circle(self):
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

    def contours_from_kd_tree_concave_hull(self, alpha=0.1):
        """
        Generate contours using concave hulls (alpha shapes).

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

    def contours_from_kd_tree_complex_hull(self):
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

    def refine_contours(self):
        pass

    def get_contours(self, radius, min_neighbours, type_contouring='simple_circle'):
        self.radius = radius
        self.min_neighbours = min_neighbours  # number of neighboors of each point to be considered. > than.
        # FOr GD I just want 1 point

        # 1 find points that have neighboors in a radius of N pixels from the kd_tree_data
        self.find_points_with_neighoors(radius, min_neighbours)

        # if self.points_w_neig.ndim == 3:
        # self.points_w_neig = self.points_w_neig[:, :2]
        # self.points_w_neig = self.points_w_neig[:, 1:3]
        # 2. Label close points with DBSCAN
        self.label_points_with_neigbors()

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

    def plot_point_clusters_with_contours(self, show=False, figsize=(10, 10)):
        # todo change plot to be in accordance. axes, fig size ...
        # Step 2: Plotting the clusters and contours
        plt.figure(figsize=figsize)

        # Plot each contour
        for contour in self.contours:
            plt.plot(contour[:, 0], contour[:, 1], 'b-', alpha=0.7)  # Plot each contour in blue

        # Optionally, plot the filtered points and centroids
        for label in self.dbscan_labels:
            if label == -1:
                continue

            cluster_points = self.points_w_neig[self.dbscan_labels == label]
            plt.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f"Cluster {label}", alpha=0.7)

            # # Plot the centroid
            # centroid = np.mean(cluster_points, axis=0)
            # plt.scatter(centroid[1], centroid[0], color='red', marker='x', label=f"Centroid {label}")

        plt.title("Clusters with Boundaries (Contours)")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # plt.legend(loc='upper right')
        if show:
            plt.show()
        else:
            return plt

    def plot_dbscan_labels(self, show=True, figsize=(8, 8)):
        """
        Plot points colored by DBSCAN labels.

        Parameters:
        - points (np.ndarray): Array of point coordinates (N x 2).
        - labels (np.ndarray): Array of DBSCAN labels for each point.
        - figsize (tuple): Figure size for the plot.
        - show (bool): Whether to display the plot or return the plot object.
        """
        labels = self.dbscan_labels
        points = self.points_w_neig
        unique_labels = set(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        plt.figure(figsize=figsize)
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                plt.scatter(points[labels == label][:, 0], points[labels == label][:, 1],
                            c=[color], label="Noise", marker="x", s=5, alpha=0.6)
            else:
                # Cluster points
                plt.scatter(points[labels == label][:, 0], points[labels == label][:, 1],
                            c=[color], label=f"Cluster {label}", s=5, alpha=0.6)

        plt.title("DBSCAN Clusters and Noise Points")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        else:
            return plt
