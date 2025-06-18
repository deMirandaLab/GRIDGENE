import logging
import cv2
import numpy as np
import os
import matplotlib # added for docs generation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.axes
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from typing import Dict, List, Tuple, Union
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from gridgen.logger import get_logger
from typing import Optional, Tuple, Dict, Any, List
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import cv2
import numpy as np
from shapely.geometry import Polygon, box

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

class GetMasks:
    """
    Class to handle mask processing operations such as filtering, creation, morphology, subtraction, saving, and plotting.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, a default logger is configured.
    image_shape : tuple of int, optional
        Tuple representing the shape of the image (height, width).
    """

    def __init__(self, logger: Optional[logging.Logger] = None, image_shape: Optional[Tuple[int, int]] = None):
        """
        Initialize the GetMasks class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for logging messages. If None, a default logger is created.
        image_shape : tuple of int, optional
            Tuple representing the shape of the image (height, width).

        Returns
        -------
        None
        """
        self.image_shape = image_shape
        self.height = self.image_shape[0] if self.image_shape is not None else None
        self.width = self.image_shape[1] if self.image_shape is not None else None
        self.logger = logger or get_logger(f'{__name__}.{"GetMasks"}')
        self.logger.info("Initialized GetMasks")

    def filter_binary_mask_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Remove small connected components from a binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask (0 or 1).
        min_area : int
            Minimum area threshold.

        Returns
        -------
        np.ndarray
            Filtered binary mask.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

        output_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output_mask[labels == i] = 1
        return output_mask

    def filter_labeled_mask_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Filter a labeled mask by keeping only components with area >= min_area.

        Parameters
        ----------
        mask : np.ndarray
            Input labeled mask (integer labels).
        min_area : int
            Minimum area threshold.

        Returns
        -------
        np.ndarray
            Filtered labeled mask preserving label IDs.
        """
        mask = mask.astype(np.int32)
        unique_labels, counts = np.unique(mask, return_counts=True)
        labels_to_keep = unique_labels[(counts >= min_area) & (unique_labels != 0)]

        filtered_mask = np.zeros_like(mask, dtype=np.int32)
        for label in labels_to_keep:
            filtered_mask[mask == label] = label

        # if logger:
        self.logger.info(f'Filtered labeled mask by area >= {min_area}, kept {len(labels_to_keep)} components.')

        return filtered_mask

    def create_mask(self, contours: List[np.ndarray]) -> np.ndarray:
        """
        Create a binary mask from contours.

        Parameters
        ----------
        contours : list of np.ndarray
            List of contours.

        Returns
        -------
        np.ndarray
            Binary mask.

        Raises
        ------
        ValueError
            If image shape is not defined.
        """
        if self.height is None or self.width is None:
            raise ValueError("Image shape must be defined to create mask.")
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, color=1, thickness=cv2.FILLED)
        return mask

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes inside a binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.

        Returns
        -------
        np.ndarray
            Hole-filled binary mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, color=1, thickness=cv2.FILLED)
        return filled_mask

    def apply_morphology(self, mask: np.ndarray, operation: str = "open", kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological operations to a binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask to process.
        operation : str, optional
            Morphological operation: "open", "close", "erode", or "dilate" (default is "open").
        kernel_size : int, optional
            Size of the structuring element (default is 3).

        Returns
        -------
        np.ndarray
            Processed binary mask.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == "open":
            result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == "erode":
            result = cv2.erode(mask, kernel, iterations=1)
        elif operation == "dilate":
            result = cv2.dilate(mask, kernel, iterations=1)
        else:
            self.logger.warning(f"Unknown morphological operation '{operation}', returning original mask.")
            result = mask

        self.logger.info(f'Applied morphology operation "{operation}" with kernel size {kernel_size}.')
        return result

    def subtract_masks(self, base_mask: np.ndarray, *masks: np.ndarray) -> np.ndarray:
        """
        Subtract one or more masks from a base mask.

        Parameters
        ----------
        base_mask : np.ndarray
            Initial binary mask.
        *masks : np.ndarray
            Masks to subtract from the base mask.

        Returns
        -------
        np.ndarray
            Resulting mask after subtraction.
        """
        result_mask = base_mask.copy()
        for mask in masks:
            result_mask = cv2.subtract(result_mask, mask)
        self.logger.info(f'Subtracted masks from base mask.')
        return result_mask

    def save_masks_npy(self, mask: np.ndarray, save_path: str) -> None:
        """
        Save mask as a .npy file.

        Parameters
        ----------
        mask : np.ndarray
            Mask to save.
        save_path : str
            Path to save the .npy file.

        Returns
        -------
        None
        """
        np.save(save_path, mask)
        self.logger.info(f'Mask saved at {save_path}')

    def save_masks(self, mask: np.ndarray, path: str) -> None:
        """
        Save mask as an image file.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask to save.
        path : str
            Path to save the image file.

        Returns
        -------
        None
        """
        cv2.imwrite(path, mask * 255)
        self.logger.info(f'Mask saved at {path}')

    def plot_masks(
            self,
            masks: List[np.ndarray],
            mask_names: List[str],
            background_color: Tuple[int, int, int] = (0, 0, 0),
            mask_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
            path: Optional[str] = None,
            show: bool = True,
            ax: Optional[plt.Axes] = None,
            figsize: Tuple[int, int] = (10, 10)
    ) -> None:
        """
        Plot multiple masks with their corresponding names.

        Parameters
        ----------
        masks : list of np.ndarray
            List of masks to plot.
        mask_names : list of str
            Names corresponding to each mask.
        background_color : tuple of int, optional
            RGB color tuple for background areas (default (0, 0, 0)).
        mask_colors : dict, optional
            Mapping of mask names to RGB colors.
        path : str, optional
            Directory path to save the plot image.
        show : bool, optional
            Whether to display the plot (default True).
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to plot on. Creates new figure if None.
        figsize : tuple of int, optional
            Size of the figure in inches (width, height).

        Returns
        -------
        None
        """
        if len(masks) != len(mask_names):
            self.logger.error('The number of masks and mask names must be the same.')
            return

        # Create a background image filled with the background color
        background = np.full((self.height, self.width, 3), background_color)

        # Create a list to store the patches for the legend
        legend_patches = []

        # Choose a colormap based on the number of masks
        colormap = cm.get_cmap('tab10') if len(masks) <= 10 else cm.get_cmap('tab20')

        # Add each mask to the background image
        for i, (mask, mask_name) in enumerate(zip(masks, mask_names)):
            # Choose a color for the mask
            if mask_colors and mask_name in mask_colors:
                mask_color = np.array(mask_colors[mask_name])
            else:
                mask_color = (np.array(colormap(i % colormap.N)[:3]) * 255).astype(int)
            # Apply the mask color to the mask image
            background[mask!=0] = mask_color

            # Create a patch for the legend
            legend_patches.append(mpatches.Patch(color=mask_color / 255, label=mask_name))

        # Flip the mask horizontally and rotate 90 degrees clockwise
        background = np.fliplr(background)
        background = np.rot90(background, k=1)
        created_fig = False
        if ax is None:
            created_fig = True
            fig, ax = plt.subplots(figsize=figsize)

        # Plot the background image
        ax.imshow(background, origin='lower')
        ax.set_axis_off()

        # Add legend
        ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            bbox_transform=ax.transAxes
        )

        # Save the image if path is provided
        if path is not None:
            save_path = os.path.join(
                path,
                f'masks_{"_".join(mask_names).replace(" ", "").lower()}.png'
            )
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        # Show the plot if required
        if show:
            plt.show()
            plt.close()

        # Close the figure if it was created within this function
        if created_fig:
            plt.close(fig)

# CancerStromaInterfaceanalysis
class ConstrainedMaskExpansion(GetMasks):
    """
    Class for expanding a seed mask with constraints, generating binary, labeled, and referenced expansions.
    """

    def __init__(
            self,
            seed_mask: np.ndarray,
            constraint_mask: Optional[np.ndarray] = None,
            logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the ConstrainedMaskExpansion object.

        Parameters
        ----------
        seed_mask : np.ndarray
            Binary seed mask to expand (non-zero labeled regions).
        constraint_mask : np.ndarray, optional
            Binary mask to limit the expansion area. If None, no constraint is applied.
        logger : logging.Logger, optional
            Logger instance for logging messages.

        Raises
        ------
        ValueError
            If seed_mask is None.
        """
        if seed_mask is None:
            raise ValueError("Seed mask cannot be None.")

        self.seed_mask_raw = seed_mask.astype(np.uint8)
        self.seed_mask = label(self.seed_mask_raw)  # connected components
        self.constraint_mask = (
            constraint_mask.astype(np.uint8)
            if constraint_mask is not None
            else np.ones_like(seed_mask, dtype=np.uint8)
        )

        image_shape = self.seed_mask.shape
        super().__init__(logger=logger, image_shape=image_shape)

        self.binary_expansions: Dict[str, np.ndarray] = {}
        self.labeled_expansions: Dict[str, np.ndarray] = {}
        self.referenced_expansions: Dict[str, np.ndarray] = {}

    def expand_mask(
        self,
        expansion_pixels: List[int],
        min_area: Optional[int] = None,
        restrict_to_limit: bool = True,
    ) -> None:
        """
        Expand the seed mask outward by specified pixel distances with optional area filtering and constraints.

        Parameters
        ----------
        expansion_pixels : list of int
            List of expansion distances (in pixels) from the seed mask.
        min_area : int, optional
            Minimum area threshold for keeping connected components in each expansion ring.
        restrict_to_limit : bool, optional
            If True, limit the expansion within the constraint mask.

        Returns
        -------
        None
        """
        sorted_dists = sorted(expansion_pixels)
        dist_map = distance_transform_edt(self.seed_mask == 0)

        previous_mask = np.zeros_like(self.seed_mask, dtype=bool)

        for dist in sorted_dists:
            if dist == sorted_dists[0]:
                ring = (dist_map <= dist) & (self.seed_mask == 0)
            else:
                prev_dist = sorted_dists[sorted_dists.index(dist) - 1]
                ring = (dist_map <= dist) & (dist_map > prev_dist) & (self.seed_mask == 0)

            if restrict_to_limit:
                ring &= self.constraint_mask.astype(bool)

            ring &= ~previous_mask

            if min_area:
                ring = self.filter_binary_mask_by_area(ring.astype(np.uint8), min_area).astype(bool)

            previous_mask |= ring

            # Store binary mask
            self.binary_expansions[f"expansion_{dist}"] = ring.astype(np.uint8)

            # Store labeled components using skimage
            self.labeled_expansions[f"expansion_{dist}"] = label(ring.astype(np.uint8))

            # Store label-referenced expansion using seed_mask
            referenced = self.propagate_labels(self.seed_mask, ring)
            self.referenced_expansions[f"expansion_{dist}"] = referenced

        self.binary_expansions["seed_mask"] = (self.seed_mask > 0).astype(np.uint8)
        self.labeled_expansions["seed_mask"] = self.seed_mask.copy()
        self.referenced_expansions["seed_mask"] = self.seed_mask.copy()

        constraint_remaining = (self.constraint_mask.astype(bool) & ~previous_mask).astype(np.uint8)
        self.binary_expansions["constraint_remaining"] = constraint_remaining
        self.labeled_expansions["constraint_remaining"] = np.zeros_like(self.seed_mask, dtype=np.int32)
        self.referenced_expansions["constraint_remaining"] = np.zeros_like(self.seed_mask, dtype=np.int32)

    def propagate_labels(self, seed_labeled: np.ndarray, expansion_mask: np.ndarray) -> np.ndarray:
        """
        Propagate labels from the seed labeled mask into the expansion region using iterative morphological dilation.

        Parameters
        ----------
        seed_labeled : np.ndarray
            Labeled seed mask where non-zero values indicate components.
        expansion_mask : np.ndarray
            Binary mask indicating the expansion region to propagate labels into.

        Returns
        -------
        np.ndarray
            Labeled mask with propagated labels in the expansion area.
        """
        output = np.zeros_like(seed_labeled, dtype=np.int32)
        output[seed_labeled > 0] = seed_labeled[seed_labeled > 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        expansion_mask = expansion_mask.astype(bool)
        iteration = 0

        while True:
            iteration += 1
            prev = output.copy()

            mask_to_fill = (output == 0) & expansion_mask

            # OpenCV only supports certain dtypes for dilation — use float32 safely
            dilated = cv2.dilate(output.astype(np.float32), kernel)
            dilated = dilated.astype(np.int32)

            output[mask_to_fill] = dilated[mask_to_fill]

            if np.array_equal(output, prev):
                break
            if iteration > 1000:
                if self.logger:
                    self.logger.warning("Label propagation exceeded 1000 iterations.")
                break

        return output


class SingleClassObjectAnalysis(GetMasks):
    """
    Analyze and expand a single binary object mask using distance-based ring expansion.

    This class computes concentric ring-based expansions of a binary mask,
    assigns unique labels to each expanded region, and tracks mask lineage
    through label propagation.

    Attributes
    ----------
    mask : np.ndarray
        Binary mask of the object to be expanded.
    expansion_distances : List[int]
        List of expansion radii in pixels.
    labelled_mask : np.ndarray
        Resulting labeled mask with original and expanded areas.
    binary_masks : Dict[str, np.ndarray]
        Dictionary of binary masks keyed by expansion distance.
    labelled_masks : Dict[str, np.ndarray]
        Dictionary of labeled masks keyed by expansion distance.
    reference_masks : Dict[str, np.ndarray]
        Masks encoding reference to original object.
    """

    def __init__(
            self,
            get_masks_instance: GetMasks,
            contours_object: List[np.ndarray],
            contour_name: str = ""
    ) -> None:
        """
        Initialize SingleClassObjectAnalysis with contour data and a GetMasks utility instance.

        Parameters
        ----------
        get_masks_instance : GetMasks
            Instance of GetMasks providing access to shape and filtering methods.
        contours_object : List[np.ndarray]
            List of contours representing the object.
        contour_name : str, optional
            Optional name identifier for the object.
        """

        self.get_masks_instance = get_masks_instance
        self.height = get_masks_instance.height
        self.width = get_masks_instance.width
        self.logger = get_masks_instance.logger

        self.mask_object_SA: Optional[np.ndarray] = None
        self.binary_expansions: Dict[str, np.ndarray] = {}
        self.labeled_expansions: Dict[str, np.ndarray] = {}
        self.referenced_expansions: Dict[str, np.ndarray] = {}
        self.contours_object = contours_object
        self.contour_name = contour_name

    def get_mask_objects(
            self,
            exclude_masks: Optional[List[np.ndarray]] = None,
            filter_area: Optional[int] = None
    ) -> None:
        """
        Generate binary mask from object contours, optionally subtract other masks,
        and apply area-based filtering.

        Parameters
        ----------
        exclude_masks : list of np.ndarray, optional
            List of masks to subtract from the generated object mask.
        filter_area : int, optional
            Minimum area threshold to retain connected components in the object mask.

        Returns
        -------
        None
        """
        mask_object = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask_object, self.contours_object, -1, color=1, thickness=cv2.FILLED)

        if exclude_masks:
            for mask in exclude_masks:
                mask_object = cv2.subtract(mask_object, mask)

        if filter_area is not None:
            self.logger.info(f"Filtering object mask by area: {filter_area}")
            mask_object = self.get_masks_instance.filter_mask_by_area(mask_object, min_area=filter_area)

        self.mask_object_SA = mask_object
        self.logger.info("Mask for objects created.")

    def get_objects_expansion(
            self,
            expansions_pixels: Optional[List[int]] = None,
            filter_area: Optional[int] = None
    ) -> None:
        """
        Expand the object mask using distance-based rings and optionally filter
        each ring by minimum area. Generates binary, labeled, and propagated-label expansion masks.

        Parameters
        ----------
        expansions_pixels : list of int, optional
            List of pixel distances for expansion.
        filter_area : int, optional
            Minimum area threshold to retain connected components in each expansion ring.

        Returns
        -------
        None
        """
        if self.mask_object_SA is None:
            self.logger.error("No object mask to expand.")
            return

        if expansions_pixels is None:
            expansions_pixels = []

        seed_mask = label(self.mask_object_SA)
        dist_map = distance_transform_edt(seed_mask == 0)
        previous_mask = np.zeros_like(seed_mask, dtype=bool)

        for i, dist in enumerate(sorted(expansions_pixels)):
            if i == 0:
                ring = (dist_map <= dist) & (seed_mask == 0)
            else:
                prev_dist = sorted(expansions_pixels)[i - 1]
                ring = (dist_map <= dist) & (dist_map > prev_dist) & (seed_mask == 0)

            ring &= ~previous_mask
            if filter_area:
                ring = self.get_masks_instance.filter_binary_mask_by_area(ring.astype(np.uint8), filter_area).astype(bool)

            previous_mask |= ring

            key = f"expansion_{dist}"
            self.binary_expansions[key] = ring.astype(np.uint8)
            self.labeled_expansions[key] = label(ring.astype(np.uint8))
            self.referenced_expansions[key] = self.propagate_labels(seed_mask, ring)

        # Store the base seed info
        self.binary_expansions["seed_mask"] = (seed_mask > 0).astype(np.uint8)
        self.labeled_expansions["seed_mask"] = seed_mask.copy()
        self.referenced_expansions["seed_mask"] = seed_mask.copy()

    def propagate_labels(self, seed_labeled: np.ndarray, expansion_mask: np.ndarray) -> np.ndarray:
        """
        Propagate labeled regions from a seed mask into the expansion area using iterative dilation.

        Parameters
        ----------
        seed_labeled : np.ndarray
            Input labeled mask where each connected component has a unique integer label.
        expansion_mask : np.ndarray
            Binary mask indicating the region where labels should expand.

        Returns
        -------
        np.ndarray
            Labeled mask with labels propagated into the expansion region.
        """
        output = np.zeros_like(seed_labeled, dtype=np.int32)
        output[seed_labeled > 0] = seed_labeled[seed_labeled > 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        expansion_mask = expansion_mask.astype(bool)
        iteration = 0

        while True:
            iteration += 1
            prev = output.copy()

            mask_to_fill = (output == 0) & expansion_mask
            dilated = cv2.dilate(output.astype(np.float32), kernel)
            dilated = dilated.astype(np.int32)

            output[mask_to_fill] = dilated[mask_to_fill]

            if np.array_equal(output, prev):
                break
            if iteration > 1000:
                if self.logger:
                    self.logger.warning("Label propagation exceeded 1000 iterations.")
                break

        return output

# Propagate labels: If performance is a concern, the dilation-based propagation loop can be optimized with a queue-based BFS flood-fill instead.
class MultiClassObjectAnalysis(GetMasks):
    """
    Analyze and expand multiple object contours across different classes using Voronoi constraints.

    Constructs Voronoi diagrams to limit spatial expansion, assigns unique labels to each object,
    and tracks class-wise and parent-wise mask lineage for downstream analysis.

    Attributes
    ----------
    multiple_contours : dict[str, list[np.ndarray]]
        Input contours grouped by class.
    height : int
        Image height.
    width : int
        Image width.
    save_path : str or None
        Optional path to save outputs.
    vor : scipy.spatial.Voronoi or None
        Computed Voronoi diagram.
    all_centroids : np.ndarray or None
        Coordinates of centroids of input objects.
    class_labels : list[str] or None
        Class label for each object.
    binary_masks : dict[str, np.ndarray]
        Output binary masks by class and expansions.
    labeled_masks : dict[str, np.ndarray]
        Output labeled masks by class and expansions.
    referenced_masks : dict[str, np.ndarray]
        Output referenced masks mapping pixels back to parent objects.
    """

    def __init__(self, get_masks_instance, multiple_contours: dict, save_path: str = None):
        """
        Initialize MultiClassObjectAnalysis instance.

        Parameters
        ----------
        get_masks_instance : GetMasks
            Instance of GetMasks class with base image properties.
        multiple_contours : dict[str, list[np.ndarray]]
            Dictionary mapping class names to lists of contours.
        save_path : str, optional
            Directory path to save outputs (default is None).
        """
        super().__init__()
        self.get_masks_instance = get_masks_instance

        self.height = self.get_masks_instance.height
        self.width = self.get_masks_instance.width
        self.logger = self.get_masks_instance.logger

        # Remove tumour/stroma mask references as per your note
        self.multiple_contours = multiple_contours
        self.masks = None
        self.vor = None
        self.list_of_polygons = None
        self.class_labels = None
        self.all_centroids = None
        self.voronoi_regions = None
        self.voronoi_vertices = None
        self.save_path = save_path

        for class_label, contours in self.multiple_contours.items():
            for i, contour in enumerate(contours):
                if contour.shape[0] < 4:
                    self.logger.warning(f"Skipping contour with less than 4 points for class '{class_label}'.")
                    continue
                self.multiple_contours[class_label][i] = contour[::-1]

    @staticmethod
    def voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct finite Voronoi polygons in 2D by clipping infinite regions.

        Parameters
        ----------
        vor : scipy.spatial.Voronoi
            The original Voronoi diagram from scipy.spatial.
        radius : float, optional
            Distance to extend infinite edges (default is twice the maximum image dimension).

        Returns
        -------
        regions : list[list[int]]
            List of polygon regions as indices of vertices.
        vertices : np.ndarray
            Array of Voronoi vertices coordinates.
        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 2

        # Map of all ridges for a point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct finite polygons
        for p1, region_index in enumerate(vor.point_region):
            vertices = vor.regions[region_index]

            if all(v >= 0 for v in vertices):
                # Finite region
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v1 >= 0 and v2 >= 0:
                    continue

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal vector

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v1 if v1 >= 0 else v2] + direction * radius

                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

            # Sort region counterclockwise
            vs = np.array([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = [new_region[i] for i in np.argsort(angles)]

            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)

    def get_polygons_from_contours(self, contours: List[np.ndarray]) -> List[Polygon]:
        """
        Convert contours into Shapely polygons.

        Parameters
        ----------
        contours : list[np.ndarray]
            List of contour arrays of shape (N, 2).

        Returns
        -------
        polygons : list[Polygon]
            List of valid Shapely Polygon objects.
        """
        polygons = []
        for cnt in contours:
            if cnt.shape[0] < 4:
                continue  # Too few points to form a polygon

            coords = cnt.squeeze()

            if coords.shape[0] < 4:
                continue  # Still too few after squeezing

            # Ensure it's closed (first point == last point)
            if not np.array_equal(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])

            try:
                polygon = Polygon(coords)
                if not polygon.is_valid or polygon.area == 0:
                    continue  # Skip invalid or zero-area polygons
                polygons.append(polygon)
            except Exception:
                continue  # Defensive: skip any invalid contour
        return polygons

    def derive_voronoi_from_contours(self) -> None:
        """
        Compute a Voronoi diagram from centroids of contours.

        Computes Voronoi regions and finite polygons clipped to a large radius.
        Stores regions, vertices, class labels, and centroids for further processing.

        Raises
        ------
        ValueError
            If no contours are available to derive the Voronoi diagram.
        """
        all_contours = [contour for contour_points in self.multiple_contours.values() for contour in contour_points if contour.shape[0] >= 4]
        if not all_contours:
            raise ValueError("No contours found to derive Voronoi diagram.")

        list_of_polygons = self.get_polygons_from_contours(all_contours)

        centroids = []
        class_labels = []
        for class_label, contours in self.multiple_contours.items():
            for contour in contours:
                contour = contour.squeeze()

                if contour is not None and len(contour) >= 3:
                    polygon = Polygon(contour)
                    centroids.append(polygon.centroid)
                    class_labels.append(class_label)
                else:
                    self.logger.warning(f"Skipping contour with less than 4 points for class '{class_label}'.")
                    continue

        if len(centroids) < 4:
            # Not enough data to compute Voronoi
            self.logger.warning("Not enough valid centroids for Voronoi diagram. Skipping Voronoi computation.")
            self.list_of_polygons = list_of_polygons
            self.class_labels = class_labels
            self.all_centroids = np.array([(c.x, c.y) for c in centroids]) if centroids else None
            self.vor = None
            self.voronoi_regions = None
            self.voronoi_vertices = None
            return

        all_centroids = np.array([(c.x, c.y) for c in centroids])
        vor = Voronoi(all_centroids)

        # Use finite polygons clipped to a large radius (image max dimension * 2)
        regions, vertices = self.voronoi_finite_polygons_2d(vor, radius=max(self.height, self.width) * 2)

        self.list_of_polygons = list_of_polygons
        self.class_labels = class_labels
        self.all_centroids = all_centroids
        self.vor = vor
        self.voronoi_regions = regions
        self.voronoi_vertices = vertices

    def get_voronoi_mask(self, category_name: str) -> np.ndarray:
        """
        Get a binary mask for the Voronoi region of a given category.

        If Voronoi regions are not computed (e.g. too few centroids), returns a full mask.

        Parameters
        ----------
        category_name : str
            The category/class name for which the mask is requested.

        Returns
        -------
        mask : np.ndarray
            Binary mask of shape (height, width) with Voronoi regions for the category.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # If Voronoi could not be computed, default to full image for that category
        if self.voronoi_regions is None or self.voronoi_vertices is None:
            # Option 1: Allow expansion to go anywhere
            mask[:, :] = 255
            return mask

        # Normal case
        for idx, (label, region) in enumerate(zip(self.class_labels, self.voronoi_regions)):
            if label != category_name:
                continue
            polygon = self.voronoi_vertices[region]
            polygon[:, 0] = np.clip(polygon[:, 0], 0, self.width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, self.height - 1)
            int_polygon = polygon.astype(np.int32)
            if len(int_polygon) >= 3:
                cv2.fillPoly(mask, [int_polygon], color=255)

        return mask

    def expand_mask(self, mask: np.ndarray, expansion_distance: int) -> np.ndarray:
        """
        Expand a binary mask by a given pixel distance using distance transform.

        The returned mask corresponds to the expansion region excluding the original mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary input mask to expand.
        expansion_distance : int
            Number of pixels to expand the mask by.

        Returns
        -------
        np.ndarray
            Binary mask representing the expansion area only.
        """
        if not np.any(mask):
            return np.zeros_like(mask, dtype=np.uint8)

            # Compute distance from the background to the object mask
        dist_transform = distance_transform_edt(mask == 0)

        # Select pixels within the expansion distance (excluding original mask)
        expanded_mask = (dist_transform <= expansion_distance) & (mask == 0)
        expanded_mask = expanded_mask.astype(np.uint8)  # Convert to binary mask
        return expanded_mask

    def generate_expanded_masks_limited_by_voronoi(
            self,
            expansion_distances: list[int]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Generate expanded masks for each object limited by their Voronoi regions.

        For each class and its contours, original masks are created and then expanded
        by the specified distances, clipped to the corresponding Voronoi region.
        All expansions are labeled and tracked with parent IDs.

        Parameters
        ----------
        expansion_distances : list[int]
            List of pixel distances for mask expansion rings.

        Returns
        -------
        tuple of dict
            - binary_masks: dict mapping mask names to binary masks.
            - labeled_masks: dict mapping mask names to labeled masks with unique IDs.
            - referenced_masks: dict mapping mask names to masks referencing parent object IDs.
        """
        masks = {}  # Step 1: Generate masks for each contour, and label objects
        labeled_masks = {}
        referenced_labeled_mask = np.zeros((self.height, self.width), dtype=np.int32)

        parent_id_counter = 1  # unique ID for each original object across all classes

        # Map from category -> list of (parent_id, mask)
        original_masks_info = {}

        # Create binary masks for each individual contour, label them, assign parent IDs
        for category_name, contours in self.multiple_contours.items():
            if not contours or all(c.shape[0] < 4 for c in contours):
                empty_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                empty_labeled = np.zeros_like(empty_mask, dtype=np.int32)
                key = f"{category_name}"

                masks[key] = empty_mask
                labeled_masks[key] = empty_labeled

                original_masks_info[category_name] = []
                # Add empty expansions too
                for expansion_distance in expansion_distances:
                    exp_key = f"{category_name}_expansion_{expansion_distance}"
                    masks[exp_key] = empty_mask.copy()
                    labeled_masks[exp_key] = empty_labeled.copy()

            category_masks = []
            for contour in contours:
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
                # Label connected components (should be 1 per mask but be safe)
                labeled = label(mask > 0)
                # Extract regionprops if needed, here we just assign parent_id directly
                labeled_mask = np.zeros_like(labeled, dtype=np.int32)
                # Assign the unique parent ID to all pixels in this object
                labeled_mask[labeled > 0] = parent_id_counter

                # Update global referenced mask
                referenced_labeled_mask[labeled_mask > 0] = parent_id_counter

                # Store original mask and label
                masks[f'{category_name}_{parent_id_counter}'] = mask
                labeled_masks[f'{category_name}_{parent_id_counter}'] = labeled_mask

                category_masks.append((parent_id_counter, mask))
                parent_id_counter += 1
            original_masks_info[category_name] = category_masks

        # Step 2: Generate expansions and label them, mapping back to parent IDs
        expanded_masks = {}
        expanded_labeled_masks = {}

        for category_name, masks_info in original_masks_info.items():
            voronoi_mask = self.get_voronoi_mask(category_name)
            for parent_id, base_mask in masks_info:
                previous_expansion_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                for expansion_distance in expansion_distances:
                    current_expansion_mask = self.expand_mask(base_mask.copy(), expansion_distance)
                    current_expansion_mask = cv2.bitwise_and(current_expansion_mask,
                                                             cv2.bitwise_not(previous_expansion_mask))
                    current_expansion_mask = cv2.bitwise_and(current_expansion_mask, voronoi_mask)

                    # Label this expanded mask (connected components)
                    labeled_expansion = label(current_expansion_mask > 0)
                    labeled_mask = np.zeros_like(labeled_expansion, dtype=np.int32)

                    # For each component in expansion assign a unique label encoding:
                    # parent_id * 1000 + expansion_distance (assuming expansion_distance < 1000)
                    # This allows tracing expansions to parent
                    # label_value = parent_id * 1000 + expansion_distance
                    label_value = parent_id

                    labeled_mask[labeled_expansion > 0] = label_value

                    # Update global referenced mask — careful to avoid overwriting originals
                    referenced_labeled_mask[labeled_mask > 0] = label_value

                    key = f'{category_name}_expansion_{expansion_distance}_parent_{parent_id}'
                    expanded_masks[key] = current_expansion_mask
                    expanded_labeled_masks[key] = labeled_mask

                    previous_expansion_mask = cv2.bitwise_or(previous_expansion_mask, current_expansion_mask)

        # Combine all masks and labeled masks
        masks.update(expanded_masks)
        labeled_masks.update(expanded_labeled_masks)
        # Step 3: Aggregate masks by class and expansion name
        aggregate_binary = {}
        aggregate_labeled = {}
        aggregate_referenced = {}

        for key, mask in masks.items():
            parts = key.split('_')

            if 'expansion' in parts:
                category = parts[0]
                expansion_distance = parts[2]
                agg_key = f"{category}_expansion_{expansion_distance}"
            else:
                category = parts[0]
                agg_key = category

            if agg_key not in aggregate_binary:
                aggregate_binary[agg_key] = np.zeros_like(mask)
                aggregate_labeled[agg_key] = np.zeros_like(mask, dtype=np.int32)
                aggregate_referenced[agg_key] = np.zeros_like(mask, dtype=np.int32)

            aggregate_binary[agg_key] = cv2.bitwise_or(aggregate_binary[agg_key], mask)
            aggregate_labeled[agg_key] = np.maximum(aggregate_labeled[agg_key], labeled_masks[key])

            # Referenced mask is pulled from the global referenced_labeled_mask
            aggregate_referenced[agg_key] = np.maximum(
                aggregate_referenced[agg_key],
                np.where(mask > 0, referenced_labeled_mask, 0)
            )

        # Final output
        self.binary_masks = aggregate_binary
        self.labeled_masks = aggregate_labeled
        self.referenced_masks = aggregate_referenced
        return self.binary_masks, self.labeled_masks, self.referenced_masks

    def plot_masks_with_voronoi(self,
                                mask_colors: Dict[str, Tuple[int, int, int]],
                                background_color: Tuple[int, int, int] = (255, 255, 255),
                                show: bool = True,
                                axes: Optional["matplotlib.axes.Axes"] = None,
                                figsize: Tuple[int, int] = (8, 8)
                                ) -> Optional["matplotlib.axes.Axes"]:
        """
        Plots the generated masks overlaid with Voronoi edges.

        Args:
            mask_colors (Dict[str, Tuple[int, int, int]]): Mapping from class name to RGB color.
            background_color (Tuple[int, int, int], optional): RGB color for background. Defaults to white.
            show (bool, optional): If True, displays the plot. Defaults to True.
            axes (matplotlib.axes.Axes, optional): Existing axes to plot on.
            figsize (Tuple[int, int], optional): Figure size for new plot.

        Returns:
            matplotlib.axes.Axes: The plot axes (if `axes` was provided).
        """
        masks = self.binary_masks
        background = np.full((self.height, self.width, 3), background_color, dtype=np.uint8)
        fig, ax = plt.subplots(figsize=figsize) if axes is None else (None, axes)
        legend_patches = []
        seen_classes = set()

        for mask_name, mask in masks.items():
            # Identify base class: 'gd' or 'cd8' from names like 'gd_expansion_30_0'
            base_class = mask_name.split('_')[0]

            # Get color for this base class
            color = np.array(mask_colors.get(base_class, (128, 128, 128)))
            background[mask != 0] = color

            # Add legend entry only once per base class
            if base_class not in seen_classes:
                legend_patches.append(mpatches.Patch(color=color / 255, label=base_class))
                seen_classes.add(base_class)

        ax.imshow(background, origin='lower')

        # Draw Voronoi edges
        if self.vor:
            voronoi_plot_2d(self.vor, ax=ax, show_vertices=False, line_colors='black', line_alpha=0.6)

            # Plot centroids (smaller dots)
            if self.all_centroids is not None:
                centroids = np.array(self.all_centroids)
                ax.plot(centroids[:, 0], centroids[:, 1], '*', markersize=1, alpha=0.6)

        # Add clean legend (gd, cd8)
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', bbox_transform=ax.transAxes)

        if self.save_path:
            save_path = os.path.join(self.save_path, 'masks_with_voronoi_edges.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        if show:
            plt.show()

        return ax if axes is not None else None
