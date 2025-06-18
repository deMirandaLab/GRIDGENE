import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import patches as mpatches
from matplotlib import patches
from PIL import Image, ImageDraw
import spatialdata as sd
import xarray as xr
from functools import wraps
import time
import logging
from functools import wraps
from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, ListedColormap
from PIL import Image, ImageDraw
import shapely.geometry as sg

try:
    import spatialdata as sd
except ImportError:
    sd = None

from shapely.geometry import Polygon
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull
from gridgen.logger import get_logger

# TODO spatialdata support

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]  # assumes it's a method
        start = time.time()
        result = func(*args, **kwargs)
        duration = end = time.time() - start
        if hasattr(self, "logger"):
            self.logger.info(f"{func.__name__} took {duration:.4f} seconds")
        else:
            print(f"{func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper

class Overlay:
    """
    Overlay segmentation with binary masks for comparison and visualization.

    Supports polygon-based and label-mask segmentations.

    Parameters
    ----------
    mask_dict : dict[str, np.ndarray]
        Dictionary of masks, where keys are mask names and values are binary mask arrays.
    segmentation : dict or np.ndarray or SpatialData
        Segmentation data, either GeoJSON-like dict, label mask ndarray, or SpatialData.
    segmentation_type : {'auto', 'polygons', 'label_mask'}, optional
        Type of segmentation input. Default is 'auto' to detect automatically.
    save_path : str, optional
        Optional path to save visualizations.
    min_x : float, optional
        Minimum x coordinate to shift polygons. Default is 0.
    min_y : float, optional
        Minimum y coordinate to shift polygons. Default is 0.
    flip_masks : bool, optional
        Whether to flip masks vertically and rotate. Default is True.
    logger : logging.Logger, optional
        Custom logger. If None, a default logger is used.
    """

    def __init__(self, mask_dict: Dict[str, np.ndarray], segmentation, segmentation_type: str = "auto",
                 save_path: Optional[str] = None, min_x: float = 0, min_y: float = 0, flip_masks: bool = True,
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the Overlay object.

        Parameters
        ----------
        mask_dict : dict[str, np.ndarray]
            Dictionary of masks {mask_name: mask_array}.
        segmentation : dict or np.ndarray or SpatialData
            Segmentation data (GeoJSON, label mask, or SpatialData).
        segmentation_type : str, optional
            Type of segmentation ('polygons', 'label_mask', or 'auto'), by default "auto".
        save_path : str, optional
            Optional path to save visualizations, by default None.
        min_x : float, optional
            Minimum x to shift polygons, by default 0.
        min_y : float, optional
            Minimum y to shift polygons, by default 0.
        flip_masks : bool, optional
            Whether to flip masks vertically and rotate, by default True.
        logger : logging.Logger, optional
            Optional custom logger, by default None.
        """
        self.mask_dict = mask_dict
        self.save_path = save_path
        self.segmentation = segmentation
        self.min_x = min_x
        self.min_y = min_y
        self.logger = logger or get_logger(__name__)
        self.logger.info("Initialized Overlay")

        self.segmentation_type = self._detect_segmentation_type() if segmentation_type == "auto" else segmentation_type
        if self.min_x != 0 or self.min_y != 0:
            self.logger.info(f"Shifting polygons by min_x={self.min_x}, min_y={self.min_y}")
            if self.segmentation_type == 'polygons':
                self.shift_polygons()
        self.results = None
        if flip_masks:
            self.logger.info("Flipping masks vertically")
            for mask_name, mask in self.mask_dict.items():
                mask = np.flip(mask, 0)  # Flip vertically
                mask = np.rot90(mask, -1)  # Rotate 90 degrees to the right
                self.mask_dict[mask_name] = mask
    @property
    def mask_shape(self):
        """
        Return the shape of the masks.

        Returns
        -------
        tuple
            Shape of the first mask in the dictionary.
        """
        return next(iter(self.mask_dict.values())).shape

    def _detect_segmentation_type(self) -> str:
        """
        Detect segmentation type based on input structure.

        Returns
        -------
        str
            Detected segmentation type: 'polygons' or 'label_mask'.

        Raises
        ------
        ValueError
            If segmentation type cannot be detected.
        """
        if isinstance(self.segmentation, dict) and 'geometries' in self.segmentation:
            return 'polygons'
        elif isinstance(self.segmentation, np.ndarray):
            return 'label_mask'
        elif sd and isinstance(self.segmentation, sd.SpatialData):
            if 'shapes' in self.segmentation:
                return 'polygons'
            elif 'labels' in self.segmentation:
                return 'label_mask'
        raise ValueError("Unable to detect segmentation type. Please specify it explicitly.")

    def shift_polygons(self) -> None:
        """
        Shift all polygon coordinates by (min_x, min_y).

        Modifies
        --------
        self.segmentation : dict
            Polygon coordinates are shifted in place.
        """
        for geometry in self.segmentation['geometries']:
            polygon = np.array(geometry['coordinates'][0])
            shifted = polygon - np.array([self.min_x, self.min_y])
            geometry['coordinates'][0] = [(round(x), round(y)) for x, y in shifted]

    def _get_cell_masks_from_polygons(self) -> Dict[int, np.ndarray]:
        """
        Create binary masks from polygon segmentation.

        Returns
        -------
        dict[int, np.ndarray]
            Dictionary mapping cell IDs to binary masks.
        """
        masks = {}
        shape = self.mask_shape
        for geometry in self.segmentation['geometries']:
            cell_id = int(geometry['cell'])
            poly = [(round(x), round(y)) for x, y in geometry['coordinates'][0]]
            img = Image.new('L', (shape[1], shape[0]), 0)
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
            masks[cell_id] = np.array(img)
        return masks

    def _get_cell_masks_from_label_mask(self, label_mask) -> Dict[int, np.ndarray]:
        """
        Convert labeled mask to binary masks per label.

        Parameters
        ----------
        label_mask : np.ndarray
            Label mask array.

        Returns
        -------
        dict[int, np.ndarray]
            Dictionary mapping label IDs to binary masks.
        """
        return {cid: (label_mask == cid).astype(np.uint8)
                for cid in np.unique(label_mask) if cid != 0}

    def _extract_segmentation_masks(self) -> Dict[int, np.ndarray]:
        """
        Extract binary cell masks from segmentation input.

        Returns
        -------
        dict[int, np.ndarray]
            Dictionary mapping cell IDs to binary masks.

        Raises
        ------
        ValueError
            If segmentation type is unsupported.
        """
        if self.segmentation_type == 'polygons':
            if sd and isinstance(self.segmentation, sd.SpatialData):
                shapes = self.segmentation.shapes[list(self.segmentation.shapes.keys())[0]]
                polygons = [poly.exterior.coords for poly in shapes.geometry]
                cell_ids = shapes.obs.get("cell", np.arange(len(shapes)))
                geojson_like = {'geometries': [
                    {'coordinates': [list(p)], 'cell': str(cid)}
                    for p, cid in zip(polygons, cell_ids)
                ]}
                return self._get_cell_masks_from_polygons()
            return self._get_cell_masks_from_polygons()
        elif self.segmentation_type == 'label_mask':
            if sd and isinstance(self.segmentation, sd.SpatialData):
                label_mask = self.segmentation.labels[list(self.segmentation.labels.keys())[0]].data.values
            else:
                label_mask = self.segmentation
            return self._get_cell_masks_from_label_mask(label_mask)
        raise ValueError("Unsupported segmentation type")

    @timeit
    def compute_overlap(self):
        """
        Compute overlap between masks and segmented regions.

        Returns
        -------
        dict
            Nested dictionary of overlap counts {cell_id: {mask_name: pixel_overlap_count}}.

        Raises
        ------
        NotImplementedError
            If overlap for label masks is requested.
        """
        if self.segmentation_type == 'polygons':
            self.map_mask_cell_polygons()
        elif self.segmentation_type == 'label_mask':
            # TODO add label mask overlap computation
            raise NotImplementedError("Label mask overlap not implemented.")
        self.logger.info("Computed overlap between masks and segmentation.")
        return self.results

    def map_mask_cell_masks(self) -> None:
        """
        Placeholder for label mask-based overlap computation.
        """
        pass  # Placeholder for label mask-based overlap

    def map_mask_cell_polygons(self) -> None:
        """
        Compute overlap counts between each segmentation polygon and each mask.

        Stores results in `self.results` as:
            {cell_id: {mask_name: pixel_overlap_count, ...}, ...}
        """
        shape = self.mask_shape
        results = {}
        for geometry in self.segmentation['geometries']:
            cell_id = int(geometry['cell'])
            polygon = [(round(x), round(y)) for x, y in geometry['coordinates'][0]]
            img = Image.new('L', (shape[1], shape[0]), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            poly_mask = np.array(img, dtype=bool)

            results[cell_id] = {}
            for mask_name, mask in self.mask_dict.items():
                results[cell_id][mask_name] = np.count_nonzero(mask[poly_mask])
        self.results = results

    def plot_masks_overlay_segmentation(self, titles: List[str], colors: List[str], background: str = 'white',
                                        save_path: Optional[str] = None, show: bool = True,
                                        show_legend: bool = True) -> None:
        """
        Overlay binary masks and segmentation polygons for visualization.

        Parameters
        ----------
        titles : list of str
            Titles for each mask.
        colors : list of str
            Colors corresponding to each mask.
        background : str, optional
            Background color, by default 'white'.
        save_path : str, optional
            Path to save the overlay plot, by default None.
        show : bool, optional
            Whether to display the plot, by default True.
        show_legend : bool, optional
            Whether to show legend, by default True.
        """
        if self.segmentation_type == 'polygons':
            fig, ax = self._plot_masks_overlay_segmentation_polygons(titles, colors, background)
        else:
            raise NotImplementedError("Label mask overlay plot not implemented.")

        ax.set_title("Overlay of Masks and Segmentation", fontsize=16)
        ax.axis('off')
        if show_legend:
            handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, titles)]
            legend = ax.legend(
                handles=handles,
                loc='lower left',
                bbox_to_anchor=(0, -0.1),
                frameon=True,
                framealpha=1,
                facecolor='white',
                edgecolor='black',
                fontsize=14
            )
            legend.set_title("Masks", prop={'size': 16})

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=500)
            self.logger.info(f"Saved overlay plot to {save_path}")
        if show:
            plt.show()
        # else:
        plt.close(fig)

    def _plot_masks_overlay_segmentation_polygons(self, titles: List[str], colors: List[str],
                                                  background: str = 'white') -> Tuple[plt.Figure, plt.Axes]:
        """
        Internal: overlay binary masks and polygon outlines on canvas.

        Parameters
        ----------
        titles : list of str
            Titles for each mask.
        colors : list of str
            Colors for each mask.
        background : str, optional
            Background color, by default 'white'.

        Returns
        -------
        tuple
            Matplotlib figure and axes with overlay.
        """
        if not hasattr(self, 'mask_shape'):
            # Get the shape from the first mask in the dict
            self.mask_shape = next(iter(self.mask_dict.values())).shape

        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = ListedColormap([background] + colors)

        # Initialize canvas
        overlay = np.zeros(self.mask_shape, dtype=np.int64)

        # Combine masks
        for i, (mask_name, mask) in enumerate(self.mask_dict.items()):
            overlay += (mask * (i + 1))

        # Show image
        ax.imshow(overlay, cmap=cmap, origin='lower')

        # Plot polygons
        for geometry in self.segmentation['geometries']:
            polygon = np.array(geometry['coordinates'][0])
            ax.plot(polygon[:, 0], polygon[:, 1], color='black', linewidth=0.3)

        ax.axis('off')
        return fig, ax

    def plot_colored_by_mask_overlap(self, mask_to_color: List[str], color_map: str = 'Reds', show: bool = True,
                                     save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 15)) -> None:
        """
        Color segmented polygons based on overlap percentage with specified masks.

        Parameters
        ----------
        mask_to_color : list of str
            Mask names to base coloring on.
        color_map : str, optional
            Matplotlib colormap, by default 'Reds'.
        show : bool, optional
            Whether to display the plot, by default True.
        save_path : str, optional
            Path to save the plot, by default None.
        figsize : tuple, optional
            Figure size, by default (15, 15).
        """
        if self.results is None:
            self.logger.info("No results found. Computing overlap...")
            self.compute_overlap()
        if self.segmentation_type == 'polygons':
            fig, ax = self._plot_colored_by_mask_overlap_polygons(mask_to_color, color_map, figsize)
        else:
            raise NotImplementedError("Only polygon plotting is implemented.")

        if save_path:
            fig.savefig(save_path, dpi=500)
            self.logger.info(f"Saved colored overlap plot to {save_path}")

        if show:
            plt.show()
        # else:
        plt.close(fig)  # Save memory if not showing

    def _plot_colored_by_mask_overlap_polygons(self, mask_to_color: List[str], color_map: str,
                                               figsize: Tuple[int, int]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Internal: color polygons based on mask overlap percentages.

        Parameters
        ----------
        mask_to_color : list of str
            Masks used for coloring.
        color_map : str
            Colormap name.
        figsize : tuple
            Figure size.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for feature in self.segmentation['geometries']:
            cell_id = int(feature['cell'])
            polygon = feature['coordinates'][0]
            polygon_array = np.array(polygon)
            polygon_results = self.results.get(cell_id, {})
            total_pixels = sum(polygon_results.values())

            if total_pixels != 0:
                percentages = {k: v / total_pixels * 100 for k, v in polygon_results.items()}
            else:
                percentages = {k: 0 for k in mask_to_color}

            total_percentage = sum(percentages.get(k, 0) for k in mask_to_color)
            color = plt.get_cmap(color_map)(Normalize(0, 100)(total_percentage))

            patch = mpatches.Polygon(polygon_array, facecolor=color, edgecolor='black', linewidth=0.3)
            ax.add_patch(patch)
        ax.axis('off')
        all_coords = np.vstack([np.array(f['coordinates'][0]) for f in self.segmentation['geometries']])
        ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
        ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
        ax.set_aspect('equal')

        ax.axis('off')
        return fig, ax
