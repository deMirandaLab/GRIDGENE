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

# TODO add logging
# TODO add docstrings
# TODO add type hints
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
    """

    def __init__(self,
        mask_dict: Dict[str, np.ndarray],
        segmentation,
        segmentation_type: str = "auto",
        save_path: Optional[str] = None,
        min_x: float = 0,
        min_y: float = 0,
        flip_masks:bool = True,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the Overlay object.

        Parameters:
            mask_dict: Dictionary of masks {mask_name: mask_array}.
            segmentation: Segmentation data (GeoJSON, label mask, or SpatialData).
            segmentation_type: Type of segmentation ('polygons', 'label_mask', or 'auto').
            save_path: Optional path to save visualizations.
            min_x: Minimum x to shift polygons.
            min_y: Minimum y to shift polygons.
            logger: Optional custom logger.
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
        """Return the shape of the masks (assumes all masks have the same shape)."""
        return next(iter(self.mask_dict.values())).shape

    def _detect_segmentation_type(self)-> str:
        """Detect the segmentation type based on input structure."""
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

    def shift_polygons(self)-> None:
        """Shift all polygon coordinates by (min_x, min_y)."""
        for geometry in self.segmentation['geometries']:
            polygon = np.array(geometry['coordinates'][0])
            shifted = polygon - np.array([self.min_x, self.min_y])
            geometry['coordinates'][0] = [(round(x), round(y)) for x, y in shifted]

    def _get_cell_masks_from_polygons(self)-> Dict[int, np.ndarray]:
        """Create binary masks from polygon segmentation."""
        masks = {}
        shape = self.mask_shape
        for geometry in self.segmentation['geometries']:
            cell_id = int(geometry['cell'])
            poly = [(round(x), round(y)) for x, y in geometry['coordinates'][0]]
            img = Image.new('L', (shape[1], shape[0]), 0)
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
            masks[cell_id] = np.array(img)
        return masks

    def _get_cell_masks_from_label_mask(self, label_mask):
        """Convert a labeled mask to binary masks per label."""
        return {cid: (label_mask == cid).astype(np.uint8)
                for cid in np.unique(label_mask) if cid != 0}

    def _extract_segmentation_masks(self)-> Dict[int, np.ndarray]:
        """Extract binary cell masks from segmentation input."""
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
        """Compute overlap between masks and segmented regions."""
        if self.segmentation_type == 'polygons':
            self.map_mask_cell_polygons()
        elif self.segmentation_type == 'label_mask':
            # TODO add label mask overlap computation
            raise NotImplementedError("Label mask overlap not implemented.")
        self.logger.info("Computed overlap between masks and segmentation.")
        return self.results

    def map_mask_cell_masks(self) -> None:
        pass  # Placeholder for label mask-based overlap

    def map_mask_cell_polygons(self)-> None:
        """Computes the number of overlapping pixels between each segmentation polygon (cell) and each mask.

            This method:
            - Iterates through all polygon geometries defined in the segmentation.
            - Rasterizes each polygon into a binary mask using Pillow.
            - Counts how many pixels in each input mask fall within each polygon.

            The results are stored in `self.results` as a nested dictionary:
                {
                    cell_id: {
                        mask_name: pixel_overlap_count,
                        ...
                    },
                    ...
                }
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

    def plot_masks_overlay_segmentation(
            self,
            titles: List[str],
            colors: List[str],
            background: str = 'white',
            save_path: Optional[str] = None,
            show: bool = True,
            show_legend=True
    ) -> None:
        """Overlay binary masks and segmentation polygons for visualization.
        Parameters:
            titles: List of titles for each mask.
            colors: List of colors corresponding to each mask.
            background: Background color for the plot.
            save_path: Optional path to save the overlay plot.
            show: Whether to display the plot.
            show_legend: Whether to show the legend in the plot.
        Returns: None
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

    def _plot_masks_overlay_segmentation_polygons(self,
        titles: List[str],
        colors: List[str],
        background: str = 'white'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Overlay binary masks and polygon outlines onto a canvas."""
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

    def plot_colored_by_mask_overlap(self,
        mask_to_color: List[str],
        color_map: str = 'Reds',
        show: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 15)
    ) -> None:
        """
        Color segmented polygons based on overlap percentage with specified masks.

        Parameters:
            mask_to_color: List of mask names to base coloring on.
            color_map: Matplotlib colormap.
            show: Whether to show the figure.
            save_path: Path to save image if desired.
            figsize: Size of the figure.
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


    def _plot_colored_by_mask_overlap_polygons(self,
        mask_to_color: List[str],
        color_map: str,
        figsize: Tuple[int, int]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Internal: Color polygons based on mask overlap."""

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

#
# class Overlay:
#     def __init__(self, mask_dict, segmentation, segmentation_type="auto",
#                  save_path=None, min_x=0, min_y=0, logger: Optional[logging.Logger] = None,):
#         """
#         Parameters:
#         - mask_dict: dict of {mask_name: 2D np.array} - usually from gridgen
#         - segmentation: GeoJSON-like dict, np.array, or SpatialData Shapes/Labels
#         - segmentation_type: 'polygons', 'label_mask', or 'auto'
#         - save_path: folder path to save plots
#         """
#         self.mask_dict = mask_dict
#         self.save_path = save_path
#
#         # Automatically detect segmentation type if needed
#         self.segmentation_type = self._detect_segmentation_type(segmentation) if segmentation_type == "auto" else segmentation_type
#         self.segmentation = segmentation
#         self.min_x = min_x
#         self.min_y = min_y
#         self.logger = logger or get_logger(f'{__name__}.{"GetMasks"}')
#         self.logger.info("Initialized GetMasks")
#
#         if self.min_x != 0 or self.min_y != 0:
#             self.logger.info(f"Warning: min_x={self.min_x}, min_y={self.min_y} are set. "
#                   "This will shift the polygon coordinates in the segmentation.")
#             if self.segmentation_type == 'polygons':
#                 self.shift_polygons()
#
#
#     def _detect_segmentation_type(self, segmentation):
#         if isinstance(segmentation, dict) and 'geometries' in segmentation:
#             return 'polygons'
#         elif isinstance(segmentation, np.ndarray):
#             return 'label_mask'
#         elif isinstance(segmentation, sd.SpatialData):
#             if 'shapes' in segmentation:
#                 return 'polygons'
#             elif 'labels' in segmentation:
#                 return 'label_mask'
#         raise ValueError("Unable to detect segmentation type. Please specify it explicitly.")
#
#     def shift_polygons(self):
#         for geometry in self.segmentation['geometries']:
#             polygon = np.array(geometry['coordinates'][0])
#             shifted = polygon - np.array([self.min_x, self.min_y])
#             shifted = [(round(x), round(y)) for x, y in shifted]
#             geometry['coordinates'][0] = shifted  # Update in-place
#
#     def _get_cell_masks_from_polygons(self, geojson_data, shape):
#         """Create binary masks for each cell polygon."""
#         masks = {}
#         polygons = [geometry['coordinates'][0] for geometry in geojson_data['geometries']]
#         cell_ids = [int(geometry['cell']) for geometry in geojson_data['geometries']]
#         for poly, cell_id in zip(polygons, cell_ids):
#             img = Image.new('L', (shape[1], shape[0]), 0)
#             ImageDraw.Draw(img).polygon([(round(x), round(y)) for x, y in poly], outline=1, fill=1)
#             masks[cell_id] = np.array(img)
#         return masks
#
#     def _get_cell_masks_from_label_mask(self, label_mask):
#         """Split a labelled segmentation mask into binary masks."""
#         masks = {}
#         for cell_id in np.unique(label_mask):
#             if cell_id == 0:  # skip background
#                 continue
#             masks[cell_id] = (label_mask == cell_id).astype(np.uint8)
#         return masks
#
#     def _extract_segmentation_masks(self, mask_shape):
#         if self.segmentation_type == 'polygons':
#             if isinstance(self.segmentation, sd.SpatialData):
#                 # Get from SpatialData 'shapes'
#                 shapes = self.segmentation.shapes[list(self.segmentation.shapes.keys())[0]]
#                 cell_ids = shapes.obs.get("cell", np.arange(len(shapes)))
#                 polygons = [poly.exterior.coords for poly in shapes.geometry]
#                 geojson_like = {'geometries': [{'coordinates': [list(polygon)], 'cell': str(cid)} for polygon, cid in zip(polygons, cell_ids)]}
#                 return self._get_cell_masks_from_polygons(geojson_like, mask_shape)
#             else:
#                 return self._get_cell_masks_from_polygons(self.segmentation, mask_shape)
#
#         elif self.segmentation_type == 'label_mask':
#             if isinstance(self.segmentation, sd.SpatialData):
#                 labels = self.segmentation.labels[list(self.segmentation.labels.keys())[0]]
#                 label_mask = labels.data.values
#             else:
#                 label_mask = self.segmentation
#             return self._get_cell_masks_from_label_mask(label_mask)
#
#         raise ValueError("Unsupported segmentation type.")
#
#     @timeit
#     def compute_overlap(self):
#         """
#         Computes the overlap between segmentation and masks.
#
#         Returns:
#             dict: A nested dictionary with overlap pixel counts {cell_id: {mask_name: count}}.
#         """
#         if self.segmentation_type == 'polygons':
#             self.map_mask_cell_polygons()
#         elif self.segmentation_type == 'label_mask':
#             self.logger.warning("Label mask overlap not implemented.")
#             # self.map_mask_cell_masks()  # Placeholder
#         else:
#             raise ValueError("Unsupported segmentation type.")
#
#         self.logger.info("Computed overlap between masks and segmentation.")
#         return self.results
#
#     def map_mask_cell_masks(self, ):
#         pass
#
#     def map_mask_cell_polygons(self):
#         """Compute overlap without storing all cell masks in memory."""
#         mask_shape = next(iter(self.mask_dict.values())).shape
#         results = {}
#
#         # Extract polygons from geojson
#         polygons = [geometry['coordinates'][0] for geometry in self.segmentation['geometries']]
#         cell_ids = [int(geometry['cell']) for geometry in self.segmentation['geometries']]
#
#         for polygon, cell_id in zip(polygons, cell_ids):
#             # polygon = np.array(polygon) - np.array([min_x, min_y])
#             polygon = [(round(x), round(y)) for x, y in polygon]
#
#             img = Image.new('L', (mask_shape[1], mask_shape[0]), 0)
#             ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
#             polygon_mask = np.array(img, dtype=bool)  # smaller memory than int
#
#             results[cell_id] = {}
#             for mask_name, mask in self.mask_dict.items():
#                 # Avoid large intermediate arrays (polygon_mask * mask)
#                 overlap = np.count_nonzero(mask[polygon_mask])
#                 results[cell_id][mask_name] = overlap
#         self.results = results
#
#
#
#     def plot_masks_overlay_segmentation(self, titles, colors,
#                                         background='white', save_path=None, show = True):
#
#         # Initialize an empty mask to store the overlay
#         if self.segmentation_type == 'polygons':
#             fig, ax = self._plot_masks_overlay_segmentation_polygons(titles, colors,background)
#         elif self.segmentation_type == 'label_mask':
#             fig, ax = self._plot_masks_overlay_label_mask(titles, colors, background)
#         else:
#             raise ValueError(f"Unsupported segmentation type: {self.segmentation_type}")
#
#         # Set the title
#         ax.set_title("Overlay of Masks and Segmentation", fontsize=16)
#         ax.axis('off')
#         # Set the legend
#         handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, titles)]
#         legend = ax.legend(
#             handles=handles,
#             loc='center',
#             frameon=True,
#             framealpha=1,
#             facecolor='white',
#             edgecolor='black',
#             fontsize=14
#         )
#         legend.set_title("Masks", prop={'size': 16})
#
#
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path, dpi=500)
#             self.logger.info(f"Saved overlay plot to {save_path}")
#         if show:
#             plt.show()
#         else:
#             plt.close(fig)
#
#
#     def _plot_masks_overlay_segmentation_polygons(self, titles, colors,
#                                         background='white'):
#         geojson_data = self.segmentation
#         # Create a new figure
#         fig, ax = plt.subplots(figsize=(10, 10))
#
#         # Generate a colormap
#         cmap = ListedColormap([background] + colors)
#
#         # Initialize an empty mask to store the overlay
#         overlay = np.zeros_like(next(iter(self.mask_dict.values())), dtype=np.int64)
#
#         for i, (mask_name, mask) in enumerate(self.mask_dict.items()):
#             overlay += (mask * (i + 1))
#
#         # Display the overlay
#         ax.imshow(overlay, cmap=cmap, origin='lower')
#
#         # Extract all polygons
#         polygons = [np.array(geometry['coordinates'][0]) for geometry in geojson_data['geometries']]
#
#         for polygon in polygons:
#             ax.plot(polygon[:, 0], polygon[:, 1], color='black', linewidth=0.3)
#         ax.axis('off')
#
#         return fig, ax
#
#     def _plot_masks_overlay_label_mask(self, titles, colors):
#         pass
#
#
#     # THIS IS POLYGONS
#     def plot_colored_by_mask_overlap(self, mask_to_color, color_map='Reds', show = True, save_path=None, figsize=(15, 15)):
#         """Color cells based on the percentage of overlap with selected masks"""
#         if self.results is None:
#             self.logger.info("there are no mapping results stored. "
#                   "Computing results from segmentation masks...")
#             self.results = self.compute_overlap()
#
#         if self.segmentation_type == 'polygons':
#            fig, ax = self._plot_colored_by_mask_overlap_polygons(mask_to_color, color_map, figsize)
#
#         elif self.segmentation_type == 'label_mask':
#             pass
#
#
#         if save_path: # todo change this to self.save_path or masking name or shit
#             fig.savefig(save_path, dpi=500)
#             self.logger.info(f"Saved colored overlap plot to {save_path}")
#
#         if show:
#             fig.show()
#         else:
#             plt.close(fig)  # Save memory if not showing
#
#     def _plot_colored_by_mask_overlap_polygons(self, mask_color, color_map, figsize):
#         geojson_data_selection = self.segmentation
#         results = self.results
#
#         # Create a new figure
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#
#         # For each feature in the geojson data
#         for feature in geojson_data_selection['geometries']:
#             # Get the cell ID and polygon coordinates
#             cell_id = int(feature['cell'])
#             polygon = feature['coordinates'][0]
#             # Get the results for this cell
#             # polygon_results = {mask_name: count for (polygon_id, mask_name), count in results.items() if polygon_id == cell_id}
#             polygon_results = results.get((cell_id), {})
#             # Calculate the total number of pixels in this polygon
#             total_pixels = sum(polygon_results.values())
#
#             # Calculate the percentage of pixels in the tumour mask
#             if total_pixels != 0:
#                 percentages = {mask_name: (count / total_pixels) * 100 for mask_name, count in polygon_results.items()}
#             else:
#                 percentages = {mask_name: 0 for mask_name in polygon_results.keys()}
#
#             # Calculate the total percentage for the tumour mask
#             total_percentage = sum(percentages.get(mask, 0) for mask in mask_color)
#
#             # Get a color from the 'Reds' colormap based on the total percentage
#             color = plt.get_cmap(color_map)(Normalize(0, 100)(total_percentage))
#             # Create a polygon and fill it with the color
#             polygon_array = np.array(polygon)
#             plt.plot(polygon_array[:, 0], polygon_array[:, 1], color=color, linewidth=0.3)
#             polygon_patch = patches.Polygon(np.array(polygon), facecolor=color, edgecolor='black', linewidth=0.3)
#
#             # Add the polygon to the plot
#             ax.add_patch(polygon_patch)
#         ax.axis('off')
#         return fig, ax
