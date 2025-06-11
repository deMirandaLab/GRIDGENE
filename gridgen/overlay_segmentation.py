import numpy as np
from PIL import Image, ImageDraw
from matplotlib.colors import ListedColormap
from matplotlib import patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from matplotlib.colors import ListedColormap
from matplotlib import patches as mpatches
def plot_polygons(geojson_data, title, show=True, save=False):
    polygons_baysor_cellprior = [geometry['coordinates'][0] for geometry in geojson_data['geometries']]
    # Create a figure with three subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # Plot the polygons from both polygons_baysor_cellprior and polygons_baysor_nucprior in the third subplot
    for polygon in polygons_baysor_cellprior:
        polygon = np.array(polygon)
        plt.plot(polygon[:, 0], polygon[:, 1], color='black', linewidth=0.3)

    plt.title(title)
    if show: plt.show()
    # add save path
    if save: plt.savefig(f'{title}.png')

class Overlay():
    def __init__(self, mask_dict, segmentation, save_path=None):
        self.mask_dict = mask_dict
        self.segmentation = segmentation

        # add this to mask segmentations
        self.segmentation_type = "polygons"
        self.save_path = save_path
    # check if I always need to flip  masks

    def map_mask_cell(self, min_x=None, min_y=None, save_path = None):
        """maps the cell ids to the masks"""
        mask_dict = self.mask_dict
        geojson_data = self.segmentation
        # Get the shape from the first mask in the dictionary
        mask_shape = next(iter(mask_dict.values())).shape

        # Initialize a dictionary to store the results
        results = {}

        polygons_baysor_cellprior = [geometry['coordinates'][0] for geometry in geojson_data['geometries']]
        cell_ids = [int(geometry['cell']) for geometry in geojson_data['geometries']]
        for polygon, cell_id in zip(polygons_baysor_cellprior, cell_ids):
            polygon = np.array(polygon)
            # if min_x is not None and min_y is not None:
            polygon = polygon - np.array([min_x, min_y])  # Shift the polygon coordinates
            polygon = [(round(x), round(y)) for x, y in polygon]

            # Create an empty image
            img = Image.new('L', (mask_shape[1], mask_shape[0]), 0)
            # Create an ImageDraw object
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            # Convert the image to a numpy array
            polygon_mask = np.array(img)

            # Initialize a dictionary for this cell
            results[cell_id] = {}

            # For each mask
            for mask_name, mask in mask_dict.items():
                # Multiply the polygon mask with the loaded mask
                masked_pixels = polygon_mask * mask
                # Count the number of matching pixels
                count = np.count_nonzero(masked_pixels)
                # Store the result
                results[cell_id][mask_name] = count
        self.mapping = results
        if save_path:
            with open(os.path.join(save_path), 'w') as file:
                json.dump(data, file, indent=4)
        return results

    def plot_masks_overlay_segmentation(self, titles, colors, min_x=None, min_y=None,
                                        background='white', save_path=None, show = False):
        mask_dict = self.mask_dict
        geojson_data = self.segmentation
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create a colormap: first color is background, then mask colors
        cmap = ListedColormap([background] + colors)

        # Create an empty array for overlay
        overlay = np.zeros_like(next(iter(mask_dict.values())), dtype=np.uint8)

        # Assign unique values to each mask
        for i, (mask_name, mask) in enumerate(mask_dict.items(), start=1):  # Start from 1 (0 is background)
            overlay = np.where(mask > 0, i, overlay)  # Assign index i where mask is nonzero

        # Display the overlay with explicit value range
        ax.imshow(overlay, cmap=cmap, origin='lower', vmin=0, vmax=len(colors))

        # Plot the polygons (GeoJSON boundaries)
        polygons_baysor_cellprior = [geometry['coordinates'][0] for geometry in geojson_data['geometries']]
        for polygon in polygons_baysor_cellprior:
            polygon = np.array(polygon)
            if min_x is not None and min_y is not None:
                polygon = polygon - np.array([min_x, min_y])  # Shift polygon coordinates
            ax.plot(polygon[:, 0], polygon[:, 1], color='black', linewidth=0.3)
        # legend
        patches = [mpatches.Patch(color=color, label=title) for title, color in zip(titles, colors)]
        ax.legend(handles=patches, loc='upper right', title='Masks')

        # Show the plot
        if save_path: plt.savefig(save_path)
        if show: plt.show()
        return fig, ax


    def plot_mask_results(self, results, mask_color, color_map, save_path=None, show = False):
        mask_dict = self.mask_dict
        geojson_data_selection = self.segmentation
        if results is None:
            results = self.results

        # Create a new figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        # For each feature in the geojson data
        for feature in geojson_data_selection['geometries']:
            # Get the cell ID and polygon coordinates
            cell_id = int(feature['cell'])
            polygon = feature['coordinates'][0]
            # Get the results for this cell
            # polygon_results = {mask_name: count for (polygon_id, mask_name), count in results.items() if polygon_id == cell_id}
            #         polygon_results = results.get((str(cell_id)), {})
            polygon_results = results.get(cell_id, {})
            # Calculate the total number of pixels in this polygon
            total_pixels = sum(polygon_results.values())

            # Calculate the percentage of pixels in the tumour mask
            if total_pixels != 0:
                percentages = {mask_name: (count / total_pixels) * 100 for mask_name, count in polygon_results.items()}
            else:
                percentages = {mask_name: 0 for mask_name in polygon_results.keys()}

            # Calculate the total percentage for the tumour mask
            total_percentage = sum(percentages.get(mask, 0) for mask in mask_color)

            # Get a color from the 'Reds' colormap based on the total percentage
            color = plt.get_cmap(color_map)(Normalize(0, 100)(total_percentage))
            # Create a polygon and fill it with the color
            polygon_array = np.array(polygon)
            plt.plot(polygon_array[:, 0], polygon_array[:, 1], color=color, linewidth=0.3)
            polygon_patch = patches.Polygon(np.array(polygon), facecolor=color, edgecolor='black', linewidth=0.3)

            # Add the polygon to the plot
            ax.add_patch(polygon_patch)

        # Show the plot
        if save_path: plt.savefig(save_path)
        if show: plt.show()
        return fig, ax