import logging
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from typing import Dict, List, Tuple, Union
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

class GetMasks:
    def __init__(self, logger=None, image_shape = None):
        """
        Initialize the GetMasks class.

        :param logger: Logger instance for logging messages. If None, a default logger is configured.
        :param image_shape: Tuple representing the shape of the image (height, width).
        """

        self.image_shape = image_shape
        self.height = self.image_shape[0] if self.image_shape is not None else None
        self.width = self.image_shape[1] if self.image_shape is not None else None

        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def filter_mask_by_area(self, mask, min_area):
        """
        Filter the mask by removing connected components with an area smaller than min_area.

        :param mask: Input mask to be filtered.
        :param min_area: Minimum area threshold for connected components to be retained.
        :return: Filtered mask with only the valid connected components.
        """

        # Compute the area of each connected component in the mask
        labels, counts = np.unique(mask, return_counts=True)

        # Filter out background label (usually 0)
        labels = labels[1:]
        counts = counts[1:]

        # Find labels whose area meets the minimum requirement
        valid_labels = labels[counts >= min_area]

        # Create a new mask containing only the valid labels
        filtered_mask = np.isin(mask, valid_labels).astype(np.uint8)
        self.logger.info(f'Filtering mask by area: {min_area}')

        return filtered_mask

    def get_masks_T_S_empty(self,contours_tum, contours_empty,filter_area = None):
        """
        Define the main parts of the image: Tum, Stroma, Empty
        Stroma is defined as the area minus the Tum and minus the Empty.

        :param contours_tum: Contours for the tumor regions.
        :param contours_empty: Contours for the empty regions.
        :param filter_area: Minimum area threshold for filtering the stroma mask.
        """

        if self.height is None or self.width is None:
            self.logger.error("Image shape is not defined.")
            return
        self.contours_T = contours_tum
        self.contours_E = contours_empty

        # Define tum and stroma (stroma cannot contain tum)
        self.mask_T = np.zeros((self.height, self.width), dtype=np.uint8)
        self.mask_empty = np.zeros((self.height, self.width), dtype=np.uint8)
        self.mask_S = np.ones((self.height, self.width), dtype=np.uint8)

        # Draw contours on the masks
        cv2.drawContours(self.mask_T, self.contours_T, -1, 1, thickness=cv2.FILLED)
        cv2.drawContours(self.mask_empty, self.contours_E, -1, 1, thickness=cv2.FILLED)

        # Create the negative of the two masks
        self.mask_S = cv2.subtract(self.mask_S, self.mask_T)
        self.mask_S = cv2.subtract(self.mask_S, self.mask_empty)
        if filter_area is not None:
            self.logger.info(f'Filtering stroma mask by area: {filter_area}')
            self.mask_S = self.filter_mask_by_area(self.mask_S, min_area=filter_area)

            # todo above just filtering the stroma mask, should be filtering the tumour mask as well??


    def save_masks_npy(self, mask, save_path):
        """
       Save the mask as a .npy file.

       :param mask: Mask to be saved.
       :param save_path: Path where the mask will be saved.
       """
        np.save(save_path, mask)
        self.logger.info(f'mask saved at {save_path}')

    def save_masks(self, mask, path):
        """
        Save the mask as an image file.

        :param mask: Mask to be saved.
        :param path: Path where the mask will be saved.
        """

        cv2.imwrite(os.path.join(path), mask * 255)
        self.logger.info(f'mask saved at {os.path.join(path)}')

    def plot_masks(self, masks, mask_names, background_color=(0, 0, 0), mask_colors=None, path=None, show=True, ax=None, figsize=(10, 10)):
        """
        Plots the given masks with their corresponding names.

        :param masks: List of masks to plot.
        :param mask_names: List of names corresponding to the masks.
        :param background_color: Tuple to use for areas not assigned in any mask.
        :param mask_colors: Dictionary mapping mask names to colors.
        :param path: Directory path where the plots will be saved.
        :param show: Whether to display the plot.
        :param ax: Matplotlib axis object. If None, a new figure will be created.
        :param figsize: Tuple representing the figure size (width, height) in inches.
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

        # Display the image
        if ax is None:
            plt.figure(figsize=figsize)
            plt.imshow(background, origin='lower')
        else:
            ax.imshow(background, origin='lower')

        # Add the legend
        if ax is None:
            plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', bbox_transform=plt.gcf().transFigure)
        else:
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', bbox_transform=ax.transAxes)

        if path is not None:
            save_path = os.path.join(path, f'masks_{"_".join(mask_names).replace(" ", "").lower()}.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        if show:
            if ax is None:
                plt.imshow(background, origin='lower')
            else:
                ax.imshow(background, origin='lower')
            plt.show()
        return

    # todo how to do this keeping non overapping
    # i deleted the empty but check
    def apply_morphological_operations(self, kernel_size=3, iterations=1, min_area=5000, set_as_self=True):
        pass

class TumBorderAnalysis(GetMasks):
    """
    Class to define the tumour border analysis
    """
    def __init__(self, get_masks_instance):
        self.get_masks_instance = get_masks_instance

        if self.get_masks_instance.mask_T is None:
            self.get_masks_instance.logger.error('tumour mask not defined. Returning')
            return

        self.mask_S = self.get_masks_instance.mask_S
        self.mask_T = self.get_masks_instance.mask_T
        self.height = self.get_masks_instance.height
        self.width = self.get_masks_instance.width
        self.logger = self.get_masks_instance.logger

        self.mask_T_tborder = None
        self.mask_S_tborder = None
        self.mask_TB_tborders = None


    # TODO should the filter area be by the tumour or the border????
    def get_mask_tumour_border(self, expansions_pixels=[], filter_area=None):
        """
        Expands the tumour border by a specified number of pixels and returns the updated masks.

        This method expands the tumour mask by the specified number of pixels in `expansions_pixels`.
        It then updates the stroma mask by subtracting the expanded tumour regions and returns the
        updated masks for tumour, stroma, and the expanded tumour borders.

        :param expansions_pixels: List of integers specifying the number of pixels to expand the tumour border.
        :param filter_area: Minimum area threshold for filtering the expanded tumour border masks. If None, no filtering is applied.
        :return: None. Updates the following instance attributes:
            - self.mask_T_tborder: Updated tumour mask.
            - self.mask_S_tborder: Updated stroma mask.
            - self.mask_TB_tborders: List of masks for the expanded tumour borders.
        """
        assigned = self.mask_T
        masks_T_borders = []
        # Iterate over each kernel size
        for kernel_size in expansions_pixels:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            mask_T_border = cv2.dilate(self.mask_T, kernel, iterations=1)
            mask_T_border = cv2.bitwise_and(mask_T_border, self.mask_S)  # only if they are in stroma
            mask_T_border = cv2.subtract(mask_T_border, assigned)  # subtract the tum
            if filter_area is not None:
                self.logger.info(f'Filtering Border mask by area: {filter_area}')
                mask_T_border = self.filter_mask_by_area(mask_T_border, min_area=filter_area)

            assigned = cv2.bitwise_or(assigned, mask_T_border)
            masks_T_borders.append(mask_T_border)


        # Subtract the combined mask from the original mask stroma
        mask_remaining_S = cv2.subtract(self.mask_S, assigned)

        self.mask_T_tborder = self.mask_T
        self.mask_S_tborder = mask_remaining_S
        self.mask_TB_tborders = masks_T_borders
        return

class SingleClassObjectAnalysis(GetMasks):
    """
    A class to perform analysis on single class objects within masks, including creating masks for specific objects
    and expanding these masks.

    Attributes:
        get_masks_instance (GetMasks): An instance of the GetMasks class.
        mask_T (np.ndarray): Tumour mask.
        mask_S (np.ndarray): Stroma mask.
        height (int): Height of the mask.
        width (int): Width of the mask.
        logger (logging.Logger): Logger instance for logging information.
        mask_T_SA (np.ndarray): Tumour mask after single object analysis.
        mask_S_SA (np.ndarray): Stroma mask after single object analysis.
        mask_object_SA (np.ndarray): Mask for the specific object after single object analysis.
        masks_object_expansions (list): List of masks for the object expansions.
        contours_object (list): Contours defining the object area.
        contour_name (str): Name of the contour.
    """
    # todo change the name of this class
    def __init__(self, get_masks_instance, contours_object, contour_name = ''):
        """
        Initialize the SingleClassObjectAnalysis class.

        :param get_masks_instance: An instance of the GetMasks class.
        :param contours_object: Contours defining the object area.
        :param contour_name: Optional. Name of the contour.
        """
        self.get_masks_instance = get_masks_instance

        # Check if tumour and stroma masks are defined
        if hasattr(self.get_masks_instance, 'mask_T'):
            self.mask_T = self.get_masks_instance.mask_T
        else:
            self.mask_T = None
            self.get_masks_instance.logger.info('Tumour mask not defined.')

        if hasattr(self.get_masks_instance, 'mask_S'):
            self.mask_S = self.get_masks_instance.mask_S
        else:
            self.mask_S = None
            self.get_masks_instance.logger.info('Stroma mask not defined.')

        self.height = self.get_masks_instance.height
        self.width = self.get_masks_instance.width
        self.logger = self.get_masks_instance.logger


        self.mask_T_SA = None
        self.mask_S_SA = None
        self.mask_object_SA = None
        self.masks_object_expansions = None
        self.contours_object = contours_object
        self.contour_name = contour_name

    def get_mask_objects(self, exclude_masks=None, filter_area=None):
        """
        Create a mask for specific objects, with optional exclusion from specified masks.

        :param exclude_masks: List of masks to exclude the objects from (e.g., ['T', 'empty']).
        :param filter_area: Optional. Minimum area to filter the mask.
        :return: None
        """
        mask_object = np.zeros((self.height, self.width), dtype=np.uint8)

        # Draw contours on the mask
        cv2.drawContours(mask_object, self.contours_object, -1, 1, thickness=cv2.FILLED)

        # Exclude areas defined in exclude_masks
        # exclude zones where the mask is in empty or in tumour
        if exclude_masks:
            for mask_ in exclude_masks:
                mask_object = cv2.subtract(mask_object, mask_)
        if filter_area is not None:
            self.logger.info(f'Filtering stroma mask by area: {filter_area}')
            mask_object = self.get_masks_instance.filter_mask_by_area(mask_object, min_area=filter_area)

        self.mask_object_SA = mask_object
        self.logger.info(f'Mask for objects created with exclusions: {exclude_masks}. '
                         f'Pixels of objects inside {exclude_masks} were removed')


        self.logger.info(f'creating non overlaping tumour and stroma mask of '
                         f'single object analysis. ')

        if self.mask_T is not None:
            self.mask_T_SA = cv2.subtract(self.mask_T, self.mask_object_SA)
        if self.mask_S is not None:
            self.mask_S_SA = cv2.subtract(self.mask_S, self.mask_object_SA)

    def get_objects_expansion(self, expansions_pixels=[], exclude_masks=None, filter_area=None): # add empty as default?
        """
        Expand the mask for specific objects, with optional exclusion from specified masks.

        :param expansions_pixels: List of kernel sizes for expansion.
        :param exclude_masks: Optional. List of masks to exclude the objects from.
        :param filter_area: Optional. Minimum area to filter the mask.
        :return: None
        """
        if self.mask_object_SA is None:
            self.logger.error('No mask of object analysis defined to expand. Returning')
            return

        if exclude_masks:
            self.logger.info(f'Mask for objects created with exclusions for {len(exclude_masks)}. '
                             f'masks. Pixels of objects inside these will be removed')

        # assigned = np.zeros((self.height, self.width), dtype=np.uint8)
        assigned = self.mask_object_SA

        masks_objects_expansions = []
        # Iterate over each kernel size
        for kernel_size in expansions_pixels:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            mask_object_close = cv2.dilate(self.mask_object_SA, kernel, iterations=1)

            mask_object_close = cv2.subtract(mask_object_close, assigned,
                                                 dtype=cv2.CV_8U)
            if exclude_masks:
                self.logger.info(f'Excluding pixels from {len(exclude_masks)} masks.')
                for mask_ in exclude_masks:
                    mask_object_close = cv2.subtract(self.mask_object_SA, mask_)

            if filter_area is not None:
                self.logger.info(f'Filtering stroma mask by area: {filter_area}')
                mask_object_close = self.get_masks_instance.filter_mask_by_area(mask_object_close, min_area=filter_area)

            assigned =  cv2.bitwise_or(assigned, mask_object_close)
            masks_objects_expansions.append(mask_object_close)

        for mask_expansion in masks_objects_expansions:
            if self.mask_T_SA is not None:
                self.logger.info(f'creating non overlaping tumour mask of '
                                 f'single object analysis. ')
                self.mask_T_SA = cv2.subtract(self.mask_T_SA, mask_expansion)
            if self.mask_S_SA is not None:
                self.logger.info(f'creating non overlaping stroma mask of '
                                 f'single object analysis. ')
                self.mask_S_SA = cv2.subtract(self.mask_S_SA, mask_expansion)


        self.masks_object_expansions = masks_objects_expansions
        return


# todo need some fixing , improvment and testing
# todo fix soe points are not defined. check why. related to infinite edges?

# todo uniformize in relation to single. the dicts and how things are done
class MultiClassObjectAnalysis(GetMasks):
    def __init__(self,get_masks_instance, multiple_contours: Dict,  save_path: str = None): #mask_T: np, mask_S: np, height: int, width: int,
        self.get_masks_instance = get_masks_instance

        self.height = self.get_masks_instance.height
        self.width = self.get_masks_instance.width
        self.logger = self.get_masks_instance.logger

        # Check if tumour and stroma masks are defined
        if hasattr(self.get_masks_instance, 'mask_T'):
            self.mask_T = self.get_masks_instance.mask_T
        else:
            self.mask_T = None
            self.get_masks_instance.logger.info('Tumour mask not defined.')

        if hasattr(self.get_masks_instance, 'mask_S'):
            self.mask_S = self.get_masks_instance.mask_S
        else:
            self.mask_S = None
            self.get_masks_instance.logger.info('Stroma mask not defined.')

        self.multiple_contours = multiple_contours
        self.masks = None
        self.vor = None
        self.list_of_polygons = None
        self.class_labels = None
        self.all_centroids = None
        # self.mask_T = mask_T
        # self.mask_S = mask_S
        self.save_path = save_path

        for class_label, contours in self.multiple_contours.items():
            for i, contour in enumerate(contours):
                # Reverse the order of points in each contour
                self.multiple_contours[class_label][i] = contour[::-1]

    def get_polygons_from_contours(self, contours):
        # Create a list to store the polygons
        list_of_polygons = []

        # Iterate over each contour
        for contour_data in contours:
            # Create a Polygon object from the contour coordinates
            polygon = Polygon(contour_data)

            # Add the polygon to the list
            list_of_polygons.append(polygon)

        return list_of_polygons

    def derive_voronoi_from_contours_approximate(self):

        """

        :return:
        """


        all_contours = [contour for contour_points in self.multiple_contours.values() for contour in contour_points]
        list_of_polygons = self.get_polygons_from_contours(all_contours)


        # # Step 1: Compute the centroid of each polygon
        # centroids = [polygon.centroid for polygon in list_of_polygons]
        #
        # # Step 2: Collect all the centroids
        # all_centroids = np.array([(centroid.x, centroid.y) for centroid in centroids])

        # Step 1: Compute the centroid of each polygon and assign class labels
        centroids = []
        class_labels = []
        for class_label, contours in self.multiple_contours.items():
            for contour in contours:
                polygon = Polygon(contour)
                centroids.append(polygon.centroid)
                class_labels.append(class_label)

        # Step 2: Collect all the centroids and class labels
        all_centroids = np.array([(centroid.x, centroid.y) for centroid in centroids])

        # Step 3: Compute the Voronoi diagram using the centroids
        vor = Voronoi(all_centroids)

        # self.clip_voronoi_to_bbox()
        self.list_of_polygons = list_of_polygons
        self.class_labels = class_labels
        self.vor = vor
        self.all_centroids = all_centroids


    def get_voronoi_mask(self, category_name):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        vor =  self.vor
        # Iterate over centroids or polygons
        for idx, (centroid, polygon, label) in enumerate(zip(self.all_centroids, self.list_of_polygons,
                                                             self.class_labels)):
            # Check if the centroid or polygon has the target label
            if label == category_name:
                # Get the region index associated with the centroid or polygon
                region_index = vor.point_region[idx]
                # If the region index is valid, fill it in the mask
                if region_index != -1:
                    region_vertices = [vor.vertices[vertex] for vertex in vor.regions[region_index] if vertex != -1]
                    cv2.fillPoly(mask, [np.array(region_vertices, dtype=np.int32)], color=255)

        return mask # voronoi mask
    def expand_mask(self, mask, expansion_distance):
        # todo probably shoudl be the same method as above Single object

        kernel = np.ones((expansion_distance, expansion_distance), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        expanded_mask = cv2.subtract(expanded_mask, mask)

        return expanded_mask

    def generate_expanded_masks_limited_by_voronoi(self,expansion_distances):
        # todo add the part of rmeoving in S and T masks
        # TODO FILTERING area and from a mask
        # todo also check if masks are not overapping and stuff
        """

        :param self:
        :param contours_dict:
        :param expansion_distances:
        :return:
        """
        # Step 1: Generate masks for each contour
        masks = {}
        for contour_name, contour_points in self.multiple_contours.items():
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(mask, contour_points, -1, 255, thickness=cv2.FILLED)
            masks[contour_name] = mask

        # Step 2: Generate expanded masks for each category
        expanded_masks = {}
        for category_name, mask in masks.items():
            voronoi_mask = self.get_voronoi_mask(category_name)
            previous_expansion_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for expansion_distance in expansion_distances:
                current_expansion_mask = self.expand_mask(mask.copy(), expansion_distance)
                # Ensure the current expansion doesn't overlap with previous ones
                current_expansion_mask = cv2.bitwise_and(current_expansion_mask,
                                                         cv2.bitwise_not(previous_expansion_mask))
                # Apply Voronoi mask
                current_expansion_mask = cv2.bitwise_and(current_expansion_mask, voronoi_mask)
                expanded_masks[f'{category_name}_expansion_{expansion_distance}'] = current_expansion_mask
                # Update previous expansion mask
                previous_expansion_mask = cv2.bitwise_or(previous_expansion_mask, current_expansion_mask)

        # Step 3: Update the final dictionary
        masks.update(expanded_masks)
        self.masks = masks

        self.subtract_to_T_S()

        return masks

    def subtract_to_T_S(self):
        assigned = np.zeros((self.height, self.width), dtype=np.uint8)
        for mask in self.masks.values():
            assigned = cv2.bitwise_or(assigned, mask)

        if self.mask_T is not None:
            self.logger.info(f'creating non overlaping tumour mask of '
                             f'single object analysis. ')
            self.mask_T = cv2.subtract(self.mask_T, assigned)
            self.masks['Tumour_remaining'] = self.mask_T

        if self.mask_S is not None:
            self.logger.info(f'creating non overlaping stroma mask of '
                             f'single object analysis. ')
            self.mask_S = cv2.subtract(self.mask_S, assigned)
            self.masks['Stroma_remaining'] = self.mask_S
        return

    # todo fix so accepts other masknames
    # todo fix needds to be as flexible and with axs as the ones above-
    # accept colors

    def plot_contours_expansion_with_voronoi_edges(self, mask_colors,
                                                   background_color = (255,255,255),
                                                   show=True, axes=None, figsize=(8, 8)):
        masks = self.masks
        mask_names = list(self.masks.keys())

        # Create a background image filled with the background color
        background = np.full((self.height, self.width, 3), background_color)

        fig, ax = plt.subplots(figsize=figsize)

        # List to store legend patches
        legend_patches = []

        # Add each mask to the background image
        for mask_name, mask in masks.items():
            # Choose a color for the mask
            mask_color = np.array(
                mask_colors.get(mask_name, (128, 128, 128)))  # Default gray if mask_name not in mask_colors

            # Apply the mask color to the background where the mask is present
            background[mask != 0] = mask_color

            # Create a legend patch with the mask color
            legend_patches.append(mpatches.Patch(color=mask_color / 255, label=mask_name))

        # # Plot Voronoi edges

        # # Plot Voronoi edges
        voronoi_plot_2d(self.vor, ax=ax, show_vertices=False, line_colors='black')

        ax.imshow(background, origin='lower')

        # Add legend outside the plot area
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', bbox_transform=ax.transAxes)

        # Save the figure if a save path is defined
        if getattr(self, 'save_path', None):
            save_path = os.path.join(self.save_path, 'masks_with_voronoi_edges.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        # Show the plot if specified
        if show:
            plt.show()

        return ax if axes is not None else None



        # # Create figure and axis if not provided
        # if ax is None:
        #     fig, ax = plt.subplots(figsize=figsize)
        # ax.set_facecolor('White')
        #
        # # Iterate over masks
        # for mask_name, mask in masks.items():
        #     # Get color based on the mask name; default to gray if not in mask_colors
        #     color = mask_colors.get(mask_name, (128, 128, 128))  # Default gray if mask_name not in mask_colors
        #     normalized_color = np.array(color) / 255.0  # Convert to [0, 1] range
        #     cmap = ListedColormap([normalized_color])
        #     # Display the mask with the specified color
        #     ax.imshow(mask, cmap = cmap, alpha=0.5, vmin=0, vmax=1, extent=[0, mask.shape[1], 0, mask.shape[0]],
        #               origin='lower')
        #
        #
        # # Set background color
        # ax.set_facecolor('White')
        #
        # # Set title
        # ax.set_title('Masks with Voronoi Edges')
        #
        # # Hide axis
        # ax.axis('off')
        #
        # # Create legend dynamically based on mask_colors
        # legend_elements = []
        # for mask_name, color in mask_colors.items():
        #     legend_elements.append(
        #         Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color) / 255, markersize=10,
        #                label=mask_name))
        #
        # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        #
        #
        # if self.save_path is not None:
        #     save_path = os.path.join(self.save_path, 'masks_with_voronoi_edges.png')
        #     plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        #     self.logger.info(f'Plot saved at {save_path}')
        #
        # if show:
        #     plt.show()
        #
        # return ax if ax is not None else fig
