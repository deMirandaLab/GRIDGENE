import logging
import time
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import center_of_mass

from skimage.measure import label, regionprops


def save_labeled_mask_image(mask, save_path):
    # Create a figure and axis to plot the mask
    fig, ax = plt.subplots()

    # Plot the mask
    ax.imshow(mask)

    # For each label, find its center of mass and plot the label number
    for region in regionprops(mask):
        y, x = center_of_mass(region.image)
        ax.text(x, y, str(region.label), color='white')

    # Save the figure with the mask name
    fig.savefig(save_path)

    # Close the figure to free up memory
    plt.close(fig)

def get_hierarchy(original_mask,original_labels,
                  expansion_mask, expansion_labels):


    # Get the properties of the original and expansion objects
    original_props = regionprops(original_labels) # pass zero to ignore background
    expansion_props = regionprops(expansion_labels)
    hierarchy = {}

    # For each object in the expansion mask, find the original object that it belongs to
    for expansion_prop in expansion_props:
        if expansion_prop.label == 0:
            continue
        for original_prop in original_props:
            if original_prop.label == 0:
                continue
            if expansion_prop.bbox[0] <= original_prop.centroid[0] <= expansion_prop.bbox[2] and \
                    expansion_prop.bbox[1] <= original_prop.centroid[1] <= expansion_prop.bbox[3]:

                # The centroid of the expansion is inside the original object
                if expansion_prop.label not in hierarchy:
                    hierarchy[expansion_prop.label] = []
                hierarchy[expansion_prop.label].append(original_prop.label)

    return hierarchy



class GetMaskProperties:
    def __init__(self, mask, mask_labels, array_counts, target_dict, mask_tum = None, mask_stroma=None,
                 logger=None):

        self.mask = mask
        self.properties = {}
        self.array_counts, self.target_dict = array_counts, target_dict
        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        if mask_tum is not None and mask_stroma is not None:
            mask_tum = mask_tum.astype(np.uint8)
            mask_stroma = mask_stroma.astype(np.uint8)

            self.mask_tum = mask_tum
            self.mask_stroma = mask_stroma

        self.original_labels = mask_labels
        if self.original_labels is None:
            self.original_labels = label(self.mask)


    def get_tum_stroma_annotations(self):
        if self.mask_tum is None and self.mask_stroma is None:
            self.logger.error('Both mask_tum and mask_stroma are None.'
                              'Setting mask_tum and mask_stroma to None')
            self.mask_tum = None
            self.mask_stroma = None
            return None

        self.mask_tum_labels = label(self.mask_tum)
        self.mask_stroma_labels = label(self.mask_stroma)

        for object_id in np.unique(self.original_labels[self.original_labels != 0]):
            object_mask = (self.original_labels == object_id)

            # Calculate the intersection with mask_tum and mask_stroma
            tum_pixels = np.sum(object_mask * self.mask_tum)
            stroma_pixels = np.sum(object_mask * self.mask_stroma)

            # Find the IDs of the objects in mask_tum and mask_stroma that the object overlaps with
            tum_ids = np.unique(self.mask_tum_labels[object_mask])
            stroma_ids = np.unique(self.mask_stroma_labels[object_mask])

            # Ignore the background (ID 0)
            tum_ids = tum_ids[tum_ids != 0]
            stroma_ids = stroma_ids[stroma_ids != 0]

            # Store the properties in the self.properties dictionary
            self.properties[object_id] = {
                'tum_pixels': tum_pixels,
                'stroma_pixels': stroma_pixels,
                'tum_ids': tum_ids.tolist(),
                'stroma_ids': stroma_ids.tolist()
            }

    def get_morpho_properties(self):
        for object_id in np.unique(self.original_labels[self.original_labels != 0]):
            object_mask = (self.original_labels == object_id)

            # Calculate the area, perimeter, and centroid of the object
            prop = regionprops(object_mask.astype(np.int16))[0]

            # Store the properties in the self.properties dictionary
            self.properties[object_id].update({
                'area': prop.area,
                'perimeter': prop.perimeter,
                'centroid': prop.centroid,
                'min_x': prop.bbox[1],
                'min_y': prop.bbox[0],
                'max_x': prop.bbox[3],
                'max_y': prop.bbox[2],
                'vertices': len(prop.coords),
                'BoundingBox': [prop.bbox],

            })


    def get_counts_properties(self):
        for object_id in np.unique(self.original_labels[self.original_labels != 0]):
            object_mask = (self.original_labels == object_id)

            # Calculate counts for each gene over the current object
            gene_counts = np.sum(self.array_counts * object_mask[:, :, None], axis=(0, 1)).astype(np.int64)
            gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in
                               self.target_dict.items()}

            # Store the gene counts in the self.properties dictionary
            self.properties[object_id].update({
                **gene_count_dict
            })


    def get_tum_distance_properties(self):
            """
            Calculate the minimum distance from each object to the tumour border
            get_distance_to_tum method calculates the distance from each object in
             the first mask to the closest border of the tum mask and stores these
             distances in the self.properties dictionary.

            :param self:
            :return:
            """
            # Invert the tum mask
            inverted_tum_mask = np.logical_not(self.mask_tum)

            # Calculate the distance transform of the inverted tum mask
            distance_transform = distance_transform_edt(inverted_tum_mask)

            for object_id in np.unique(self.original_labels[self.original_labels != 0]):
                object_mask = (self.original_labels == object_id)

                # Find the minimum distance value within the object's area
                min_distance = np.min(distance_transform[object_mask])

                # Store the minimum distance in the self.properties dictionary
                self.properties[object_id]['min_distance_to_tum_border'] = min_distance


class GetMaskPropertiesBulk:
    def __init__(self, mask,array_counts, target_dict, logger=None):


        self.mask = mask
        self.properties = {'1': {}}
        self.array_counts, self.target_dict = array_counts, target_dict

        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def get_counts_properties(self):
        gene_counts = np.sum(self.array_counts * self.mask[:, :, None], axis=(0, 1)).astype(np.int64)
        gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in
                           self.target_dict.items()}

        # Store the gene counts in the self.properties dictionary
        self.properties['1'].update({
            **gene_count_dict
        })

    def get_morpho_properties(self):
        # Calculate the area, perimeter, and centroid of the object
        prop = regionprops(self.mask.astype(np.int8))[0]

        # Store the properties in the self.properties dictionary
        self.properties['1'].update({
            'area': prop.area,
            'perimeter': prop.perimeter,
            'centroid': prop.centroid,
            'min_x': prop.bbox[1],
            'min_y': prop.bbox[0],
            'max_x': prop.bbox[3],
            'max_y': prop.bbox[2],
            'vertices': len(prop.coords),
            'BoundingBox': [prop.bbox],

        })


class GetMasksProperties:
    def __init__(self, masks_dict, array_counts,target_dict, mask_tum = None, mask_stroma=None,
                 logger=None, image_shape=None, save_path = None):
        """
        Initialize the GetMasksProperties class.

        :param masks_dict: List of dictionaries containing mask information. Each dictionary should have the keys:
                           'mask', 'mask_name', 'per_object', and 'level_hierarchy'.

                            'mask': The mask image as a numpy array.
                            'mask_name': The name of the mask.
                            'per_object': Boolean indicating whether to calculate properties per object, or agglomerate
                            all objects in the same mask.
                            'level_hierarchy': Integer indicating the hierarchy level of the mask.
                                if level hierarchy == 1: this would be the mask from where the hierarchy will relate to
                                if level_hierarchy == 2: this would be the mask that will be expanded and relate to a mask hierarchy name 1
                                if level_hierarchy == None: not run hierarchy

        :param array_counts: Array of counts for each mask.
        :param target_dict: Dictionary mapping target names to indices in the array_counts.
        :param mask_tum: Optional. Tumour mask.
        :param mask_stroma: Optional. Stroma mask.
        :param logger: Optional. Logger instance for logging information. If None, a default logger is configured.
        :param image_shape: Optional. Shape of the image as a tuple (height, width).
        :param save_path: Optional. Path to save the labeled mask images.
        """
        self.masks_dict = masks_dict
        # {mask, mask_name, per_object, level_hierarchy}

        self.mask_tum = mask_tum
        self.mask_stroma = mask_stroma
        self.array_counts, self.target_dict = array_counts, target_dict

        self.image_shape = image_shape
        self.height = self.image_shape[0] if self.image_shape is not None else None
        self.width = self.image_shape[1] if self.image_shape is not None else None
        self.logger = logger
        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.save_path = save_path


    def hierarchical_analysis(self):
        """
        Perform hierarchical analysis on the masks.

        This method identifies the original mask (level_hierarchy == 1) and performs hierarchical analysis
        to determine the relationships between the original mask and expansion masks (level_hierarchy == 2).
        """

        # todo
        #  probably check if more than 1 hierarchy

        original_mask_dict = next(
            (mask_dict for mask_dict in self.masks_dict if mask_dict['level_hierarchy'] == 1), None)

        if original_mask_dict is not None:
            self.logger.info('running hierarchy analysis')
            original_mask = original_mask_dict['mask']
            original_labels = original_mask_dict['label_masks']

        else:
            self.logger.info('No mask with hierarchy level 1. Not running hierarchy analysis.')
            for mask_dict in self.masks_dict:
                mask_dict['hierarchies'] = None
                return

        for mask_dict in self.masks_dict:
            if mask_dict['level_hierarchy'] == 1:
                # Set hierarchies to the IDs of the objects in the mask
                mask_dict['hierarchies'] = {label: [label] for label in np.unique(mask_dict['label_masks']) if
                                            label != 0}
            elif mask_dict['level_hierarchy'] == 2:
                # run the hierarchy analysis to the index 1
                mask_dict['hierarchies'] = get_hierarchy(original_mask, original_labels,
                                         expansion_mask=mask_dict['mask'],
                                         expansion_labels=mask_dict['label_masks'])
            else:
                mask_dict['hierarchies'] = None

    def run(self):
        """
        Execute the analysis on the masks and save the results.

        This method processes each mask in masks_dict, performs hierarchical analysis, and calculates properties
        for each mask. The results are saved in a DataFrame.
        """

        start_time = time.time()  # Start time before the method execution

        for mask_dict in self.masks_dict:
            mask_dict['properties'] = None
            mask_dict['label_masks'] = label(mask_dict['mask'])
            mask_dict['hierarchies'] = None

            # Save the labeled mask image
            if self.save_path is not None:
                save_mask_label = os.path.join(self.save_path, f'{mask_dict["mask_name"]}.png')
                save_labeled_mask_image(mask_dict['label_masks'], save_mask_label)

        self.hierarchical_analysis()
        self.df_results_total = pd.DataFrame()

        for mask_dict in self.masks_dict:
            mask = mask_dict['mask']
            mask_name = mask_dict['mask_name']
            per_object = mask_dict['per_object']
            label_masks = mask_dict['label_masks']
            level_hierarchy = mask_dict['level_hierarchy']
            hierarchies = mask_dict['hierarchies']
            if np.unique(mask).size == 1:
                self.logger.info(f"Mask {mask_name} is empty. Skipping.")
                continue

            if per_object:
                MP = GetMaskProperties(mask, label_masks,
                                       array_counts=self.array_counts,
                                       target_dict=self.target_dict,
                                       mask_tum = self.mask_tum,
                                       mask_stroma = self.mask_stroma,
                                       logger = self.logger)

                if self.mask_tum is not None and self.mask_stroma is not None:
                    self.logger.info('Getting tumour and stroma annotations')
                    MP.get_tum_stroma_annotations()

                else:
                    self.logger.error('Both mask_tum and mask_stroma are None.'
                                      'Setting mask_tum and mask_stroma to None')


                MP.get_morpho_properties()
                MP.get_counts_properties()
                mask_dict['properties'] = MP.properties

            else:
                GMBulk = GetMaskPropertiesBulk(mask,array_counts = self.array_counts,
                                               target_dict = self.target_dict,
                                               logger = self.logger)
                GMBulk.get_morpho_properties()
                GMBulk.get_counts_properties()
                mask_dict['properties'] = GMBulk.properties

            df_mask = pd.DataFrame(mask_dict['properties']).fillna(0).T
            df_mask['mask_name'] = mask_name
            df_mask['per_object'] = per_object
            df_mask['level_hierarchy'] = level_hierarchy
            df_mask['hierarchies'] = hierarchies
            # r append and then concat once?
            self.df_results_total = pd.concat([self.df_results_total, df_mask], ignore_index=True)

        end_time = time.time()  # End time after the method execution
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        self.logger.info(f'The run method took {elapsed_time} seconds to execute')  # Log the elapsed time

    def save_df(self, path):
        self.df_results_total.to_csv(path, index=False)
        logging.info(f"Dataframe saved to {path}")

    def check_counts(self):
        positive_values_in_array = np.count_nonzero(self.array_counts)
        sum_of_dataframe = self.df_results_total[self.target_dict.keys()].values.sum()
        logging.info(f"Positive values in array: {positive_values_in_array}")
        logging.info(f"Sum of DataFrame: {sum_of_dataframe}")
        self.counts_in_array = positive_values_in_array
        self.counts_df = sum_of_dataframe
        self.check_counts_difference = positive_values_in_array - sum_of_dataframe
        return self.check_counts_difference

    def time(self):
        # get the time for mask and overall
        pass





