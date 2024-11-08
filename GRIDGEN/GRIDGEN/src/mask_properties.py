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

def plot_labeled_masks(mask_dict, show=False):
    """
    Plot the labeled mask with colored objects and bounding boxes.
    Parameters
    ----------
    mask_dict : dict (required)
    show    : bool (optional)

    Returns
    -------

    """
    mask = mask_dict['mask']
    label_mask = mask_dict['label_masks']
    unique_labels = np.unique(label_mask)

    # Generate random colors for each label using a colormap
    colormap = cm.get_cmap('tab10', len(unique_labels))
    colors = {label: colormap(i) for i, label in enumerate(unique_labels) if label != 0}

    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.float32)
    for label in unique_labels:
        if label == 0:
            continue
        colored_mask[label_mask == label] = colors[label][:3]

    # Create a figure and axis to plot the mask
    fig, ax = plt.subplots()
    ax.imshow(colored_mask, origin='lower')

    # Plot each labeled object with its corresponding color and label number
    for region in regionprops(label_mask):
        if region.label == 0:
            continue
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor=colors[region.label], linewidth=2)
        ax.add_patch(rect)
        y, x = region.centroid
        ax.text(x, y, str(region.label), color='white', fontsize=8, ha='center', va='center')

    # Set the title and show the plot
    ax.set_title(mask_dict['mask_name'])

    # # Save the plot as a high-resolution image
    # save_file_path = os.path.join(save_path, f"label_{mask_dict['mask_name']}.png")
    # fig.savefig(save_file_path, dpi=dpi)

    # Show the plot if requested
    if show:
        plt.show()

    return fig, ax

def get_annotations(mask_base_labels, mask_to_annot_labels, mask_base_name):
    """
    Get annotations by comparing two masks and calculating the intersection properties.

    This function compares two masks, calculates the intersection properties, and returns a dictionary
    containing the number of pixels and the IDs of the objects in the base mask that overlap with each object
    in the annotation mask.

    Parameters
    ----------
    mask_base_labels : np.ndarray
        The base mask with labeled objects.
    mask_to_annot_labels : np.ndarray
        The mask to be annotated with labeled objects.
    mask_base_name : str
        The name of the base mask for labeling the properties.

    Returns
    -------
    dict
        A dictionary where keys are object IDs from the annotation mask and values are dictionaries containing
        the number of overlapping pixels and the IDs of the overlapping objects in the base mask.
    """
    properties = {}
    labels_to_annot = np.unique(mask_to_annot_labels[mask_to_annot_labels != 0])

    for object_id in labels_to_annot:
        # object_mask = mask_to_annot_labels[(mask_to_annot_labels == object_id)]
        # base_object = mask_base_labels[object_mask]
        object_mask = (mask_to_annot_labels == object_id)
        overlap_mask = object_mask & (mask_base_labels > 0)  # Only regions where both masks are non-zero
        base_object = mask_base_labels[overlap_mask]


        # Calculate the intersection with mask_tum and mask_stroma
        # tum_pixels = np.sum(object_mask * mask_base_labels)
        tum_pixels = np.sum(base_object>0)
        # Find the IDs of the objects in mask_tum and mask_stroma that the object overlaps with
        tum_ids = np.unique(base_object[base_object != 0])
        # Store the properties in the self.properties dictionary
        properties[object_id] = {
            f'{mask_base_name}_pixels': tum_pixels,
            f'{mask_base_name}_ids': tum_ids.tolist(),
        }
    return properties

# todo hierarchies
# def get_hierarchy(original_mask, original_labels, expansion_mask, expansion_labels):
#     # Get the properties of the original and expansion objects
#     original_props = regionprops(original_labels)
#     expansion_props = regionprops(expansion_labels)
#
#     print(f"Number of expansion objects: {len(expansion_props)}")
#     hierarchy = {}
#     bbox_margin = 2
#     max_distance = 10
#     # For each object in the expansion mask, find the original object it belongs to
#     for expansion_prop in expansion_props:
#         hierarchy[expansion_prop.label] = []
#         if expansion_prop.label == 0:
#             continue
#
#         # Extract the bounding box for the current expansion object
#         exp_bbox = expansion_prop.bbox
#         exp_label = expansion_prop.label
#         found_match = False  # Track if any original object matches this expansion
#
#         for original_prop in original_props:
#             if original_prop.label == 0:
#                 continue
#
#             original_centroid = original_prop.centroid
#
#             # Bounding box margin adjustment
#             if (exp_bbox[0] - bbox_margin <= original_centroid[0] <= exp_bbox[2] + bbox_margin) and \
#                     (exp_bbox[1] - bbox_margin <= original_centroid[1] <= exp_bbox[3] + bbox_margin):
#                 hierarchy[exp_label].append(original_prop.label)
#
#         # After loop, debug if no original label is found within expansion
#         if not hierarchy[exp_label]:
#             print(f"No original object found within expansion object {exp_label}")
#
#     print(f"Total matched hierarchies: {len([k for k, v in hierarchy.items() if v])}")
#     return hierarchy

class GetMaskPropertiesObject:
    """
    A class to calculate morphological and gene count properties for individual objects in a mask.

    Attributes:
        mask (np.ndarray): The mask image as a numpy array.
        properties (dict): Dictionary to store properties of each object.
        array_counts (np.ndarray): Array of counts for each mask.
        target_dict (dict): Dictionary mapping target names to indices in the array_counts.
        original_labels (np.ndarray): Labeled mask image.
        logger (logging.Logger): Logger instance for logging information.
    """

    def __init__(self, mask, mask_labels, array_counts, target_dict,
                 logger=None):
        """
        Initialize the GetMaskPropertiesObject class.

        :param mask: The mask image as a numpy array.
        :param mask_labels: Labeled mask image.
        :param array_counts: Array of counts for each mask.
        :param target_dict: Dictionary mapping target names to indices in the array_counts.
        :param logger: Optional. Logger instance for logging information. If None, a default logger is configured.
        """
        self.mask = mask
        self.properties = {}
        self.array_counts, self.target_dict = array_counts, target_dict
        self.original_labels =  mask_labels
        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


    def get_morpho_properties(self):
        """
        Calculate and store morphological properties for each object in the mask.

        The properties include area, perimeter, centroid, bounding box, and number of vertices.
        """

        for object_id in np.unique(self.original_labels[self.original_labels != 0]):
            object_mask = (self.original_labels == object_id)

            # Calculate the area, perimeter, and centroid of the object
            prop = regionprops(object_mask.astype(np.int16))[0]
            if object_id not in self.properties:
                self.properties[object_id] = {}
            # Store the properties in the self.properties dictionary
            self.properties[object_id].update({
                'object_id': object_id,
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
        """
        Calculate and store gene count properties for each object in the mask.

        The properties include counts for each gene over the current object.
        """
        # Precompute the mask for each object
        object_masks = {object_id: (self.original_labels == object_id) for object_id in np.unique(self.original_labels[self.original_labels != 0])}

        for object_id, object_mask in object_masks.items():
            object_mask = (self.original_labels == object_id)

            # Calculate counts for each gene over the current object
            gene_counts = np.sum(self.array_counts * object_mask[:, :, None], axis=(0, 1)).astype(np.int64)
            gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in
                               self.target_dict.items()}
            gene_count_dict['object_id'] = object_id
            # Store the gene counts in the self.properties dictionary
            self.properties[object_id].update({
                **gene_count_dict
            })

            # # Store the gene counts in the self.properties dictionary
            # if object_id not in self.properties:
            #     self.properties[object_id] = {}
            # self.properties[object_id].update(gene_count_dict)


class GetMaskPropertiesBulk:
    """
    A class to calculate morphological and gene count properties for a bulk mask.

    Attributes:
        mask (np.ndarray): The mask image as a numpy array.
        properties (dict): Dictionary to store properties of the mask.
        array_counts (np.ndarray): Array of counts for each mask.
        target_dict (dict): Dictionary mapping target names to indices in the array_counts.
        logger (logging.Logger): Logger instance for logging information.
    """

    def __init__(self, mask,array_counts, target_dict, logger=None):
        """
        Initialize the GetMaskPropertiesBulk class.

        :param mask: The mask image as a numpy array.
        :param array_counts: Array of counts for each mask.
        :param target_dict: Dictionary mapping target names to indices in the array_counts.
        :param logger: Optional. Logger instance for logging information. If None, a default logger is configured.
        """

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
        """
         Calculate and store gene count properties for the mask.

         The properties include counts for each gene over the entire mask.
         """

        gene_counts = np.sum(self.array_counts * self.mask[:, :, None], axis=(0, 1)).astype(np.int64)
        gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in
                           self.target_dict.items()}

        # Store the gene counts in the self.properties dictionary
        self.properties['1'].update({
            **gene_count_dict
        })

    def get_morpho_properties(self):
        """
        Calculate and store morphological properties for the mask.

        The properties include area, perimeter, centroid, bounding box, and number of vertices.
        """

        # Calculate the area, perimeter, and centroid of the object
        prop = regionprops(self.mask.astype(np.int8))[0]

        # Store the properties in the self.properties dictionary
        self.properties['1'].update({
            'object_id': 0,
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

        self.df_results_total = pd.DataFrame(columns=['mask_name', 'object_id'])


        # label the masks so the labelling is done once and is consistent
        for mask_dict in self.masks_dict:
            mask_dict['properties'] = None
            mask_dict['label_masks'] = label(mask_dict['mask'])
            mask_dict['hierarchies'] = None

            if self.save_path is not None:
                fig, ax = plot_labeled_masks(mask_dict, show=False)
                save_file_path = os.path.join(self.save_path, f"label_{mask_dict['mask_name']}.png")
                fig.savefig(save_file_path, dpi=300)
                plt.close(fig)

            if mask_dict['per_object']:
                unique_labels = np.unique(mask_dict['label_masks'])
                for object_id in unique_labels:
                    if object_id != 0:  # Exclude background
                        new_row = pd.DataFrame({'mask_name': [mask_dict['mask_name']], 'object_id': [object_id]})
                        self.df_results_total = pd.concat([self.df_results_total, new_row], ignore_index=True)
            else:
                new_row = pd.DataFrame({'mask_name': [mask_dict['mask_name']], 'object_id': 0})
                self.df_results_total = pd.concat([self.df_results_total, new_row], ignore_index=True)



    def get_gene_counts(self):
        """
        Calculate and store gene count properties for the masks.

        This method iterates over each mask in the masks_dict, calculates the gene counts, and updates the properties
        of each mask with the calculated values.
        The properties are calculated either for the entire mask or per object based on the 'per_object' attribute.
        """

        # todo separate the morphological? / todo better performance

        start_time = time.time()
        all_properties_df = pd.DataFrame()
        for mask_dict in self.masks_dict:
            mask = mask_dict['mask']
            mask_name = mask_dict['mask_name']
            per_object = mask_dict['per_object']
            label_masks = mask_dict['label_masks']
            if np.unique(mask).size == 1:
                self.logger.info(f"Mask {mask_name} is empty. Skipping.")
                continue

            if per_object:
                MP = GetMaskPropertiesObject(mask, label_masks,
                                       array_counts=self.array_counts,
                                       target_dict=self.target_dict,
                                       logger = self.logger)

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
            all_properties_df = pd.concat([all_properties_df, df_mask], ignore_index=True)

        # Merge all_annotations_df with df_results_total on object_id and mask_name
        self.df_results_total = pd.merge(
            self.df_results_total, all_properties_df,
            on=['object_id', 'mask_name'],
            how='outer',  # Keeps all data even if no match
            suffixes=('', '_annot')  # Handles any duplicate column names
        )

        end_time = time.time()  # End time after the method execution
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        self.logger.info(f'The run method took {elapsed_time} seconds ({elapsed_time/60} minutes) to execute')
        return self.df_results_total

    def get_mask_interest_annotations(self, base_mask_dicts):
        """
        Perform annotation analysis on the masks.

        This method iterates over each base mask in base_mask_dicts, calculates the annotations, and updates the
        properties of each mask with the calculated values.
        """

        start_time = time.time()

        if not base_mask_dicts:
            self.logger.info('No mask for annotation. Not running annotation analysis.')
            for mask_dict in self.masks_dict:
                mask_dict['annotations'] = None
            return

        self.logger.info('Running annotation analysis')

        for base_mask_dict in base_mask_dicts:
            if 'label_masks' not in base_mask_dict or base_mask_dict['label_masks'] is None or base_mask_dict[
                'label_masks'].size == 0:
                self.logger.info(f"Label Mask {base_mask_dict['mask_name']} is empty. Labelling.")
                base_mask_dict['label_masks'] = label(base_mask_dict['mask'])

            if self.save_path is not None:
                fig, ax = plot_labeled_masks(base_mask_dict, show=False)
                save_file_path = os.path.join(self.save_path, f"label_base_annot_{mask_dict['mask_name']}.png")
                fig.savefig(save_file_path, dpi=300)
                plt.close(fig)

            # Placeholder DataFrame to hold all annotation data
            all_annotations_df = pd.DataFrame()
            for mask_dict in self.masks_dict:
                if mask_dict['annotation'] == 2:
                    annotations = (
                        get_annotations(mask_base_labels = base_mask_dict['label_masks'],
                                        mask_to_annot_labels = mask_dict['label_masks'],
                                        mask_base_name = base_mask_dict['mask_name']))
                else:
                    annotations = None

                if annotations:
                    for object_id, props in annotations.items():
                        props['object_id'] = object_id
                        props['mask_name'] = mask_dict['mask_name']

                        # Convert to DataFrame and add to all_annotations_df
                        df_mask = pd.DataFrame(annotations).T

                else:
                    df_mask = pd.DataFrame({'mask_name': mask_dict['mask_name'], 'object_id': 0}, index=[0])

                all_annotations_df = pd.concat([all_annotations_df, df_mask], ignore_index=True)

            # Merge all_annotations_df with df_results_total on object_id and mask_name
            self.df_results_total = pd.merge(
                self.df_results_total, all_annotations_df,
                on=['object_id', 'mask_name'],
                how='outer',  # Keeps all data even if no match
                suffixes=('', '_annot')  # Handles any duplicate column names
            )
        end_time = time.time()  # End time after the method execution
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        self.logger.info(f'The annotation analysis took {elapsed_time:.2f} seconds to execute')

        return self.df_results_total

    def hierarchical_analysis(self):
        """
        Perform hierarchical analysis on the masks.

        This method identifies the original mask (level_hierarchy == 1) and performs hierarchical analysis
        to determine the relationships between the original mask and expansion masks (level_hierarchy == 2).
        """

        # todo
        #  probably check if more than 1 hierarchy
        start_time = time.time()
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
                mask_dict['hierarchies'] = get_hierarchy(original_labels,
                                         expansion_labels=mask_dict['label_masks'])
            else:
                mask_dict['hierarchies'] = None


            if mask_dict['hierarchies'] is not None:
                print(mask_dict['hierarchies'])
                none_count = sum(1 for key, value in mask_dict['hierarchies'].items() if value is None)
                print(none_count)


                mask_name = mask_dict['mask_name']
                hierarchies = mask_dict['hierarchies']
                for label, hierarchy in hierarchies.items():
                    hierarchy_str = ','.join(map(str, hierarchy))
                    self.df_results_total.loc[(self.df_results_total['mask_name'] == mask_name) &
                                              (self.df_results_total['label'] == label), 'hierarchy'] = hierarchy_str

        end_time = time.time()  # End time after the method execution
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        self.logger.info(f'The run method took {elapsed_time} seconds to execute')  # Log the elapsed time

        return self.df_results_total


    def check_counts(self):
        """
         Check the counts in the array and DataFrame.

         This method calculates the positive values in the array and the sum of the DataFrame, and logs the difference.
         Discrepancies may occur due isolated points that are not assigned to a region.
         """

        positive_values_in_array = np.count_nonzero(self.array_counts)
        sum_of_dataframe = self.df_results_total[self.target_dict.keys()].values.sum()
        logging.info(f"Positive values in array: {positive_values_in_array}")
        logging.info(f"Sum of DataFrame: {sum_of_dataframe}")
        self.counts_in_array = positive_values_in_array
        self.counts_df = sum_of_dataframe
        self.check_counts_difference = positive_values_in_array - sum_of_dataframe
        logging.info(f"Counts difference: {self.check_counts_difference}")

    def save_df(self, path):
        """
        Save the DataFrame to a CSV file.

        :param path: Path to save the CSV file.
        """

        self.df_results_total.to_csv(path, index=False)
        logging.info(f"Dataframe saved to {path}")

    def time(self):
        """
        Placeholder method to get the time for mask and overall.
        """

        # get the time for mask and overall
        pass





  # def get_tum_distance_properties(self):
    #         """
    #         Calculate the minimum distance from each object to the tumour border
    #         get_distance_to_tum method calculates the distance from each object in
    #          the first mask to the closest border of the tum mask and stores these
    #          distances in the self.properties dictionary.
    #
    #         :param self:
    #         :return:
    #         """
    #         # Invert the tum mask
    #         inverted_tum_mask = np.logical_not(self.mask_tum)
    #
    #         # Calculate the distance transform of the inverted tum mask
    #         distance_transform = distance_transform_edt(inverted_tum_mask)
    #
    #         for object_id in np.unique(self.original_labels[self.original_labels != 0]):
    #             object_mask = (self.original_labels == object_id)
    #
    #             # Find the minimum distance value within the object's area
    #             min_distance = np.min(distance_transform[object_mask])
    #
    #             # Store the minimum distance in the self.properties dictionary
    #             self.properties[object_id]['min_distance_to_tum_border'] = min_distance
    #

