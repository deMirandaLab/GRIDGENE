from typing import Dict, List, Tuple, Union
import numpy as np
from scipy.spatial import Voronoi,voronoi_plot_2d
from shapely.geometry import Polygon
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops

# from extract_count_info import get_info_single_mask_all_objects,get_info_single_mask_stroma_tum_classif
import pandas as pd
import logging


import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm


import gc
gc.set_threshold(0)

gc.collect()


def get_info_single_mask_stroma_tum_classif(mask, gene_data, target_dict, region_T = None, region_S = None, zone='Tum'):
    # Convert the mask to binary
    if np.any(mask > 0):
        label_image = label((mask > 0)).astype(np.int16)
        # label_image = label(mask > 0, connectivity=2)

        # Calculate connected components in the mask
        num_objects = len(np.unique(label_image))

        # Initialize an empty DataFrame to store the results
        df_objects = pd.DataFrame()

        for object_label in range(1, num_objects):
            # Extract the mask for the current object
            label_mask = (label_image == object_label).astype(np.int16)
            prop = regionprops(label_mask.astype(np.int16))[0]

            # Calculate counts for each gene over the current object
            gene_counts = np.sum(gene_data * label_mask[:, :, None], axis=(0, 1)).astype(np.int64)
            gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in target_dict.items()}

            # # confirm contours with a very small set of genes
            # genes_of_interest = ['EPCAM', 'IGHG1', 'IGHG2', 'TRGC1/TRGC2', 'TRDC']  # Replace with your actual gene names
            # filtered_genes = {gene: target_dict[gene] for gene in genes_of_interest if gene in target_dict}
            # for gene_name, count in filtered_genes.items():
            #     logging.info(f"Gene: {gene_name}, Count: {count}")
            #


            # Calculate the number of pixels in regions A and B for the current object
            pixels_in_T = np.sum(label_mask * region_T)
            pixels_in_S = np.sum(label_mask * region_S)

            # Extract the objects in A and B that intersect with the current object
            objects_in_T = np.unique(label_mask * region_T)
            objects_in_S = np.unique(label_mask * region_S)

            df_object = pd.DataFrame({
                'Zone': [zone],
                'MaskID': [object_label],
                'Area': [prop.area],
                'Perimeter': [prop.perimeter],
                'Vertices': [len(prop.coords)],
                'Centroid': [prop.centroid],
                'MinX': [prop.bbox[1]],
                'MinY': [prop.bbox[0]],
                'MaxX': [prop.bbox[3]],
                'MaxY': [prop.bbox[2]],
                'BoundingBox': [prop.bbox],
                'PixelsInTumour': [pixels_in_T],
                'PixelsInStroma': [pixels_in_S],
                'ObjectsTumour': [objects_in_T],
                'ObjectsStroma': [objects_in_S],
                **gene_count_dict
            })

            df_objects = pd.concat([df_objects, df_object], ignore_index=True)

    else:
        # If no object is found in the mask, create a DataFrame with None values
        gene_count_dict = {gene_name: None for gene_name, target_index in target_dict.items()}

        df_objects = pd.DataFrame({
            'Zone': [zone],
            'MaskID': [None],
            'Area': [None],
            'MinX': [None],
            'MinY': [None],
            'MaxX': [None],
            'MaxY': [None],
            'Perimeter': [None],
            'Vertices': [None],
            'Centroid': [None],
            'BoundingBox': [None],
            **gene_count_dict
        })

    # check if it is the same as using the all mask
    # label_mask = (mask > 0).astype(np.int16)
    # Calculate counts for each gene over the entire mask
    # gene_counts = np.sum(gene_data * label_mask[:, :, None], axis=(0, 1)).astype(np.int64)
    # gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in target_dict.items()}
    # total_area = np.sum(label_mask)

    # Check if the sums are equal for each gene
    # for gene_name, target_index in target_dict.items():
    #     total_sum = gene_counts[target_index]
    #     per_object_sum = np.sum(df_objects[gene_name])
    #     print(gene_name)
    #     print(total_sum)
    #     print(per_object_sum)
        # assert total_sum == per_object_sum


    return df_objects

def get_info_single_mask(mask, gene_data, target_dict, zone='Tum'):
    # Convert the mask to binary
    if np.any(mask > 0):
        label_image = label((mask > 0)).astype(np.int16)
        # label_image = label(mask > 0, connectivity=2)

        # Calculate connected components in the mask
        num_objects = len(np.unique(label_image))

        # Initialize an empty DataFrame to store the results
        df_objects = pd.DataFrame()

        for object_label in tqdm(range(1, num_objects), desc=f'Processing Objects{zone}'):
            # Extract the mask for the current object
            label_mask = (label_image == object_label).astype(np.int16)
            prop = regionprops(label_mask.astype(np.int16))[0]

            # Calculate counts for each gene over the current object
            gene_counts = np.sum(gene_data * label_mask[:, :, None], axis=(0, 1)).astype(np.int64)
            gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in target_dict.items()}

            # Assuming gene_counts is a list of counts corresponding to each gene
            genes_of_interest = ['EPCAM', 'IGHG1', 'IGHG2']  # Replace with your actual gene names

            for gene_name, count in zip(target_dict.keys(), gene_counts):
                if gene_name in genes_of_interest:
                    print(f"Gene: {gene_name}, Count: {count}")

            df_object = pd.DataFrame({
                'Zone': [zone],
                'MaskID': [object_label],
                'Area': [prop.area],
                'Perimeter': [prop.perimeter],
                'Vertices': [len(prop.coords)],
                'Centroid': [prop.centroid],
                'MinX': [prop.bbox[1]],
                'MinY': [prop.bbox[0]],
                'MaxX': [prop.bbox[3]],
                'MaxY': [prop.bbox[2]],
                'BoundingBox': [prop.bbox],
                **gene_count_dict
            })

            df_objects = pd.concat([df_objects, df_object], ignore_index=True)

    else:
        # If no object is found in the mask, create a DataFrame with None values
        gene_count_dict = {gene_name: None for gene_name, target_index in target_dict.items()}

        df_objects = pd.DataFrame({
            'Zone': [zone],
            'MaskID': [None],
            'Area': [None],
            'MinX': [None],
            'MinY': [None],
            'MaxX': [None],
            'MaxY': [None],
            'Perimeter': [None],
            'Vertices': [None],
            'Centroid': [None],
            'BoundingBox': [None],
            **gene_count_dict
        })

    # check if it is the same as using the all mask
    label_mask = (mask > 0).astype(np.int16)
    # Calculate counts for each gene over the entire mask
    gene_counts = np.sum(gene_data * label_mask[:, :, None], axis=(0, 1)).astype(np.int64)
    gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in target_dict.items()}
    total_area = np.sum(label_mask)

    # Check if the sums are equal for each gene
    for gene_name, target_index in target_dict.items():
        total_sum = gene_counts[target_index]
        per_object_sum = np.sum(df_objects[gene_name])
        print(gene_name)
        print(total_sum)
        print(per_object_sum)
        # assert total_sum == per_object_sum


    return df_objects

def get_info_single_mask_all_objects(mask, gene_data, target_dict, zone='Tum'):
    # Convert the mask to binary
    # mask_fli = np.flipud(mask)
    if np.any(mask > 0):
        label_mask = (mask > 0).astype(np.int16)
        # Calculate counts for each gene over the entire mask
        gene_counts = np.sum(gene_data * label_mask[:, :, None], axis=(0, 1)).astype(np.int64)
        gene_count_dict = {gene_name: gene_counts[target_index] for gene_name, target_index in target_dict.items()}

        # Assuming gene_counts is a list of counts corresponding to each gene
        genes_of_interest = ['EPCAM', 'IGHG1', 'IGHG2', 'VIM']  # Replace with your actual gene names

        for gene_name, count in zip(target_dict.keys(), gene_counts):
            if gene_name in genes_of_interest:
                print(f"Gene: {gene_name}, Count: {count}")
        # Calculate total area covered by the mask
        total_area = np.sum(label_mask)

        # Calculate bounding box
        prop = regionprops(label_mask.astype(np.int16))[0]
        min_x, min_y, max_x, max_y = prop.bbox

        df_object = pd.DataFrame({
            'Zone': [zone],
            'MaskID': [1],  # Only one mask, so MaskID is 1
            'Area': [total_area],
            'MinX': [min_x],
            'MinY': [min_y],
            'MaxX': [max_x],
            'MaxY': [max_y],
            **gene_count_dict
        })
    else:
        gene_count_dict = {gene_name: None for gene_name, target_index in target_dict.items()}

        df_object = pd.DataFrame({
            'Zone': [zone],
            'MaskID': [1],  # Only one mask, so MaskID is 1
            'Area': None,
            'MinX': None,
            'MinY': None,
            'MaxX': None,
            'MaxY': None,
            **gene_count_dict
        })
    return df_object

def draw_bounding_boxes(image, labels):
    for label_value in range(1, np.max(labels) + 1):
        mask = (labels == label_value).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        color = np.random.randint(0, 256, 3).tolist()  # Random color for each object
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, str(label_value), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# TODO typing
# receives contours from multiple interest regions

class MultiClassObjectAnalysis():
    def __init__(self, multiple_contours: Dict, mask_T:np, mask_S:np,height:int, width:int, save_path:str):
        self.multiple_contours = multiple_contours
        self.masks = None
        self.height = height
        self.width = width
        self.vor = None
        self.list_of_polygons = None
        self.class_labels = None
        self.all_centroids = None
        self.mask_T = mask_T
        self.mask_S = mask_S
        self.save_path = save_path

        for class_label, contours in self.multiple_contours.items():
            for i, contour in enumerate(contours):
                # Reverse the order of points in each contour
                self.multiple_contours[class_label][i] = contour[::-1]


    def get_voronoi(self):
        return self.vor

    def get_masks(self):
        return self.masks

    def get_polygons_from_contours(self,contours):
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

    # def annotate_voronoi(self):
    #     """
    #
    #     :param self:
    #     :return:
    #     """
    #     # annotate voronoi accordingly to the type
    #
    #     # Step 1: Determine which Voronoi regions belong to each contour
    #     contour_voronoi_regions = {contour_name: [] for contour_name in self.multiple_contours.keys()}
    #     for region_idx, region_vertices in enumerate(self.vor.regions):
    #         if region_vertices and -1 not in region_vertices:
    #             region_points = [self.vor.vertices[i] for i in region_vertices]
    #             region_polygon = Polygon(region_points)
    #             for contour_name, contour_points in self.multiple_contours.items():
    #                 if any(all(point in contour_points for point in region_points)):
    #                     contour_voronoi_regions[contour_name].append(region_polygon)
    #                     break
    #
    #     # Step 5: Create numpy array mask with regions annotated by original name
    #     mask_array = np.zeros((image_height, image_width), dtype=np.uint8)
    #
    #     for contour_name, voronoi_regions in contour_voronoi_regions.items():
    #         color = np.random.randint(0, 256, size=3, dtype=np.uint8)  # Random color for each contour
    #         for region in voronoi_regions:
    #             region_points = region.exterior.coords[:-1]  # Exclude duplicate last point
    #             region_points = np.array(region_points, dtype=np.int32)
    #             cv2.fillPoly(mask_array, [region_points], color)
    #
    #     self.annotated_voronoi_dict = contour_voronoi_regions
    #     self.annotated_voronoi_mask = mask_array


    def plot_voronoi(self, show = False, save_path = ''):
        """

        :param self:
        :param show:
        :param save_path:
        :return:
        """
        # Plot the Voronoi diagram
        fig, ax = plt.subplots()
        voronoi_plot_2d(self.vor, ax=ax)

        # Plot the polygons
        for polygon in self.list_of_polygons:
            x, y = polygon.exterior.xy
            ax.plot(x, y, color='red')
        # plt.xlim(0, height)
        # plt.ylim(0, width)
        # if show:
        #     plt.show()
        if save_path:
            plt.savefig(os.path.join(save_path, 'voronoi.png'))



    def plot_voronoi_annotated(self, show=False, save_path = None):
        centroids = []
        class_labels = []
        class_color_mapping = {}  # Mapping to store colors for each class

        for class_label, contours in self.multiple_contours.items():
            for contour in contours:
                # polygon = Polygon(contour)
                # centroids.append(polygon.centroid)
                class_labels.append(class_label)

            # Generate a unique color for this class
            if class_label not in class_color_mapping:
                class_color_mapping[class_label] = np.random.rand(3, )  # Random RGB color

        fig, ax = plt.subplots()
        voronoi_plot_2d(self.vor, show_vertices=False, ax=ax)

        # Plot the polygons
        for i, polygon in enumerate(self.list_of_polygons):
            x, y = polygon.exterior.xy
            class_label = class_labels[i]
            color = class_color_mapping[class_label]
            ax.plot(x, y, color=color)

        if show:
            plt.show()
        if save_path:
            plt.savefig(os.path.join(save_path, 'voronoi_annotated.png'))


    def plot_voronoi_annotate_regions(self, show=False, save_path=None):
        class_labels = self.class_labels
        vor = self.vor
        print(len(class_labels))
        print(len(vor.points))
        # Get unique class labels and assign a distinct color to each label
        unique_labels = list(set(class_labels))
        num_labels = len(unique_labels)
        label_colors = {label: np.random.rand(3) for label in unique_labels}  # Random RGB color for each label

        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, show_vertices=False, ax=ax)

        # Plot the Voronoi regions with colors according to the class labels
        for i, region in enumerate(vor.regions):
            if  -1 not in region and len(region) > 0: #
                polygon = [vor.vertices[j] for j in region]
                polygon = np.array(polygon)
                region_label = class_labels[vor.point_region[i]]
                color = label_colors[region_label]
                ax.fill(*zip(*polygon), color=color, alpha=0.5)

        # Plot the contours with colors according to the class labels
        for label in unique_labels:
            points = [vor.points[i] for i in range(len(vor.points)) if class_labels[i] == label]
            if points:
                ax.plot(*zip(*points), 'o', color=label_colors[label], label=label)

        ax.legend()

        if show:
            plt.show()
        if save_path:
            plt.savefig(os.path.join(save_path, 'voronoi_annotated.png'))



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

    def plot_voronoi_masks(self, colors=None, show=True, save_path=None):
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots()
        category_names = self.multiple_contours.keys()
        print(category_names)
        # Set up colormap
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(category_names)))
        cmap = ListedColormap(colors)

        for i, category_name in enumerate(category_names):
            voronoi_mask = self.get_voronoi_mask(category_name)
            ax.imshow(voronoi_mask, cmap=cmap, alpha=0.5)

        ax.axis('off')

        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)





    def expand_mask(self, mask, expansion_distance):
        kernel = np.ones((expansion_distance, expansion_distance), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        expanded_mask = cv2.subtract(expanded_mask, mask)

        return expanded_mask

    def generate_expanded_masks(self,expansion_distances=(100, 200)):
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
            # voronoi_mask = self.get_voronoi_mask(category_name)
            previous_expansion_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for expansion_distance in expansion_distances:
                current_expansion_mask = self.expand_mask(mask.copy(), expansion_distance)
                # Ensure the current expansion doesn't overlap with previous ones
                current_expansion_mask = cv2.bitwise_and(current_expansion_mask,
                                                         cv2.bitwise_not(previous_expansion_mask))
                # Apply Voronoi mask
                # current_expansion_mask = cv2.bitwise_and(current_expansion_mask, voronoi_mask)
                expanded_masks[f'{category_name}_expansion_{expansion_distance}'] = current_expansion_mask
                # Update previous expansion mask
                previous_expansion_mask = cv2.bitwise_or(previous_expansion_mask, current_expansion_mask)

        # Step 3: Update the final dictionary
        masks.update(expanded_masks)
        return masks

    def generate_expanded_masks_limited_by_voronoi(self,expansion_distances=(100, 200)):
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
        return masks

    # todo fix so accepts other masknames
    def plot_contours_expansion_with_voronoi_edges(self, path_save = None, show = True):
        masks = self.masks
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Iterate over masks
        for mask_name, mask in masks.items():
            if 'cd8' in mask_name:
                cmap = 'Blues'  # Use blue colormap for cd8 masks
            elif 'gd' in mask_name:
                cmap = 'Reds'  # Use red colormap for gd masks
            else:
                cmap = 'gray'  # Default colormap for other masks

            ax.imshow(mask, cmap=cmap, alpha=0.5, vmin=0, vmax=1, extent=[0, mask.shape[1], 0, mask.shape[0]], origin='lower')

        # Plot Voronoi edges
        voronoi_plot_2d(self.vor, ax=ax, show_vertices=False, line_colors='black')

        # Set background color
        ax.set_facecolor('black')

        # Set title
        ax.set_title('Masks with Voronoi Edges')

        # Hide axis
        ax.axis('off')

        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='cd8'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='gd')
        ]

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        if path_save is not None:
            save_path = os.path.join(path_save, f'MCA_contours_expansion_with_voronoi_edges.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show: plt.show()
        plt.clf()


    def plot_contours_expansion_tum_stroma_mask(self, path_save = None, show = True):
        # import matplotlib.colors as mcolors
        from matplotlib.colors import ListedColormap

        masks = self.masks
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))


        masks.update({'stroma': self.mask_S, 'tumour': self.mask_T})

        # mask_colors = {'cd8':'navy', 'cd8_expansion_100':'mediumblue', 'cd8_expansion_200':'blue',
        #           'gd': 'darkred', 'gd_expansion_100':'red', 'gd_expansion_200': 'tomato'}
        mask_colors = {'stroma': (255, 255, 0), # 'yellow',
                       'tumour':(85, 107, 47), #'dark olive green'
                        'cd8': (0, 0, 128),  # navy
                       'cd8_expansion_100': (0, 0, 205),  # mediumblue
                       'cd8_expansion_200': (0, 0, 255),  # blue
                       'gd': (139, 0, 0),  # darkred
                       'gd_expansion_100': (255, 0, 0),  # red
                       'gd_expansion_200': (255, 99, 71), # tomato,

                       }

        # Define the order of masks
        mask_order = ['stroma', 'tumour',
                      'cd8', 'cd8_expansion_100', 'cd8_expansion_200',
                      'gd', 'gd_expansion_100', 'gd_expansion_200']

        # Extract masks and colors in the desired order
        ordered_masks = [masks[name] for name in mask_order]
        ordered_colors = [mask_colors[name] for name in mask_order]

        # Create combined masks
        combined_masks = np.zeros_like(ordered_masks[0])
        for i, mask in enumerate(ordered_masks):
            combined_masks[mask > 0] = i + 1  # Assign a unique color index to each mask, starting from 1

        # Plot combined masks with colors
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(combined_masks, cmap=ListedColormap(ordered_colors), origin='lower', alpha=0.8)



        # # Create legend handles for each mask
        # legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
        #
        # # Add legend
        # ax.legend(legend_handles, mask_names, loc='upper left')

        plt.show()


        # mask_s = np.repeat(self.mask_S[:, :, np.newaxis], 3, axis=0)
        # mask_t = np.repeat(self.mask_T[:, :, np.newaxis], 3, axis=0)
        #
        # mask_s_color = np.where(mask_s > 0, np.array(color_S), 0)
        # mask_t_color = np.where(mask_t > 0, np.array(color_T), 0)
        #
        # ax.imshow(mask_s_color * np.array(color_S), alpha=0.5,
        #           extent=[0, self.mask_S.shape[1], 0, self.mask_S.shape[0]],origin='lower')
        #
        # ax.imshow(mask_t_color * np.array(color_T), alpha=0.5,
        #           extent=[0, self.mask_T.shape[1], 0, self.mask_T.shape[0]],origin='lower')
        #
        # # # Overlay solid color images for each mask
        # for mask_name, mask in masks.items():
        #     # Get color for the mask from the mask_colors dictionary
        #     color = mask_colors.get(mask_name, (128, 128, 128))  # Default to gray if color not found
        #     print(mask_name)
        #     print(color)
        #
        #     mask_ = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        #     # Plot the mask with the updated colors
        #     ax.imshow(mask_, alpha=0.8, extent=[0, mask.shape[1], 0, mask.shape[0]], origin='lower')
        #


        # Set background color
        ax.set_facecolor('black')
        # Set title
        ax.set_title('Masks with tumour stroma')
        # Hide axis
        ax.axis('off')

        # Create legend
        # legend_elements = [
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='cd8'),
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='gd'),
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Tumor'),
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Stroma'),
        # ]

        # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=value, markersize=10, label=key)
        #                    for key, value in mask_colors.items()]
        # legend_elements.append(
        #     [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Stroma'),
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Tumor')])
        # ax.legend(handles=legend_elements)
        #

        # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        if path_save is not None:
            save_path = os.path.join(path_save, f'MCA_contours_expansion_tumour_stroma.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show: plt.show()
        plt.clf()


    def extract_cell_info_from_masks_simple(self):
        pass


    def extract_cell_info_from_masks_with_tum_stroma_annotation(self, mask_S, mask_T, array_total, target_dict_total, df_total):

        label_full_tum = label((mask_T > 0)).astype(np.int16)
        label_full_stroma = label((mask_S > 0)).astype(np.int16)

        # Create color images with labeled objects and bounding boxes
        color_A = cv2.cvtColor(mask_T, cv2.COLOR_GRAY2BGR)
        color_B = cv2.cvtColor(mask_S, cv2.COLOR_GRAY2BGR)

        draw_bounding_boxes(color_A, label_full_tum)
        draw_bounding_boxes(color_B, label_full_stroma)

        # # Save color images with labeled objects and bounding boxes
        cv2.imwrite(os.path.join(self.save_path, 'MCA_T_labeled_n_tum_for_cell.png'), color_A)
        cv2.imwrite(os.path.join(self.save_path, 'MCA_S_labeled_n_stroma_for_cell.png'), color_B)

        # Initialize an empty list to store individual DataFrames
        dfs = []

        # Iterate over the masks dictionary
        for mask_name, mask in self.masks.items():
            df_result = get_info_single_mask_stroma_tum_classif(mask, array_total, target_dict_total,
                                                                region_T=label_full_tum,
                                                                region_S=label_full_stroma,
                                                                zone=mask_name)
            dfs.append(df_result)

        # Concatenate all DataFrames in the list
        df_results_cells = pd.concat(dfs, ignore_index=True)

        mask_cells = sum(mask for mask in self.masks.values())
        mask_T_cells = cv2.subtract(mask_T, mask_cells)
        mask_remaining_S = cv2.subtract(mask_S, mask_cells)

        df_results_tum = get_info_single_mask_all_objects(mask_T_cells, array_total, target_dict_total, zone='Tum')

        df_results_remaining = get_info_single_mask_all_objects(mask_remaining_S, array_total, target_dict_total,
                                                                zone='Stroma_remaining')


        # get the negative masks
        combined_mask_cell = mask_cells + mask_T_cells + mask_remaining_S
        negative_combined_mask_cell = 1 - np.clip(combined_mask_cell, 0, 1)

        df_results_lost_transcripts = get_info_single_mask_all_objects(negative_combined_mask_cell,
                                                                       array_total, target_dict_total, zone='Lost')


        df_results_total = pd.concat([df_results_cells, df_results_remaining,
                                      df_results_tum, df_results_lost_transcripts],
                                     ignore_index=True)

        df_results_total = df_results_total.fillna(0)

        positive_values_in_array = np.count_nonzero(array_total)
        sum_of_dataframe = df_results_total[df_total['target'].unique()].values.sum()
        logging.info(f"Positive values in array: {positive_values_in_array}")
        logging.info(f"Sum of DataFrame: {sum_of_dataframe}")
        df_results_total.to_csv(os.path.join(self.save_path, 'MCA_multi_object_analysis.csv'))
