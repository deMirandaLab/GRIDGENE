# https://github.com/kharchenkolab/Baysor/issues/10
# plot geojson polygons
# you read geojson as a dictionary ps,
# you need to take a GeometryCollection ps["geometries"], which is an array of Polygon dicts.
# Then you can iterate through it and take ps["geometries"][i]["coordinates"], which is an array of pairs of x,y
# coordinates for a particular polygon. If you have 3d coordinates, then on the top level you have an array of
# GeometryCollection's for different z-planes. You may see the saving code here.

import os

import matplotlib.cm
from PIL import Image, ImageDraw
from shapely.geometry import Polygon,shape, GeometryCollection
import json
from tqdm import tqdm  # Import tqdm for the progress bar
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import Reds

 # Color mapping dictionary

# color_mapping = {
#         "myeloid cells": "#debaff",
#         "T cells": "#387d36",
#         "epithelial": "#FFD580",
#         "plasma cells": "#80b6f2",
#         "endothelial cells": "#d97189",
#         "fibroblasts": "#717dc7"
#     }

#     color_mapping = {
#         "Epithelial": "#FFD580",  # light yellow
#         "Fibroblasts": '#377eb8',# blue
#         "macrophages": "#FF0000", # vivid red
#         "Granulocytes": '#984ea3', # purple
#         "Smooth_muscle/perycites": '#FFC0CB',  # pink
#         "Plasma_cells": '#ff7f00',  # orange
#         "DC_cells":"#808080",  # Hex code for grey
#         "T cells": '#4daf4a',  # green
#         "Monocytes":'#984ea3', # purple
#         "Mast cells":"#8B4513", #brown,
#         "Schwann cells":"#333333", # dark grey
#         "Endothelial":"#00008B", # dark blue
#         # "myeloid cells": '#984ea3', # purple
#     }


def draw_polygons_categorical_annotation(polygons_dict, categorical_data, color_mapping):


    x_coordinates = []
    y_coordinates = []

    for geometry in geojson_data["geometries"]:
        coordinates = geometry["coordinates"][0]  # Extract coordinates of the polygon
        for x, y in coordinates:
            x_coordinates.append(x)
            y_coordinates.append(y)

    # Calculate width and height of the image
    # Calculate minimum x and y coordinates
    min_x = min(x_coordinates)
    min_y = min(y_coordinates)
    width = int(max(x_coordinates) - min_x)
    height = int(max(y_coordinates) - min_y)
    print(width, height)



    image_categorical = Image.new("RGB", (width, height), "white")
    draw_categorical = ImageDraw.Draw(image_categorical)


    polygon_dicts = geojson_data["geometries"]
    for i in range(len(polygon_dicts)):
        polygon_coordinates = geojson_data["geometries"][i]["coordinates"]
        polygon_coordinates = [[int(x) - min_x, int(y) - min_y] for polygon in polygon_coordinates for x, y in polygon]
        polygon_coordinates = [coord for point in polygon_coordinates for coord in point]

        if len(polygon_coordinates)<3:
            continue




        cell_id = geojson_data["geometries"][i]["cell"]

        # Extract categorical value for the cell from your data
        categorical_value = categorical_data.get(cell_id, "Unknown")  # If not gets unknown

        # Get the color from the color mapping dictionary
        categorical_color = color_mapping.get(categorical_value, "white") # If the categorical value is not found in the mapping, it defaults to "white."

        # Draw the filled polygon with the categorical color
        draw_categorical.polygon(polygon_coordinates, outline="black", fill=categorical_color)

    # Save the categorical image as a TIFF file
    tiff_categorical_path = segmentation_file_path.replace("segmentation_polygons.json", "cell_label_color.tiff")

    image_categorical.save(tiff_categorical_path)

    # Split the path into parts
    parts = tiff_categorical_path.split(os.path.sep)

    # Find the index of the 'AC_ICAM_4B_S0' folder
    index = parts.index('AC_ICAM_4B_S0')

    # Insert 'images_categorical' after 'AC_ICAM_4B_S0'
    parts.insert(index + 1, 'images_categorical')

    # Join the parts back together
    new_path = os.path.join(*parts)
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(new_path)):
        os.makedirs(os.path.dirname(new_path))
    print(new_path)
    image_categorical.save(new_path)




















# def get_image_dimensions_from_polygons(polygons):
#     # Get x and y coordinates of all points in polygons
#     x_coordinates = [x for polygon in polygons for x, _ in polygon]
#     y_coordinates = [y for polygon in polygons for _, y in polygon]
#
#     # Calculate width and height of the image
#     width = max(x_coordinates) - min(x_coordinates)
#     height = max(y_coordinates) - min(y_coordinates)
#
#     return width, height

def get_image_dimensions_from_polygons(polygons):
    x_coordinates = []
    y_coordinates = []

    for polygon in polygons:
        # Check if the polygon is a list of coordinate tuples
        if isinstance(polygon, list) and all(isinstance(coord, tuple) and len(coord) == 2 for coord in polygon):
            for x, y in polygon:
                x_coordinates.append(x)
                y_coordinates.append(y)
        else:
            # If not, skip this polygon
            continue

    # Calculate width and height of the image
    width = max(x_coordinates) - min(x_coordinates)
    height = max(y_coordinates) - min(y_coordinates)

    return width, height

# Define a function to process and save TIFF files
def process_and_save_tiff(segmentation_file_path, categorical_data, continuous_data):
    # Define the dimensions of your TIFF image (adjust as needed)
    width, height = 4200, 4200

    # Create a blank image with a white background
    image_outline = Image.new("RGB", (width, height), "white")
    draw_outline = ImageDraw.Draw(image_outline)

    # Create a blank image for the filled polygons
    image_filled = Image.new("RGB", (width, height), "white")
    draw_filled = ImageDraw.Draw(image_filled)

    # open GeoJson file
    f = open(segmentation_file_path)
    geojson_data = json.load(f)
    polygon_dicts = geojson_data["geometries"]
    for i in range(len(polygon_dicts)):
        polygon_coordinates = geojson_data["geometries"][i]["coordinates"][0]
        polygon_coordinates = [coord for point in polygon_coordinates for coord in point]

        cell_id = geojson_data["geometries"][i]["cell"]
        # Iterate through the polygon coordinates and draw them on the image
        # for pol_coor in polygon_coordinates:
        draw_outline.polygon(polygon_coordinates, outline="black", fill=None)

        # Draw the filled polygon with the cell ID on the image_filled
        draw_filled.polygon(polygon_coordinates, outline=None, fill=f"#{cell_id * 1000:06X}")

    # Save the image as a TIFF file in the same folder as the segmentation file
    tiff_file_path = segmentation_file_path.replace("segmentation_polygons.json", "outline.tiff")
    image_outline.save(tiff_file_path)
    tiff_file_path = segmentation_file_path.replace("segmentation_polygons.json", "filled.tiff")
    image_outline.save(tiff_file_path)


def simple_draw(segmentation_file_path):

    # Open GeoJson file
    with open(segmentation_file_path) as f:
        geojson_data = json.load(f)

    # Initialize min and max values for x and y
    min_x, min_y = 0, 0
    max_x, max_y = 4246, 4246

    # Extract coordinates from the GeoJSON
    for feature in geojson_data["geometries"]:
        polygon_coordinates = feature["coordinates"][0]
        for point in polygon_coordinates:
            x, y = point
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # Calculate width and height
    width = int(max_x - min_x)
    height = int(max_y - min_y)

    image_categorical = Image.new("RGB", (width, height), "white")
    draw_categorical = ImageDraw.Draw(image_categorical)

    # open GeoJson file
    f = open(segmentation_file_path)
    geojson_data = json.load(f)
    polygon_dicts = geojson_data["geometries"]
    for i in range(len(polygon_dicts)):
        polygon_coordinates = geojson_data["geometries"][i]["coordinates"][0]
        polygon_coordinates = [coord for point in polygon_coordinates for coord in point]
        if len(polygon_coordinates)<3:
            continue

        # Draw the filled polygon with the categorical color
        draw_categorical.polygon(polygon_coordinates, outline="black")

    # Save the categorical image as a TIFF file
    tiff_categorical_path = segmentation_file_path.replace("segmentation_polygons.json", "cell_draw.tiff")
    image_categorical.save(tiff_categorical_path)



# def draw_categorical(segmentation_file_path, categorical_data, width, height):
#     # Color mapping dictionary
#     color_mapping = {
#         "myeloid cells": "#debaff",
#         "T cells": "#387d36",
#         "epithelial": "#FFD580",
#         "plasma cells": "#80b6f2",
#         "endothelial cells": "#d97189",
#         "fibroblasts": "#717dc7"
#     }
#
#     color_mapping = {
#         "myeloid cells": '#984ea3', # purple
#         "T cells": '#4daf4a', # green
#         "epithelial": "#FFD580", # light yellow
#         "plasma cells": '#ff7f00', # orange
#         "endothelial cells":  '#e41a1c', # red
#         "fibroblasts": '#377eb8'  # blue
#     }
#
#
#     # Define the dimensions of your TIFF image (adjust as needed)
#     # width, height = 4200, 4200
#
#     # Create a blank image with a white background
#     # image_outline = Image.new("RGB", (width, height), "white")
#     # draw_outline = ImageDraw.Draw(image_outline)
#
#     image_categorical = Image.new("RGB", (width, height), "white")
#     draw_categorical = ImageDraw.Draw(image_categorical)
#
#     # open GeoJson file
#     f = open(segmentation_file_path)
#     geojson_data = json.load(f)
#     polygon_dicts = geojson_data["geometries"]
#     for i in range(len(polygon_dicts)):
#         polygon_coordinates = geojson_data["geometries"][i]["coordinates"][0]
#         polygon_coordinates = [coord for point in polygon_coordinates for coord in point]
#         if len(polygon_coordinates)<3:
#             continue
#         cell_id = geojson_data["geometries"][i]["cell"]
#
#         # Extract categorical value for the cell from your data
#         categorical_value = categorical_data.get(cell_id, "Unknown")  # If not gets unknown
#
#         # Get the color from the color mapping dictionary
#         categorical_color = color_mapping.get(categorical_value, "white") # If the categorical value is not found in the mapping, it defaults to "white."
#
#         # Draw the filled polygon with the categorical color
#         draw_categorical.polygon(polygon_coordinates, outline="black", fill=categorical_color)
#
#         # # Store the categorical label
#         # if categorical_value not in categorical_labels:
#         #     categorical_labels[categorical_value] = f"{categorical_value}"
#
#     # Save the categorical image as a TIFF file
#     tiff_categorical_path = segmentation_file_path.replace("segmentation_polygons.json", "cell_label_color2.tiff")
#     image_categorical.save(tiff_categorical_path)

def draw_categorical(segmentation_file_path, categorical_data):
    # Color mapping dictionary
    color_mapping = {
        "myeloid cells": "#debaff",
        "T cells": "#387d36",
        "epithelial": "#FFD580",
        "plasma cells": "#80b6f2",
        "endothelial cells": "#d97189",
        "fibroblasts": "#717dc7"
    }

    color_mapping = {
        "Epithelial": "#FFD580",  # light yellow
        "Fibroblasts": '#377eb8',# blue
        "macrophages": "#FF0000", # vivid red
        "Granulocytes": '#984ea3', # purple
        "Smooth_muscle/perycites": '#FFC0CB',  # pink
        "Plasma_cells": '#ff7f00',  # orange
        "DC_cells":"#808080",  # Hex code for grey
        "T cells": '#4daf4a',  # green
        "Monocytes":'#984ea3', # purple
        "Mast cells":"#8B4513", #brown,
        "Schwann cells":"#333333", # dark grey
        "Endothelial":"#00008B", # dark blue
        # "myeloid cells": '#984ea3', # purple
    }

    # # open GeoJson file
    # f = open(segmentation_file_path)
    # geojson_data = json.load(f)
    # polygon_dicts = geojson_data["geometries"]
    #
    #
    # # Extract polygons coordinates and CellID from polygon_dicts
    # polygons_with_id = []
    # for i in range(len(polygon_dicts)):
    #     polygon_coordinates = geojson_data["geometries"][i]["coordinates"][0]
    #     polygon_coordinates = [coord for point in polygon_coordinates for coord in point]
    #     if len(polygon_coordinates) < 3:
    #         continue
    #     cell_id = geojson_data["geometries"][i]["cell"]
    #     polygons_with_id.append((polygon_coordinates, cell_id))
    #
    # # Get image dimensions
    # width, height = get_image_dimensions_from_polygons([polygon[0] for polygon in polygons_with_id])
    #
    # # Create a blank image with a white background
    # image_categorical = Image.new("RGB", (width, height), "white")
    # draw_categorical = ImageDraw.Draw(image_categorical)
    #
    # # Loop through polygons and draw them on the image
    # for polygon_coordinates, cell_id in polygons_with_id:
    #     # Extract categorical value for the cell from your data
    #     categorical_value = categorical_data.get(cell_id, "Unknown")  # If not gets unknown
    #     # Get the color from the color mapping dictionary
    #     categorical_color = color_mapping.get(categorical_value, "white") # If the categorical value is not found in the mapping, it defaults to "white."
    #
    #     # Draw the filled polygon with the categorical color
    #     draw_categorical.polygon(polygon_coordinates, outline="black", fill=categorical_color)
    #
    #     # # Store the categorical label
    #     # if categorical_value not in categorical_labels:
    #     #     categorical_labels[categorical_value] = f"{categorical_value}"
    #
    # # Save the categorical image as a TIFF file
    # tiff_categorical_path = segmentation_file_path.replace("segmentation_polygons.json", "cell_label_color.tiff")
    # image_categorical.save(tiff_categorical_path)
    # Define the dimensions of your TIFF image (adjust as needed)



    # open GeoJson file
    f = open(segmentation_file_path)
    geojson_data = json.load(f)

    x_coordinates = []
    y_coordinates = []

    for geometry in geojson_data["geometries"]:
        coordinates = geometry["coordinates"][0]  # Extract coordinates of the polygon
        for x, y in coordinates:
            x_coordinates.append(x)
            y_coordinates.append(y)

    # Calculate width and height of the image
    # Calculate minimum x and y coordinates
    min_x = min(x_coordinates)
    min_y = min(y_coordinates)
    width = int(max(x_coordinates) - min_x)
    height = int(max(y_coordinates) - min_y)
    print(width, height)



    image_categorical = Image.new("RGB", (width, height), "white")
    draw_categorical = ImageDraw.Draw(image_categorical)


    polygon_dicts = geojson_data["geometries"]
    for i in range(len(polygon_dicts)):
        polygon_coordinates = geojson_data["geometries"][i]["coordinates"]
        polygon_coordinates = [[int(x) - min_x, int(y) - min_y] for polygon in polygon_coordinates for x, y in polygon]
        polygon_coordinates = [coord for point in polygon_coordinates for coord in point]

        if len(polygon_coordinates)<3:
            continue




        cell_id = geojson_data["geometries"][i]["cell"]

        # Extract categorical value for the cell from your data
        categorical_value = categorical_data.get(cell_id, "Unknown")  # If not gets unknown

        # Get the color from the color mapping dictionary
        categorical_color = color_mapping.get(categorical_value, "white") # If the categorical value is not found in the mapping, it defaults to "white."

        # Draw the filled polygon with the categorical color
        draw_categorical.polygon(polygon_coordinates, outline="black", fill=categorical_color)

    # Save the categorical image as a TIFF file
    tiff_categorical_path = segmentation_file_path.replace("segmentation_polygons.json", "cell_label_color.tiff")

    image_categorical.save(tiff_categorical_path)

    # Split the path into parts
    parts = tiff_categorical_path.split(os.path.sep)

    # Find the index of the 'AC_ICAM_4B_S0' folder
    index = parts.index('AC_ICAM_4B_S0')

    # Insert 'images_categorical' after 'AC_ICAM_4B_S0'
    parts.insert(index + 1, 'images_categorical')

    # Join the parts back together
    new_path = os.path.join(*parts)
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(new_path)):
        os.makedirs(os.path.dirname(new_path))
    print(new_path)
    image_categorical.save(new_path)




def draw_continuous(segmentation_file_path, continuous_data):
    cmap = matplotlib.cm.get_cmap('Reds')  # You can change this to any other colormap from matplotlib

    # Define the dimensions of your TIFF image (adjust as needed)
    width, height = 4200, 4200

    image_continuous = Image.new("RGB", (width, height), "white")
    draw_continuous = ImageDraw.Draw(image_continuous)

    # open GeoJson file
    f = open(segmentation_file_path)
    geojson_data = json.load(f)
    polygon_dicts = geojson_data["geometries"]
    for i in range(len(polygon_dicts)):
        polygon_coordinates = geojson_data["geometries"][i]["coordinates"][0]
        if len(polygon_coordinates)<3:
            continue
        polygon_coordinates = [coord for point in polygon_coordinates for coord in point]

        cell_id = geojson_data["geometries"][i]["cell"]

        # Extract continuous and categorical values for the cell from your data
        continuous_value = continuous_data.get(cell_id, 0.0)

        # Get the color from the colormap based on the normalized value
        color = cmap(continuous_value)[:3]

        # Convert the color values to 8-bit integers (0-255)
        fill_color = tuple(int(val * 255) for val in color)

        draw_continuous.polygon(polygon_coordinates, outline="black", fill=fill_color)

        # Save the categorical image as a TIFF file
        tiff_continuous_path = segmentation_file_path.replace("segmentation_polygons.json", "epi_proportion.tiff")
        image_continuous.save(tiff_continuous_path)

# root_directory = '/home/martinha/PycharmProjects/phd/cosmx/Baysor-master/OUTPUT/cellpose09_v2'
# label_file = '/home/martinha/PycharmProjects/phd/cosmx/Baysor-master/OUTPUT/Meta_data_CellPose_Baysor09_label.csv'
# meta = pd.read_csv(label_file)
# # print(meta.columns)
# # print(meta.head(10))
# # Get a list of FOV folders
# fov_folders = [folder_name for folder_name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith("FOV")]
#
# # Create a progress bar
# progress_bar = tqdm(total=len(fov_folders), desc="Processing FOV Folders")
#
# # Traverse through the directories and process segmentation files FOV folders
# # for folder_name in fov_folders:
# #     print(folder_name)
# folder_name = 'FOV003'
# folder_path = os.path.join(root_directory, folder_name)
#
# segmentation_file_path = os.path.join(folder_path, "segmentation_polygons.json")
# df = meta.loc[meta['orig.ident'] == folder_name]
#
# # epi_proportion = df['proportion_epi_counts']
# # label = df['label']
# # cellID = df['CellID']
#
# categorical_data = dict(zip(df['CellID'], df['label']))
# continuous_data = dict(zip(df['CellID'], df['proportion_epi_counts']))
#
# # Check if the segmentation file exists in the folder
# if os.path.exists(segmentation_file_path):
#     # process_and_save_tiff(segmentation_file_path, categorical_data, continuous_data) # draw polygons
#     # print('draw polygons')
#     draw_categorical(segmentation_file_path, categorical_data)
#     print('draw cell labels')
#     draw_continuous(segmentation_file_path, continuous_data)
#     print('draw epi proportion')
# # Update the progress bar
# progress_bar.update(1)
#
#
# # Close the progress bar
# progress_bar.close()


#
# root_directory = '/home/martinha/PycharmProjects/phd/cosmx/Baysor-master/OUTPUT/S1'
# # Get a list of FOV folders
# fov_folders = [folder_name for folder_name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith("FOV")]
#
# # Create a progress bar
# progress_bar = tqdm(total=len(fov_folders), desc="Processing FOV Folders")
#
# # Traverse through the directories and process segmentation files FOV folders
# for folder_name in fov_folders:
#     print(folder_name)
#     folder_path = os.path.join(root_directory, folder_name)
#
#     segmentation_file_path = os.path.join(folder_path, "segmentation_polygons.json")
#     # seg_file = os.path.join(folder_path, "segmentation.csv")
#     #
#     # df = pd.read_csv(seg_file)
#     # print(df.columns)
#     # print(df[['cell','assignment_confidence', 'is_noise']])
#     simple_draw(segmentation_file_path)
#
#     # Update the progress bar
#     progress_bar.update(1)
#
#
# # Close the progress bar
# progress_bar.close()





root_directory = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/baysor/results/AC_ICAM_4B_S0'
# Get a list of FOV folders
fov_folders = [folder_name for folder_name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith("transcripts")]
print(fov_folders)
labels_file = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/baysor/results/AC_ICAM_4B_S0/single_cell_phenotype_table.xlsx'
labels = pd.read_excel(labels_file)
print(labels['phenotyping'].unique())

# Create a progress bar
progress_bar = tqdm(total=len(fov_folders), desc="Processing FOV Folders")

# Traverse through the directories and process segmentation files FOV folders
for folder_name in fov_folders:
    print(folder_name)
    folder_path = os.path.join(root_directory, folder_name)

    segmentation_file_path = os.path.join(folder_path, "segmentation_polygons.json")

    label_fov = labels.loc[labels['CellID_FOV'].str.contains(folder_name)]
    print(label_fov.shape)

    categorical_data = dict(zip(label_fov['CellID'], label_fov['phenotyping']))



    # # Check if the segmentation file exists in the folder
    if os.path.exists(segmentation_file_path):
        # process_and_save_tiff(segmentation_file_path, categorical_data, continuous_data) # draw polygons
        # print('draw polygons')
        draw_categorical(segmentation_file_path, categorical_data)
        print('draw cell labels')

    # Update the progress bar
    progress_bar.update(1)


# Close the progress bar
progress_bar.close()

#
# import matplotlib.pyplot as plt
#
# color_mapping = {
#     "Epithelial": "#FFD580",  # light yellow
#     "Fibroblasts": '#377eb8', # blue
#     "Macrophages": "#FF0000", # vivid red
#     "Granulocytes": '#984ea3', # purple
#     "Smooth_muscle/perycites": '#FFC0CB',  # pink
#     "Plasma_cells": '#ff7f00',  # orange
#     "DC_cells": "#808080",  # Hex code for grey
#     "T cells": '#4daf4a',  # green
#     "Monocytes":'#984ea3', # purple
#     "Mast cells":"#8B4513", # brown
#     "Schwann cells":"#333333", # dark grey
#     "Endothelial":"#00008B", # dark blue
# }
#
# fig, ax = plt.subplots(figsize=(8, 3))
#
# # Create legend
# handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in color_mapping]
# labels = list(color_mapping.keys())
# ax.legend(handles, labels, loc='center', ncol=3, fontsize='small')
#
# # Hide axes
# ax.axis('off')
#
# plt.show()