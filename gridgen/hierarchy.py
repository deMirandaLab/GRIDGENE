import numpy as np
import pandas as pd
from scipy.stats import mode


def create_mapping_df(child_mask_dict, reference_mask_name="Mother"):
    """
    Creates a DataFrame mapping each object in the child masks to the reference mask objects.

    Parameters:
        ref_mask (np.ndarray): The reference mask as a 2D array.
        child_mask_dict (dict): A dictionary where keys are child mask names (str) and values are dictionaries with:
            - 'mask': 2D array with the child mask's own labels.
            - 'mapped': 2D array with the corresponding reference mask labels.
        reference_mask_name (str): Identifier for the reference mask (default "Mother").

    Returns:
        pd.DataFrame: DataFrame with columns:
            ['mask_name', 'reference_mask', 'child_object_id', 'mapped_reference_label'].
    """
    rows = []

    for mask_name, masks in child_mask_dict.items():
        child_mask = masks['labelled']
        child_mask_mapped = masks['reference']

        # Get unique object IDs in child mask (exclude background 0)
        unique_ids = np.unique(child_mask)
        unique_ids = unique_ids[unique_ids != 0]

        for obj_id in unique_ids:
            # Get pixels in the mapped version corresponding to this child object.
            mapped_values = child_mask_mapped[child_mask == obj_id]

            if mapped_values.size > 0:
                # Optionally ignore background (0) in mapped_values.
                valid_values = mapped_values[mapped_values != 0]
                if valid_values.size > 0:
                    mode_result = mode(valid_values)
                    mapped_ref = np.atleast_1d(mode_result.mode)[0]
            else:
                mapped_ref = np.nan  # No pixels found

            rows.append({
                'mask_name': mask_name,
                'reference_mask': reference_mask_name,
                'child_object_id': obj_id,
                'mapped_reference_label': mapped_ref
            })

    df_mapping = pd.DataFrame(rows)
    return df_mapping

def map_object_ids(mask_dict, reference_mask_name = "Reference"):
    """
    Maps object IDs from labelled masks to corresponding IDs in the reference mask.

    Parameters:
    - mask_dict: Dictionary containing masks with their labelled and reference versions.
                 Each key is a mask name, and its value is another dictionary with keys 'labelled' and 'reference',
                 representing the labelled mask and its corresponding reference mask, respectively.
    - reference_mask_name: The key name of the reference mask in the dictionary.

    Returns:
    - DataFrame with columns: ['mask_name', 'object_id', 'reference_mask_name', 'reference_object_id']
    """
    mappings = []

    for mask_name, masks in mask_dict.items():
        labelled_mask = masks['labelled']
        reference_mask = masks['reference']

        # Ensure both masks have the same shape
        if labelled_mask.shape != reference_mask.shape:
            raise ValueError(f"Shape mismatch between labelled and reference masks for '{mask_name}'.")

        # Find unique object IDs in the labelled mask (excluding background, assumed to be 0)
        labelled_ids = np.unique(labelled_mask)
        labelled_ids = labelled_ids[labelled_ids != 0]
        print(labelled_ids)
        print(np.unique(reference_mask))
        for obj_id in labelled_ids:
            # Create a binary mask for the current object in the labelled mask
            obj_mask = labelled_mask == obj_id

            # Find the most frequent reference object ID overlapping with the current object
            overlapping_ids, counts = np.unique(reference_mask[obj_mask], return_counts=True)

            # Exclude background (0) from consideration
            if 0 in overlapping_ids:
                zero_index = np.where(overlapping_ids == 0)
                overlapping_ids = np.delete(overlapping_ids, zero_index)
                counts = np.delete(counts, zero_index)

            if overlapping_ids.size > 0:
                # Select the reference object ID with the maximum overlap
                ref_obj_id = overlapping_ids[np.argmax(counts)]
            else:
                # If no overlap is found, assign None
                ref_obj_id = None

            mappings.append({
                'mask_name': mask_name,
                'object_id': obj_id,
                'reference_mask_name': reference_mask_name,
                'reference_object_id': ref_obj_id
            })

    return pd.DataFrame(mappings)