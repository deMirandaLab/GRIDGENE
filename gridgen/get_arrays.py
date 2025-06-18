import pandas as pd
import numpy as np
import timeit

def transform_df_to_array(df: pd.DataFrame, target_dict: dict, array_shape: tuple) -> np.ndarray:
    """
    Transforms a DataFrame into a 3D numpy array based on specified target dictionary and array shape.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing 'X', 'Y', and 'target' columns.
    target_dict : dict
        A dictionary mapping target values to unique indices.
    array_shape : tuple
        The shape of the output array (max(X)+1, max(Y)+1, number of targets).

    Returns
    -------
    np.ndarray
        A 3D numpy array with dimensions specified by array_shape, where each position [x, y, target_index] is set to 1
        if there is an entry in the DataFrame with coordinates (x, y) and the corresponding target.
    """

    # Create a numpy array of zeros with the specified shape
    output_array = np.zeros(array_shape, dtype=np.int8)

    # Map the target values to their indices using the target_dict
    target_indices = df['target'].map(target_dict).values

    # Extract x and y coordinates
    x_coords = df['X'].astype(int).values
    y_coords = df['Y'].astype(int).values

    # Set the appropriate positions in the output array to 1 using advanced indexing
    output_array[x_coords, y_coords, target_indices] = 1

    return output_array





def get_subset_arrays_V1(df_total: pd.DataFrame, target_list: list, target_col: str = 'target',
               col_x: str = 'X', col_y: str = 'Y') -> tuple:
    """
    PROBABLY LESS EFFICIENT !

    Filters the DataFrame based on target_list, then creates and returns a subset DataFrame, a dictionary of target mappings,
    a 3D array representing the data, and a 2D summed array along the third axis.

    Parameters
    ----------
    df_total : pd.DataFrame
        The input DataFrame containing the data.
    target_list : list
        List of target values to filter the DataFrame.
    target_col : str, optional
        Column name in the DataFrame containing target values, by default 'target'.
    col_x : str, optional
        Column name in the DataFrame representing the X-coordinate, by default 'X'.
    col_y : str, optional
        Column name in the DataFrame representing the Y-coordinate, by default 'Y'.

    Returns
    -------
    tuple
        A tuple containing:
        - df_subset (pd.DataFrame): The filtered DataFrame.
        - target_dict_subset (dict): A dictionary mapping each target to a unique index.
        - array_subset (np.ndarray): A 3D numpy array of shape (max(X)+1, max(Y)+1, len(target_list)), filled based on the filtered DataFrame.
        - array_subset_2d (np.ndarray): A 2D numpy array obtained by summing `array_subset` along the third axis.
    """

    # Filter the DataFrame based on target_list
    df_subset = df_total.loc[df_total[target_col].isin(target_list)]

    # Create a dictionary mapping each target to a unique index
    target_dict_subset = {target: index for index, target in enumerate(df_subset[target_col].unique())}

    # Define the shape of the 3D array
    array_shape_subset = (df_total[col_x].max() + 1, df_total[col_y].max() + 1, len(target_list))

    # Create the 3D array using the provided get_array function
    array_subset = transform_df_to_array(df=df_subset, target_dict=target_dict_subset, array_shape=array_shape_subset).astype(np.int8)

    # # Sum the 3D array along the third axis to create a 2D array
    # array_subset_2d = np.sum(array_subset, axis=2)

    return df_subset, array_subset, target_dict_subset


def get_subset_arrays(df_total: pd.DataFrame, array_total: np.ndarray, target_dict_total: dict,
                      target_list: list, target_col: str = 'target') -> tuple:
    """
    Get a subset of the DataFrame, the corresponding slices from the total array, and the subset target dictionary.

    Parameters
    ----------
    df_total : pd.DataFrame
        The input DataFrame containing the data.
    array_total : np.ndarray
        The 3D array representing the entire dataset.
    target_dict_total : dict
        A dictionary mapping each target in the total dataset to its index.
    target_list : list
        List of target values to filter the DataFrame and array.
    target_col : str, optional
        Column name in the DataFrame containing target values, by default 'target'.

    Returns
    -------
    tuple
        A tuple containing:
        - df_subset (pd.DataFrame): The filtered DataFrame.
        - array_subset (np.ndarray): The subset of the array corresponding to the target_list.
        - target_dict_subset (dict): The subset dictionary mapping the filtered targets to indices.
    """

    # Filter the DataFrame based on target_list
    df_subset = df_total.loc[df_total[target_col].isin(target_list)]

    # Create a mapping from target_list to indices in the total array
    target_indices_subset = [target_dict_total.get(target, -1) for target in target_list]

    # Initialize an array of zeros with the same shape as array_total for the first two dimensions,
    # and the length of target_list for the last dimension
    array_subset = np.zeros(array_total.shape[:2] + (len(target_list),))

    # Extract the relevant slices from the array
    for i, target_index in enumerate(target_indices_subset):
        if target_index != -1:  # if the target is in target_dict_total
            array_subset[:, :, i] = array_total[:, :, target_index]

    # Create the subset target dictionary
    target_dict_subset = {target: index for index, target in enumerate(target_list)}

    return df_subset, array_subset, target_dict_subset

if __name__ == "__main__":

    def compare_functions(df_total, array_total, target_dict_total, target_list):
        setup_code = """
import pandas as pd
import numpy as np
from __main__ import get_subset_arrays, get_subset_arrays_V1, df_total, array_total, target_dict_total, target_list
"""
        stmt_V1 = "get_subset_arrays(df_total, array_total, target_dict_total, target_list)"
        stmt_V2 = "get_subset_arrays_V1(df_total, target_list)"

        time_V1 = timeit.timeit(stmt=stmt_V1, setup=setup_code, number=100)
        time_V2 = timeit.timeit(stmt=stmt_V2, setup=setup_code, number=100)

        result_V1 = get_subset_arrays(df_total, array_total, target_dict_total, target_list)
        result_V2 = get_subset_arrays_V1(df_total, target_list)

        df_equal = result_V1[0].equals(result_V2[0])
        target_dict_equal = result_V1[2] == result_V2[2]
        arrays_equal = np.array_equal(result_V1[1], result_V2[1])

        return time_V1, time_V2, df_equal, target_dict_equal, arrays_equal

    # Sample data for testing
    data = {'X': np.random.randint(0, 10, size=1000),
            'Y': np.random.randint(0, 10, size=1000),
            'target': np.random.choice(['target1', 'target2', 'target3'], size=1000)}
    df_total = pd.DataFrame(data)

    target_dict_total = {target: index for index, target in enumerate(df_total['target'].unique())}
    height, width = df_total['X'].max() + 1, df_total['Y'].max() + 1
    array_total = transform_df_to_array(df=df_total, target_dict=target_dict_total,
                                        array_shape=(height, width, len(target_dict_total))).astype(np.int8)

    target_list = ['target1', 'target2']

    time_V1, time_V2, df_equal, target_dict_equal, arrays_equal = compare_functions(df_total, array_total,
                                                                                    target_dict_total, target_list)

    print(f"Execution time for get_subset_arrays: {time_V1:.6f} seconds")
    print(f"Execution time for get_subset_arrays_V1: {time_V2:.6f} seconds")
    print(f"DataFrames are equal: {df_equal}")
    print(f"Target dictionaries are equal: {target_dict_equal}")
    print(f"Arrays are equal: {arrays_equal}")


    """
    Execution time for get_subset_arrays: 0.028953 seconds        -----  !!!!!!!
    Execution time for get_subset_arrays_V1: 0.087730 seconds
    DataFrames are equal: True
    Target dictionaries are equal: True
    Arrays are equal: True

    """
