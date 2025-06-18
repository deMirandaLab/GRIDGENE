"""
File for get the tum stroma mask using bins of image
and the SOM clustering
"""
import logging
import cv2
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from minisom import MiniSom
from itertools import product
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any, Union
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from gridgen.logger import get_logger


# TODO make bins overlapping?


class GetBins:
    """
    Bin spatial transcriptomics data into grid cells and create AnnData objects.
    """

    def __init__(self, bin_size: int, unique_targets: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize GetBins.

        Parameters
        ----------
        bin_size : int
            Size of bins in pixels.
        unique_targets : List[str]
            List of target genes.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.bin_size = bin_size
        self.unique_targets = unique_targets
        self.adata = None
        self.eval_som_statistical_df = None
        self.logger = logger or get_logger(f'{__name__}.{contour_name or "GetContour"}')
        self.logger.info("Initialized GetContour")

    def get_bin_df(self, df: pd.DataFrame, df_name: str) -> ad.AnnData:
        """
        Convert a DataFrame of cells with spatial coordinates and target labels into a binned AnnData object.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['X', 'Y', 'target'] representing cell positions and target labels.
        df_name : str
            Identifier for the dataset.

        Returns
        -------
        ad.AnnData
            AnnData object with spatial bins and counts per target.
        """
        # Calculate grid positions
        df['x_grid'] = df['X'] // self.bin_size
        df['y_grid'] = df['Y'] // self.bin_size

        # Count occurrences of each target in each grid cell
        quadrant_counts = df.groupby(['x_grid', 'y_grid', 'target']).size().unstack(fill_value=0)

        # Reindex to ensure all targets are included, even if they have 0 counts
        quadrant_counts = quadrant_counts.reindex(columns=self.unique_targets, fill_value=0)

        # Convert the counts DataFrame to a numpy array for AnnData
        quadrant_counts_array = quadrant_counts.values

        # Create an AnnData object
        adata = sc.AnnData(X=quadrant_counts_array)

        # Set observation and variable (gene) names
        adata.obs_names = [f"grid_{x}_{y}" for x, y in quadrant_counts.index]
        adata.var_names = quadrant_counts.columns
        adata.obs['name'] = df_name

        # Calculate centroid coordinates
        adata.obs['x_centroid'] = df.groupby(['x_grid', 'y_grid'])['X'].mean().values * self.bin_size
        adata.obs['y_centroid'] = df.groupby(['x_grid', 'y_grid'])['Y'].mean().values * self.bin_size

        # Store grid positions in the observation metadata
        adata.obs['x_grid'] = [x for x, y in quadrant_counts.index]
        adata.obs['y_grid'] = [y for x, y in quadrant_counts.index]

        # Store spatial information
        adata.obs['x_coord'] = df.groupby(['x_grid', 'y_grid'])['X'].first().values * self.bin_size
        adata.obs['y_coord'] = df.groupby(['x_grid', 'y_grid'])['Y'].first().values * self.bin_size
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

        self.adata = adata
        return adata

    def get_bin_cohort(self, df_list: List[pd.DataFrame], df_name_list: List[str], cohort_name: str) -> None:
        """
        Process multiple datasets into binned AnnData objects and concatenate them into a cohort.

        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of DataFrames to process.
        df_name_list : List[str]
            List of dataset names corresponding to each DataFrame.
        cohort_name : str
            Name of the cohort to assign to all data.
        """
        start_time = time.time()
        adata_list = []
        for df, df_name in zip(df_list, df_name_list):
            adata = self.get_bin_df(df, df_name)
            adata.obs['cohort'] = cohort_name
            adata_list.append(adata)
        combined_adata = ad.concat(adata_list, join='outer')
        self.adata = combined_adata
        self.logger.info(f'Time to get bins for {len(df_list)} dataframes: {time.time() - start_time:.2f} seconds')
        self.logger.info(f'Number of bins: {len(combined_adata)}')
        self.logger.info(f'Number of genes: {len(combined_adata.var_names)}')

    def preprocess_bin(self, min_counts: int = 10, adata: Optional[ad.AnnData] = None) -> None:
        """
        Filter and normalize the binned AnnData.

        Parameters
        ----------
        min_counts : int, optional
            Minimum total counts per bin to retain it, by default 10
        adata : Optional[ad.AnnData], optional
            AnnData object to preprocess (defaults to internal one), by default None
        """
        if adata is None:
            adata = self.adata
        sc.pp.filter_cells(adata, min_counts=min_counts)
        adata.layers["counts"] = adata.X.copy()
        adata.obs['total_counts'] = adata.X.sum(axis=1)
        adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)

        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        self.adata = adata


class GetContour:
    """
    Perform SOM clustering on spatial bins and evaluate clusters.
    """

    def __init__(self, adata: ad.AnnData, logger: Optional[logging.Logger] = None):
        """
        Initialize GetContour.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing binned spatial transcriptomics data.
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        """
        self.adata = adata
        self.logger = logger
        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def run_som(
        self,
        som_shape: Tuple[int, int] = (2, 1),
        n_iter: int = 5000,
        sigma: float = 0.5,
        learning_rate: float = 0.5,
        random_state: int = 42
    ) -> None:
        """
        Apply SOM clustering on the AnnData object.

        Parameters
        ----------
        som_shape : Tuple[int, int], optional
            Shape of the SOM grid (rows, columns), by default (2, 1)
        n_iter : int, optional
            Number of iterations for SOM training, by default 5000
        sigma : float, optional
            Width of the Gaussian neighborhood function, by default 0.5
        learning_rate : float, optional
            Learning rate for SOM training, by default 0.5
        random_state : int, optional
            Random seed for reproducibility, by default 42
        """
        start = time.time()
        som = MiniSom(som_shape[0], som_shape[1], self.adata.shape[1],
                      sigma=sigma, learning_rate=learning_rate, random_seed=random_state)
        som.train_random(self.adata.X,n_iter)

        # Step 3: Assign Clusters
        clusters = np.zeros(len(self.adata), dtype=int)
        clusters = list(clusters)

        possible_tuples = list(product(range(som_shape[0]), range(som_shape[1])))
        table_values = list(range(len(possible_tuples)))
        # Create a dictionary to map tuples to values
        table_dict = {t: v for t, v in zip(possible_tuples, table_values)}
        # print(table_dict)

        for i, q in enumerate(self.adata.X):
            # print(som.winner(q))   #(x,y)
            x, y = som.winner(q)
            clusters[i] = int(table_dict.get((x, y)))

        self.adata.obs['cluster_som'] = pd.Categorical(clusters)
        self.logger.info(f'Time to run som on {len(self.adata.X)} bins: {time.time() - start:.2f}')
        self.logger.info(f'Number of clusters: {len(set(clusters))}')
        self.logger.info(f'number of bins in each cluster: {self.adata.obs["cluster_som"].value_counts()}')

    def eval_som_statistical(self, top_n: int = 20) -> None:
        """
        Compute and log top ranked features per SOM cluster.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to retrieve for each cluster, by default 20
        """
        sc.tl.rank_genes_groups(self.adata, "cluster_som", method="t-test")
        stats = []
        groups = self.adata.uns['rank_genes_groups']['names'].dtype.names
        for group in groups:
            df = sc.get.rank_genes_groups_df(self.adata, group)
            df['group'] = group
            df_sorted = df.sort_values(by='scores', ascending=False).head(top_n)
            self.logger.info(f"n top genes for group {group}")
            self.logger.info("\n" + df_sorted.to_string())
            stats.append(df_sorted)
        self.eval_som_statistical_df = pd.concat(stats, ignore_index=True)

    def create_cluster_image(self, adata: ad.AnnData, grid_size: int) -> np.ndarray:
        """
        Reconstruct an image from cluster annotations in the AnnData object.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing clustering results and grid positions.
        grid_size : int
            Size of each grid cell in pixels.

        Returns
        -------
        np.ndarray
            2D array with cluster IDs as pixel values.
        """
        # Initialize an empty image
        max_x_grid = adata.obs['x_grid'].max()
        max_y_grid = adata.obs['y_grid'].max()
        image_shape = (int((max_x_grid + 1) * grid_size), int((max_y_grid + 1) * grid_size))

        reconstructed_image = np.zeros(image_shape)

        # Iterate over the observations in the AnnData object
        for _, row in adata.obs.iterrows():
            # Retrieve the SOM cluster and grid coordinates
            cluster = row['cluster_som'] +1
            x_start = int(row['x_grid'] * grid_size)
            y_start = int(row['y_grid'] * grid_size)

            # Set all pixels in the corresponding grid to the SOM cluster value
            reconstructed_image[x_start:x_start + grid_size, y_start:y_start + grid_size] = cluster

        return reconstructed_image

    def plot_som(
        self,
        som_image: np.ndarray,
        cmap: Optional[Any] = None,
        path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (10, 10),
        ax: Optional[plt.Axes] = None,
        legend_labels: Optional[Dict[int, str]] = None
    ) -> plt.Axes:
        """
        Visualize the SOM cluster map.

        Parameters
        ----------
        som_image : np.ndarray
            2D array representing the SOM clusters.
        cmap : Optional[Any], optional
            Colormap to use for visualization, by default None (uses 'tab10')
        path : Optional[str], optional
            Optional path to save the plot image, by default None
        show : bool, optional
            Whether to display the plot, by default False
        figsize : Tuple[int, int], optional
            Size of the figure, by default (10, 10)
        ax : Optional[plt.Axes], optional
            Matplotlib Axes to plot on, by default None (creates new figure)
        legend_labels : Optional[Dict[int, str]], optional
            Dictionary mapping cluster indices to labels for legend, by default None

        Returns
        -------
        plt.Axes
            The matplotlib Axes object containing the plot.
        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        ax.imshow(som_image, cmap=cmap, interpolation='none', origin='lower')
        ax.set_title('SOM clustering')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        if legend_labels:
            # Create custom legend handles
            handles = [mpatches.Patch(color=cmap(idx / max(legend_labels.keys())), label=label)
                       for idx, label in legend_labels.items()]
            # ax.legend(handles=handles, loc='upper right', title="Clusters")
            ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Clusters")

        if path is not None:
            save_path = os.path.join(path, 'SOM_clustering.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        if show:
            plt.show()

        return ax