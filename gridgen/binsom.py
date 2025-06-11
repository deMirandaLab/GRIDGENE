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

#
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# todo make bins overlapping?
# make it like the ncvs?


class GetBins():
    def __init__(self, bin_size, unique_targets, logger = None):
        self.bin_size = bin_size
        self.unique_targets = unique_targets
        self.adata = None
        self.eval_som_statistical_df = None
        self.logger = logger
        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


    def get_bin_df(self, df, df_name):
        """
        df: dataframe with the columns x, y and target
        df_name: name of the dataframe
        :return:

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

    def get_bin_cohort(self, df_list, df_name_list, cohort_name):
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


    def preprocess_bin(self, min_counts = 10, adata=None):
        if adata is None:
            adata = self.adata
        sc.pp.filter_cells(adata, min_counts=min_counts)
        adata.layers["counts"] = adata.X.copy()
        adata.obs['total_counts'] = adata.X.sum(axis=1)
        adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)


        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)



        self.adata = adata



class GetContour():
    def __init__(self, adata, logger=None):
        self.adata = adata
        self.logger = logger
        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


    def run_som(self, som_shape = (2,1), n_iter = 5000, sigma=.5, learning_rate=.5, random_state = 42):
        """ Run the SOM clustering on the adata object
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


    def eval_som_statistical(self, top_n=20):
        sc.tl.rank_genes_groups(self.adata, "cluster_som", method="t-test")
        # df = sc.get.rank_genes_groups_df(self.adata, group=['0', '1'])
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

    def create_cluster_image(self, adata, grid_size):
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

    def get_som_2d_image(self, bin_size = 10):
        # todo not working

        unique_cases = self.adata.obs['name'].unique()
        som_images = {}
        for case in unique_cases:
            adata_case = self.adata[self.adata.obs['name'] == case, :]
            som_image = self.create_cluster_image(adata_case, grid_size=bin_size)
            som_images[case] = som_image.T
        self.som_images = som_images
        return som_images

    def plot_som(self, som_image, cmap = None, path=None, show=False, figsize=(10, 10), ax=None, legend_labels=None):
        """
        Plot the SOM clustering
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

    # def get_contour_som(self):

    #     self.local_sum_image = None
    #     self.contours = None
    #     self.contour_name = contour_name
    #     self.total_valid_contours = 0
    #     self.contours_filtered_area = 0
    #
    #
    #
    # def get_conv_sum(self, kernel_size, kernel_shape='square'):
    #     kernel = np.ones((kernel_size, kernel_size))
    #     if kernel_shape == 'circle':
    #         diameter = kernel_size
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    #     self.local_sum_image = cv2.filter2D(np.sum(self.array, axis=2), -1, kernel)
    #
    #
    # def check_contours(self):
    #     # exclude contours with less than 3 points
    #     self.contours = [np.squeeze(contour).astype(np.int32) for contour in self.contours if len(contour) > 2]
    #     # add the last coordinate to the list if the contour is not closed
    #     self.contours = [np.vstack([contour, contour[0]]) if contour[0].tolist() != contour[-1].tolist() else contour for
    #                  contour in self.contours]
    #     # transform to np int 32 to compatibility with opencv
    #     self.contours = [np.array(contour, dtype=np.int32) for contour in self.contours]
    #
    # def filter_contours_area(self,min_area_threshold):
    #     self.contours = [contour for contour in self.contours if cv2.contourArea(contour) >= min_area_threshold]
    #     self.contours_filtered_area = len(self.contours)
    #
    # def contours_from_sum(self, density_threshold, min_area_threshold, directionality = 'higher'):
    #     # Find contours coordinates   - based on sopencv
    #     if directionality == 'higher':
    #         self.contours, _ = cv2.findContours((self.local_sum_image > density_threshold).astype(np.uint8), cv2.RETR_LIST,
    #                                     cv2.CHAIN_APPROX_SIMPLE)
    #     elif directionality == 'lower':
    #         self.contours, _ = cv2.findContours((self.local_sum_image < density_threshold).astype(np.uint8), cv2.RETR_LIST,
    #                                     cv2.CHAIN_APPROX_SIMPLE)
    #     else:
    #         self.logger.error('directionality can only be -- lower -- for find contours of areas with lower'
    #               'density or -- higher -- to find contours of areas with higher density'  )
    #         return
    #
    #     self.check_contours()
    #     self.total_valid_contours = len(self.contours)
    #     self.filter_contours_area(min_area_threshold)
    #
    # #########
    # # other filtering of contours
    #
    # def filter_contours_no_counts(self):
    #     # todo check what is more efficient
    #
    #     # array2d = np.sum(self.array_to_contour, axis=2)
    #     # # Iterate through contours andeliminate those without points inside
    #     # valid_contours = []
    #     # for contour in contours:
    #     #     mask_ = np.zeros_like(array2d, dtype=np.uint8)
    #     #     cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
    #     #     sum = np.sum(array2d * mask_, axis=(0, 1)).astype(np.int16)
    #     #     if sum>0:
    #     #         valid_contours.append(contour)
    #
    #     array2d = np.sum(self.array, axis=2)
    #     valid_contours = []
    #
    #     mask_all = np.zeros_like(array2d, dtype=np.uint8)
    #     cv2.drawContours(mask_all, self.contours, -1, 1, thickness=cv2.FILLED)
    #     # Multiply once to get the masked array
    #     masked_array2d = array2d * mask_all
    #
    #     for contour in self.contours:
    #         mask_ = np.zeros_like(array2d, dtype=np.uint8)
    #         cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
    #
    #         # Use the precomputed masked_array2d to check for valid contours
    #         if np.sum(masked_array2d * mask_) > 0:
    #             valid_contours.append(contour)
    #
    #     self.contours = valid_contours
    #     return valid_contours
    #
    #
    # def filter_contours_GD(self, arraygd, arrayab):
    #
    #     # Iterate through contours andeliminate those without points inside
    #     valid_contours = []
    #     i=0
    #     for contour in self.contours:
    #         i+=1
    #         mask_ = np.zeros((arraygd.shape[0], arraygd.shape[1]), dtype=np.uint8)
    #
    #         cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
    #         gene_counts_g = np.sum(arraygd[:, :, 0] * mask_)
    #         gene_counts_d = np.sum(arraygd[:, :, 1] * mask_)
    #         total_sum_gd = gene_counts_d + gene_counts_g
    #         gene_counts_gd = (gene_counts_g,gene_counts_d)
    #
    #         gene_counts_ab = np.sum(np.sum(arrayab, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #         total_sum_ab = np.sum(gene_counts_ab)
    #
    #         if total_sum_ab > total_sum_gd:
    #             self.logger.info(f'Excluding contour {i}. AB counts {total_sum_ab} bigger than GD {total_sum_gd}')
    #         elif gene_counts_gd[0] < 1 or gene_counts_gd[1] < 1:
    #             self.logger.info(f'Excluding contour {i}. Either G or D counts are zero {gene_counts_gd}')
    #         else:
    #             valid_contours.append(contour)
    #             self.logger.info(
    #                 f'Keeping contour {i}. G or D counts are not zero {gene_counts_gd}, AB counts {total_sum_ab} lower than GD {total_sum_gd}')
    #
    #     self.logger.info(f'Number of contours remaining: {len(valid_contours)}')
    #     self.contours = valid_contours
    #
    #
    # def filter_contours_cd8(self, arraycd8, arrayab, array_cd4, arraygd):
    #     # Iterate through contours andeliminate those without points inside
    #     valid_contours = []
    #     i=0
    #     for contour in contours:
    #         i+=1
    #         mask_ = np.zeros((arraycd8.shape[0], arraycd8.shape[1]), dtype=np.uint8)
    #
    #         cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
    #         gene_counts_a = np.sum(arrayab[:, :, 0] * mask_)
    #         gene_counts_b = np.sum(arrayab[:, :, 1] * mask_)
    #         gene_counts_cd8 = np.sum(arraycd8[:, :, 1] * mask_)
    #         gene_counts_cd4 = np.sum(array_cd4[:, :, 0] * mask_)
    #
    #
    #         gene_counts_gd =  np.sum(np.sum(arraygd, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #         gene_counts_ab = np.sum(np.sum(arrayab, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #
    #
    #         # total_sum_abcd8 = gene_counts_a + gene_counts_b + gene_counts_cd8
    #         gene_counts_abcd8 = (gene_counts_a,gene_counts_b, gene_counts_cd8)
    #         gene_counts_ab = np.sum(np.sum(arrayab, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #         total_sum_ab = np.sum(gene_counts_ab)
    #
    #         if gene_counts_a < 1 or gene_counts_b < 1 or gene_counts_cd8 < 1:
    #             logging.info(f'excluding contour {i}. either a,b or cd8  counts are zero {gene_counts_abcd8}')
    #             continue
    #         if gene_counts_cd8 < 2:
    #             logging.info(f'excluding contour {i}. cd8  counts inferior to 2 {gene_counts_abcd8}')
    #             continue
    #         elif gene_counts_gd > gene_counts_ab:
    #             logging.info(f'excluding contour {i}. ab counts {gene_counts_ab} lower than gd {gene_counts_gd}')
    #             continue
    #         elif gene_counts_cd4 > gene_counts_cd8:
    #             logging.info(f'excluding contour {i}. cd4 counts {gene_counts_cd4} are higher than cd8 {gene_counts_cd8}')
    #             continue
    #
    #         else:
    #             valid_contours.append(contour)
    #             logging.info(
    #             f'kipping contour {i}. a,b and cd8  counts are not zero {gene_counts_abcd8}')
    #
    #     logging.info(f'number of contours remaining {len(valid_contours)}')
    #     self.contours = valid_contours
    #
    # def plot_contours_scatter(self, path=None, show=False, s=0.1, alpha=0.5, linewidth=1,
    #                           c_points='blue', c_contours='red',
    #                           figsize=(10, 10), ax=None, **kwargs):
    #     """
    #     Plot scatter plot with contours.
    #
    #     :param path: Path to save the plot
    #     :param show: Whether to display the plot
    #     :param s: Size of scatter points
    #     :param alpha: Alpha transparency of scatter points
    #     :param linewidth: Line width for contours
    #     :param c_points: Color of scatter points
    #     :param c_contours: Color of contours
    #     :param ax: Axes object to draw the plot on (default is None, plot is drawn on the current axes)
    #     :param kwargs: Additional keyword arguments for scatter and plot
    #     """
    #     x, y = np.where(np.sum(self.array, axis=2) > 0)
    #
    #     if ax is None:
    #         plt.figure(figsize=figsize)
    #         ax = plt.gca()
    #
    #     # Extract specific kwargs for scatter and plot if provided
    #     scatter_kwargs = kwargs.get('scatter_kwargs', {})
    #     plot_kwargs = kwargs.get('plot_kwargs', {})
    #
    #     # Scatter plot with original coordinates
    #     ax.scatter(x, y, c=c_points, marker='.', s=s, alpha=alpha, **scatter_kwargs)
    #
    #     # Rescale and plot the contours
    #     for contour in self.contours:
    #         ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=c_contours, **plot_kwargs)
    #
    #     ax.set_title(f'Scatter with contours and genes {self.contour_name}')
    #
    #     if path is not None:
    #         save_path = os.path.join(path, f'Scatter_contours_{self.contour_name}.png')
    #         plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    #         self.logger.info(f'Plot saved at {save_path}')
    #
    #     if show:
    #         plt.show()
    #
    #     return ax
    #
    #
    # def plot_conv_sum(self, cmap='plasma', c_countour='white', path=None, show=False, figsize=(10, 10), ax=None):
    #     """
    #     Plot the convolution sum image with contours.
    #
    #     :param cmap: Colormap for the convolution sum image (default is 'plasma')
    #     :param c_countour: Color for the contours (default is 'white')
    #     :param path: Path to save the plot (default is None, plot is not saved)
    #     :param show: Whether to display the plot (default is False)
    #     :param ax: Axes object to draw the plot on (default is None, plot is drawn on the current axes)
    #     """
    #     if ax is None:
    #         plt.figure(figsize=figsize)
    #         ax = plt.gca()
    #
    #     im = ax.imshow(self.local_sum_image.T, cmap=cmap, interpolation='none', origin='lower')
    #     ax.set_title(f'Count distribution with contours for {self.contour_name}')
    #     ax.set_xlabel('X-axis')
    #     ax.set_ylabel('Y-axis')
    #
    #     # Rescale and plot the contours
    #     for contour in self.contours:
    #         ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=c_countour)
    #
    #     # Add a colorbar for the colormap
    #     # cbar = plt.colorbar(im, ax=ax)
    #     # cbar.set_label('Color scale', rotation=270)
    #     # Create a divider for the existing axes instance
    #     divider = make_axes_locatable(ax)
    #     # Append axes to the right of ax, with 5% width of ax
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #
    #     # Create colorbar in the appended axes
    #     # `cax` argument places the colorbar in the cax axes
    #     cbar = plt.colorbar(im, cax=cax)
    #     cbar.set_label('Color scale', rotation=270)
    #
    #     if path is not None:
    #         save_path = os.path.join(path, f'count_dist_contours_{self.contour_name}.png')
    #         plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    #
    #     if show:
    #         plt.show()
    #
    #     return ax

# What CHATGPT saied. look into the questions
    # Suggestions / Questions
    # 1.
    # Binning
    # Coordinates – Potential
    # Fix in Centroid
    # Calculation
    # You
    # multiply
    # the
    # mean
    # of
    # grid
    # coordinates
    # by
    # bin_size, but
    # maybe
    # it
    # should
    # just
    # be
    # left as is since
    # df['X'] // bin_size
    # already
    # reduces
    # resolution? Example:
    #
    # python
    # Copiar
    # Editar
    # adata.obs['x_centroid'] = df.groupby(['x_grid', 'y_grid'])['X'].mean().values
    # Or
    # clarify if you
    # 're trying to reverse-map to original coordinates.
    #
    # 2.
    # SOM
    # Clustering
    # Optimization
    # You
    # might
    # consider
    # adding
    # a
    # feature
    # for PCA / UMAP / tSNE before SOM to reduce noise and improve separation — e.g.:
    #
    # python
    # Copiar
    # Editar
    # from sklearn.decomposition import PCA
    # X_reduced = PCA(n_components=20).fit_transform(adata.X)
    # And
    # use
    # X_reduced
    # for SOM if needed.
    #
    # 3.
    # Plot
    # Color
    # Issues
    # In
    # plot_som, cmap(idx / max(...))
    # assumes
    # cmap is a
    # colormap
    # object.But if cmap is passed as a
    # string
    # like
    # 'tab10', it’ll
    # throw
    # an
    # error.Consider:
    #
    # python
    # Copiar
    # Editar
    # import matplotlib.cm as cm
    # if isinstance(cmap, str):
    #     cmap = cm.get_cmap(cmap)
    # 4.
    # create_cluster_image
    # Coordinate
    # Confusion
    # You
    # 're assigning image values using [x_start:x_start+grid_size, y_start:y_start+grid_size], which might flip axes since images are indexed as [rows, columns] = [y, x]. Consider switching them or at least verify alignment with actual spatial axes.
    #
    # 5.
    # Add
    # a
    # run_pipeline()
    # Wrapper
    # To
    # enhance
    # usability, wrap
    # the
    # full
    # logic — binning, preprocessing, SOM, visualization — into
    # a
    # single
    # run_pipeline()
    # method
    # with optional kwargs.
    #
    # ❓Possible
    # Additions
    # Want
    # to
    # add
    # overlapping
    # bins
    # with stride < bin_size?
    #
    # Are
    # you
    # planning
    # to
    # support
    # additional
    # clustering
    # methods
    # like
    # KMeans, Leiden, or GMM
    # for comparison?
    #
    # Any
    # downstream
    # integration
    # with pathology image overlays (e.g.via napari or qupath)?
    #
