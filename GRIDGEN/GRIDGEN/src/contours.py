import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

class GetContour():
    """
    A class to handle contour extraction and filtering from a 3D array.

    Attributes
    ----------
    array : np.ndarray
        The 3D array from which contours are to be extracted. xand y positions are the first two dimensions.
        each gene is expected in the z dimension.
    local_sum_image : np.ndarray
        The 2D array representing the local sum of the input array.
    contours : list
        List of contours extracted from the array.
    contour_name : str
        Name of the contour for identification.
    total_valid_contours : int
        Total number of valid contours after filtering.
    contours_filtered_area : int
        Number of contours remaining after area filtering.
    logger : logging.Logger
        Logger for logging information and errors.

    Methods
    -------
    __init__(array_to_contour, logger=None, contour_name=None):
        Initializes the GetContour class with the given array and optional logger and contour name.
    get_conv_sum(kernel_size, kernel_shape='square'):
        Computes the convolution sum of the array with a specified kernel.
    check_contours():
        Checks and processes the contours to ensure they are valid.
    filter_contours_area(min_area_threshold):
        Filters contours based on a minimum area threshold.
    contours_from_sum(density_threshold, min_area_threshold, directionality='higher'):
        Extracts contours from the local sum image based on a density threshold and filters them by area.
    filter_contours_no_counts():
        Filters contours that have no counts in the given array.
    filter_contours_by_gene_threshold(gene_array, threshold, gene_name=""):
        Filters contours based on a gene count threshold.
    filter_contours_by_gene_comparison(gene_array1, gene_array2, gene_name1="", gene_name2=""):
        Filters contours based on the comparison of gene counts between two gene arrays.
    plot_contours_scatter(path=None, show=False, s=0.1, alpha=0.5, linewidth=1, c_points='blue', c_contours='red', figsize=(10, 10), ax=None, **kwargs):
        Plot scatter plot with contours.
    plot_conv_sum(cmap='plasma', c_countour='white', path=None, show=False, figsize=(10, 10), ax=None):
        Plot the convolution sum image with contours.

    """

    def __init__(self, array_to_contour, logger=None,contour_name = None):
        """
        Initializes the GetContour class with the given array and optional logger and contour name.

        Parameters
        ----------
        array_to_contour : np.ndarray
            The 3D array from which contours are to be extracted.
        logger : logging.Logger, optional
            Logger for logging information and errors (default is None, which configures a default logger).
        contour_name : str, optional
            Name of the contour for identification (default is None).
        """
        self.array = array_to_contour
        self.local_sum_image = None
        self.contours = None
        self.contour_name = contour_name
        self.total_valid_contours = 0
        self.contours_filtered_area = 0
        if logger is None:
            # Configure default logger if none is provided
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


    def get_conv_sum(self, kernel_size, kernel_shape='square'):
        """
        Computes the convolution sum of the array with a specified kernel.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel to be used for convolution.
        kernel_shape : str, optional
            Shape of the kernel ('square' or 'circle'), by default 'square'.
        """

        kernel = np.ones((kernel_size, kernel_size))
        if kernel_shape == 'circle':
            diameter = kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        self.local_sum_image = cv2.filter2D(np.sum(self.array, axis=2), -1, kernel)


    def check_contours(self):
        """
        Checks and processes the contours to ensure they are valid.
        Excludes contours with less than 3 points and ensures contours are closed.
        """

        # exclude contours with less than 3 points
        self.contours = [np.squeeze(contour).astype(np.int32) for contour in self.contours if len(contour) > 2]
        # add the last coordinate to the list if the contour is not closed
        self.contours = [np.vstack([contour, contour[0]]) if contour[0].tolist() != contour[-1].tolist() else contour for
                     contour in self.contours]
        # transform to np int 32 to compatibility with opencv
        self.contours = [np.array(contour, dtype=np.int32) for contour in self.contours]

    def filter_contours_area(self,min_area_threshold):
        """
        Filters contours based on a minimum area threshold.

        Parameters
        ----------
        min_area_threshold : float
            Minimum area threshold for filtering contours.
        """
        self.contours = [contour for contour in self.contours if cv2.contourArea(contour) >= min_area_threshold]
        self.contours_filtered_area = len(self.contours)

    def contours_from_sum(self, density_threshold, min_area_threshold, directionality = 'higher'):
        """
       Extracts contours from the local sum image based on a density threshold and filters them by area.

       Parameters
       ----------
       density_threshold : float
           Density threshold for extracting contours.
       min_area_threshold : float
           Minimum area threshold for filtering contours.
       directionality : str, optional
           Directionality for finding contours ('higher' or 'lower'), by default 'higher'.
       """
        # Find contours coordinates   - based on sopencv
        if directionality == 'higher':
            self.contours, _ = cv2.findContours((self.local_sum_image > density_threshold).astype(np.uint8), cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        elif directionality == 'lower':
            self.contours, _ = cv2.findContours((self.local_sum_image < density_threshold).astype(np.uint8), cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        else:
            self.logger.error('directionality can only be -- lower -- for find contours of areas with lower'
                  'density or -- higher -- to find contours of areas with higher density'  )
            return

        self.filter_contours_no_counts()
        self.check_contours()
        self.total_valid_contours = len(self.contours)
        self.filter_contours_area(min_area_threshold)
        # todo does this add a lot of time. why are ontours that are zero counts. check contour displacement




    #############################
    # other filtering of contours
    def filter_contours_no_counts(self):
        """
        Filters contours that have no counts in the given array.

        This method iterates through each contour, creates a mask for the contour, and checks if the sum of the
        masked array within the contour is greater than 0. Contours are kept if they have counts.

        Returns
        -------
        list
            The list of valid contours that have counts.
        """
        # todo check what is more efficient

        # array2d = np.sum(self.array_to_contour, axis=2)
        # # Iterate through contours andeliminate those without points inside
        # valid_contours = []
        # for contour in contours:
        #     mask_ = np.zeros_like(array2d, dtype=np.uint8)
        #     cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
        #     sum = np.sum(array2d * mask_, axis=(0, 1)).astype(np.int16)
        #     if sum>0:
        #         valid_contours.append(contour)

        array2d = np.sum(self.array, axis=2)
        valid_contours = []

        mask_all = np.zeros_like(array2d, dtype=np.uint8)
        cv2.drawContours(mask_all, self.contours, -1, 1, thickness=cv2.FILLED)
        # Multiply once to get the masked array
        masked_array2d = array2d * mask_all

        for contour in self.contours:
            mask_ = np.zeros_like(array2d, dtype=np.uint8)
            cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)

            # Use the precomputed masked_array2d to check for valid contours
            if np.sum(masked_array2d * mask_) > 0:
                valid_contours.append(contour)

        self.contours = valid_contours
        return valid_contours



    def filter_contours_by_gene_threshold(self, gene_array, threshold, gene_name = ""):
        """
        Filters contours based on a gene count threshold.

        This method iterates through each contour, creates a mask for the contour, and calculates the gene count
        within the masked region. Contours are kept if the gene count is greater than or equal to the threshold.

        Parameters
        ----------
        gene_array : np.ndarray
            The gene array to be checked. Can be a 2D or 3D array.
        threshold : float
            The gene count threshold for filtering contours.
        gene_name : str, optional
            Name of the gene for logging purposes (default is an empty string).

        Returns
        -------
        None
            The method updates the `self.contours` attribute with the valid contours.
        """
        valid_contours = []
        for i, contour in enumerate(self.contours):
            mask_ = np.zeros((gene_array.shape[0], gene_array.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
            gene_count = np.sum(gene_array * mask_)
            if gene_count >= threshold:
                valid_contours.append(contour)
            else:
                self.logger.info(f'Excluding contour {i}. Gene {gene_name} count  {gene_count} is below threshold {threshold}')
        self.contours = valid_contours
        self.logger.info(f'Number of contours remaining: {len(valid_contours)}')


    def filter_contours_by_gene_comparison(self, gene_array1, gene_array2, gene_name1="", gene_name2 = "" ):
        """
        Filters contours based on the comparison of gene counts between two gene arrays.

        This method iterates through each contour, creates a mask for the contour, and calculates the gene counts
        for the given gene arrays within the masked region. Contours are kept if the gene count in `gene_array1`
        is greater than the gene count in `gene_array2`.

        Parameters
        ----------
        gene_array1 : np.ndarray
            The first gene array to be compared. Can be a 2D or 3D array.
        gene_array2 : np.ndarray
            The second gene array to be compared. Can be a 2D or 3D array.
        gene_name1 : str, optional
            Name of the first gene for logging purposes (default is an empty string).
        gene_name2 : str, optional
            Name of the second gene for logging purposes (default is an empty string).

        Returns
        -------
        None
            The method updates the `self.contours` attribute with the valid contours.
        """

        valid_contours = []
        for i, contour in enumerate(self.contours):
            mask_ = np.zeros((gene_array1.shape[0], gene_array1.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
            if gene_array1.ndim > 2:
                gene_array1 = np.sum(gene_array1, axis=-1)
            if gene_array2.ndim > 2:
                gene_array2 = np.sum(gene_array2, axis=-1)

            gene_count1 = np.sum(gene_array1 * mask_)
            gene_count2 = np.sum(gene_array2 * mask_)
            if gene_count1 > gene_count2:
                valid_contours.append(contour)
            else:
                self.logger.info(f'Excluding contour {i}. Gene {gene_name1} count {gene_count1} is not greater than gene {gene_name2} count {gene_count2}')
        self.contours = valid_contours
        self.logger.info(f'Number of contours remaining: {len(valid_contours)}')



    def plot_contours_scatter(self, path=None, show=False, s=0.1, alpha=0.5, linewidth=1,
                              c_points='blue', c_contours='red',
                              figsize=(10, 10), ax=None, **kwargs):
        """
        Plot scatter plot with contours.

        :param path: Path to save the plot
        :param show: Whether to display the plot
        :param s: Size of scatter points
        :param alpha: Alpha transparency of scatter points
        :param linewidth: Line width for contours
        :param c_points: Color of scatter points
        :param c_contours: Color of contours
        :param ax: Axes object to draw the plot on (default is None, plot is drawn on the current axes)
        :param kwargs: Additional keyword arguments for scatter and plot
        """
        x, y = np.where(np.sum(self.array, axis=2) > 0)

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        # Extract specific kwargs for scatter and plot if provided
        scatter_kwargs = kwargs.get('scatter_kwargs', {})
        plot_kwargs = kwargs.get('plot_kwargs', {})

        # Scatter plot with original coordinates
        ax.scatter(x, y, c=c_points, marker='.', s=s, alpha=alpha, **scatter_kwargs)

        # Rescale and plot the contours
        for contour in self.contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=c_contours, **plot_kwargs)

        ax.set_title(f'Scatter with contours and genes {self.contour_name}')

        if path is not None:
            save_path = os.path.join(path, f'Scatter_contours_{self.contour_name}.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        if show:
            plt.show()

        return ax


    def plot_conv_sum(self, cmap='plasma', c_countour='white', path=None, show=False, figsize=(10, 10), ax=None):
        """
        Plot the convolution sum image with contours.

        :param cmap: Colormap for the convolution sum image (default is 'plasma')
        :param c_countour: Color for the contours (default is 'white')
        :param path: Path to save the plot (default is None, plot is not saved)
        :param show: Whether to display the plot (default is False)
        :param ax: Axes object to draw the plot on (default is None, plot is drawn on the current axes)
        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        im = ax.imshow(self.local_sum_image.T, cmap=cmap, interpolation='none', origin='lower')
        ax.set_title(f'Count distribution with contours for {self.contour_name}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Rescale and plot the contours
        for contour in self.contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=c_countour)

        # Add a colorbar for the colormap
        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label('Color scale', rotation=270)
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax, with 5% width of ax
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create colorbar in the appended axes
        # `cax` argument places the colorbar in the cax axes
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Color scale', rotation=270)

        if path is not None:
            save_path = os.path.join(path, f'count_dist_contours_{self.contour_name}.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')

        if show:
            plt.show()

        return ax




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

    # def filter_contours_GD_xenium(self, arraygd, arrayab):
    #     # array gd is not G/D but G1/G2/D and ab is B1,B2, A
    #     # Iterate through contours andeliminate those without points inside
    #     valid_contours = []
    #     i = 0
    #     for contour in self.contours:
    #         i += 1
    #         mask_ = np.zeros((arraygd.shape[0], arraygd.shape[1]), dtype=np.uint8)
    #
    #         cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
    #         gene_counts_g = np.sum(arraygd[:, :, 0] * mask_) + np.sum(arraygd[:, :, 1] * mask_)
    #         gene_counts_d = np.sum(arraygd[:, :, 2] * mask_)
    #         total_sum_gd = gene_counts_d + gene_counts_g
    #         gene_counts_gd = (gene_counts_g, gene_counts_d)
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
    # def filter_contours_cd8_xenium(self, arraycd8, arrayab, array_cd4, arraygd):
    #     # array gd is not G/D but G1/G2/D and ab is B1,B2, A
    #     # Cd8 is CD8A (dont have B) B1,B2 and A
    #     # Iterate through contours andeliminate those without points inside
    #     valid_contours = []
    #     i = 0
    #     for contour in self.contours:
    #         i += 1
    #         mask_ = np.zeros((arraycd8.shape[0], arraycd8.shape[1]), dtype=np.uint8)
    #
    #         cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
    #         gene_counts_a = np.sum(arrayab[:, :, 2] * mask_)
    #         gene_counts_b = np.sum(arrayab[:, :, 1] * mask_) + np.sum(arrayab[:, :, 0] * mask_)
    #         gene_counts_cd8 = np.sum(arraycd8[:, :, 1] * mask_)
    #         gene_counts_cd4 = np.sum(array_cd4[:, :, 0] * mask_)
    #
    #         gene_counts_gd = np.sum(np.sum(arraygd, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #         gene_counts_ab = np.sum(np.sum(arrayab, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #
    #         # total_sum_abcd8 = gene_counts_a + gene_counts_b + gene_counts_cd8
    #         gene_counts_abcd8 = (gene_counts_a, gene_counts_b, gene_counts_cd8)
    #         gene_counts_ab = np.sum(np.sum(arrayab, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #         total_sum_ab = np.sum(gene_counts_ab)
    #
    #         if gene_counts_a < 1 or gene_counts_b < 1 or gene_counts_cd8 < 1:
    #             logging.info(f'excluding contour {i}. either a,b or cd8  counts are zero {gene_counts_abcd8}')
    #             continue
    #         if gene_counts_cd8 < 1:
    #             logging.info(f'excluding contour {i}. cd8  counts inferior to 1 {gene_counts_abcd8}')
    #             continue
    #         elif gene_counts_gd > gene_counts_ab:
    #             logging.info(f'excluding contour {i}. ab counts {gene_counts_ab} lower than gd {gene_counts_gd}')
    #             continue
    #         elif gene_counts_cd4 > gene_counts_cd8:
    #             logging.info(
    #                 f'excluding contour {i}. cd4 counts {gene_counts_cd4} are higher than cd8 {gene_counts_cd8}')
    #             continue
    #
    #         else:
    #             valid_contours.append(contour)
    #             logging.info(
    #                 f'kipping contour {i}. a,b and cd8  counts are not zero {gene_counts_abcd8}')
    #
    #     logging.info(f'number of contours remaining {len(valid_contours)}')
    #     self.contours = valid_contours
    #
    # def filter_contours_cd8(self, arraycd8, arrayab, array_cd4, arraygd):
    #     #
    #     # Iterate through contours andeliminate those without points inside
    #     valid_contours = []
    #     i = 0
    #     for contour in self.contours:
    #         i += 1
    #         mask_ = np.zeros((arraycd8.shape[0], arraycd8.shape[1]), dtype=np.uint8)
    #
    #         cv2.drawContours(mask_, [contour], -1, 1, thickness=cv2.FILLED)
    #         gene_counts_a = np.sum(arrayab[:, :, 0] * mask_)
    #         gene_counts_b = np.sum(arrayab[:, :, 1] * mask_)
    #         gene_counts_cd8 = np.sum(arraycd8[:, :, 1] * mask_)
    #         gene_counts_cd4 = np.sum(array_cd4[:, :, 0] * mask_)
    #
    #         gene_counts_gd = np.sum(np.sum(arraygd, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #         gene_counts_ab = np.sum(np.sum(arrayab, axis=2) * mask_, axis=(0, 1)).astype(np.int64)
    #
    #         # total_sum_abcd8 = gene_counts_a + gene_counts_b + gene_counts_cd8
    #         gene_counts_abcd8 = (gene_counts_a, gene_counts_b, gene_counts_cd8)
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
    #             logging.info(
    #                 f'excluding contour {i}. cd4 counts {gene_counts_cd4} are higher than cd8 {gene_counts_cd8}')
    #             continue
    #
    #         else:
    #             valid_contours.append(contour)
    #             logging.info(
    #                 f'kipping contour {i}. a,b and cd8  counts are not zero {gene_counts_abcd8}')
    #
    #     logging.info(f'number of contours remaining {len(valid_contours)}')
    #     self.contours = valid_contours

