import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, rgb2hex
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RadioButtons, TextBox
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram,\
    set_link_color_palette
from scipy.spatial.distance import squareform
from scipy.stats.stats import pearsonr
from nccluster.multisliceviewer import MultiSliceViewer
from tqdm import tqdm


class CorrelationMapperBase:

    def __init__(self, corr_mat, map_shape, fig=None):

        # initialise class attributes for future reference
        self.corr_mat = corr_mat
        self.n_rows, self.n_cols = map_shape

        # create a new figure or use an existing one if supplied
        self.fig = fig if fig else plt.figure()


class CorrelationMapper(CorrelationMapperBase):

    def __init__(self, corr_mat, map_shape, fig=None):

        super().__init__(corr_mat, map_shape, fig=fig)

        # default the location to analyse to the middle of the map
        self.corr_loc = [int(self.n_rows/2), int(self.n_cols/2)]

        # the main Axes of this interactive viewer
        self.init_corr_ax()

        self.update_plots()

    def init_corr_ax(self):

        # initialise the plot that shows correlation map for clicked point
        self.corr_ax = self.fig.add_subplot()

        # trick to ensure colorbar spands the range of valuese present in the
        # correlation matrix
        norm = Normalize(vmin=np.nanmin(self.corr_mat),
                         vmax=np.nanmax(self.corr_mat)
                         )

        # put the colorbar inside inset axes by the correlation map
        cax = self.corr_ax.inset_axes([1.04, 0, 0.05, 1])
        self.fig.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'),
                          ax=self.corr_ax,
                          cax=cax,
                          ticks=np.linspace(-1, 1, num=5, endpoint=True)
                          )

        # initialise the correlation map with zeroes in the expected map shape
        # values don't matter because we'll call update_corr_map()
        self.corr_ax_image = self.corr_ax.imshow(
            np.zeros([self.n_rows, self.n_cols]),
            origin='lower',
            norm=norm,
            cmap='coolwarm'
        )

        # listen for click events only when mouse over corr_ax
        self.enter_corr_ax_cid = self.fig.canvas.mpl_connect(
            'axes_enter_event', self.enter_corr_ax_event)
        self.exit_corr_ax_cid = self.fig.canvas.mpl_connect(
            'axes_leave_event', self.leave_corr_ax_event)

    def update_corr_loc_marker(self):
        # clear previous marker if present
        try:
            for handle in self.helper_point:
                handle.remove()
        except AttributeError:
            pass

        # add new star-shaped marker in clicked location
        try:
            x_pos, y_pos = self.corr_loc
            self.helper_point = self.corr_ax.plot(
                x_pos,
                y_pos,
                color='black',
                marker='*'
            )
        except AttributeError:
            pass

    def process_corr_ax_click(self, event):
        # ignore clicks outside the plots
        if not event.inaxes:
            return

        # get position of click
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)

        # save last clicked point
        self.corr_loc = [x_pos, y_pos]

        # update plots based on new location to be analysed
        self.update_plots()

        # refresh figure
        self.fig.canvas.draw()

    def update_plots(self):
        self.update_corr_map()
        self.update_corr_loc_marker()

    def update_corr_map(self):

        # unpack coords of last clicked point
        (x_pos, y_pos) = self.corr_loc

        # translate x-y coords of click to index of grid point in flat array
        flat_index = y_pos * self.n_rows + x_pos

        # get the corresponding row in the correlation map
        corr_array = self.corr_mat[flat_index]

        # shape this row into a 2D array and plot
        corr_map = np.reshape(corr_array,
                              [self.n_rows, self.n_cols]
                              )
        self.corr_ax_image.set_data(corr_map)

    def enter_corr_ax_event(self, event):
        # check whether the entered ax is the correlation map
        if event.inaxes == self.corr_ax:
            # if so, listen for clicks
            self.click_corr_ax_cid = self.fig.canvas.mpl_connect(
                'button_press_event', self.process_corr_ax_click)

    def leave_corr_ax_event(self, event):
        # check whether the exited ax is the correlation map
        if event.inaxes == self.corr_ax:
            # if so, stop listening for clicks
            self.fig.canvas.mpl_disconnect(
                self.click_corr_ax_cid
            )


class FclusterViewer(CorrelationMapperBase):

    def __init__(self, corr_mat, map_shape, fig=None, cmap='rainbow'):

        super().__init__(corr_mat, map_shape, fig=fig)
        self.cmap = cmap

        self.init_widgets()

        # with the above can now perform correlation clustering
        # and initialise the correlation map
        self.init_cluster_ax()

        # call a first update to prepare the plots for viewing
        self.update_plots()

    def init_widgets(self):
        # initialise widgets to tweak clustering
        self.init_linkage_method_radio_ax()
        self.init_fcluster_criterion_radio_ax()
        self.init_fcluster_thresh_textbox_ax()

        # set some class attributes from the values of widgets
        # like multiple selection and text input
        self.linkage_method = self.linkage_method_radio.value_selected
        self.fcluster_criterion = self.fcluster_criterion_radio.value_selected
        self.fcluster_thresh = float(self.fcluster_thresh_textbox.text)

    def init_linkage_method_radio_ax(self):
        # radio buttons for changing clustering method
        # create a new ax and set its title
        self.linkage_method_radio_ax = self.fig.add_subplot(331)
        self.linkage_method_radio_ax.set_title('Linkage method')

        # create the radio button with predefined labels
        self.linkage_method_radio = RadioButtons(
            self.linkage_method_radio_ax,
            ('single', 'complete', 'average', 'weighted', 'centroid',
             'median', 'ward'),
            active=6
        )

        # set the radio button to call this function when clicked
        self.linkage_method_radio.on_clicked(
            self.linkage_method_radio_on_click
        )

    def init_fcluster_criterion_radio_ax(self):
        # see init_linkage_method_radio_ax comments
        self.fcluster_criterion_radio_ax = self.fig.add_subplot(334)
        self.fcluster_criterion_radio_ax.set_title('Fcluster criterion')

        self.fcluster_criterion_radio = RadioButtons(
            self.fcluster_criterion_radio_ax,
            ('inconsistent', 'distance', 'maxclust', 'monocrit',
             'maxclust_monocrit'),
            active=1
        )

        self.fcluster_criterion_radio.on_clicked(
            self.fcluster_criterion_radio_on_click
        )

    def init_fcluster_thresh_textbox_ax(self):
        # create an ax and give it a title
        self.fcluster_thresh_textbox_ax = self.fig.add_subplot(337)
        self.fcluster_thresh_textbox_ax.set_title("Fcluster threshold")

        # create the textbox giving it an initial value
        self.fcluster_thresh_textbox = TextBox(
            ax=self.fcluster_thresh_textbox_ax,
            label='value:',
            initial='5'
        )

        # tell the textbox to call this function when new text is submitted
        self.fcluster_thresh_textbox.on_submit(
            self.fcluster_thresh_textbox_on_submit
        )

    def init_cluster_ax(self):
        # create new ax for the cluster map
        self.cluster_ax = self.fig.add_subplot(222)

        # run correlation clustering to get the 2D array of labels
        _, labels_shaped = self.corr_cluster()

        # map out clusters
        self.cluster_ax_image = self.cluster_ax.imshow(
            labels_shaped,
            origin='lower',
            cmap=self.cmap
        )

        self.cluster_ax.set_xticks([])
        self.cluster_ax.set_yticks([])

    def make_df(self):
        return pd.DataFrame(self.corr_mat, index=None, columns=None)

    def make_droppedna_df(self):
        df = self.make_df()
        return df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    def flat_notna_array_to_map(self, arr, col_name):
        df = self.make_df()
        droppedna = self.make_droppedna_df()

        # put the labels into the original dataframe that includes nans
        df.loc[
                df.index.isin(droppedna.index),
                col_name,
                ] = arr

        # get the labels back as a flat array, now including nans
        labels_flat = np.ma.masked_array(df[col_name])

        # shape the label back into a 2D array for plotting clusters on a map
        labels_shaped = np.reshape(labels_flat, (self.n_rows, self.n_cols))

        return labels_shaped

    def corr_cluster(self):

        droppedna = self.make_droppedna_df()

        # form dataframe back into a correlation matrix (without nans)
        corr = droppedna.to_numpy()

        # corrections to reduce floating point errors
        corr = (corr + corr.T) / 2
        # np.fill_diagonal(corr, 1)

        # convert the correlation coefficients (higher is closer)
        # into distances (lower is closer)
        dissimilarity = 1 - corr

        # corrections to reduce floating point errors
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity, 0)

        # take sqrt of 1 - corr
        dissimilarity = np.sqrt(dissimilarity)

        # dissimilarity matrix needs to be in this form for hierarchical
        # clustering
        square = squareform(dissimilarity)

        # perform hierarchical clustering
        hierarchy = linkage(square, method=self.linkage_method)

        # flatten the hierarchy into usable clusters
        labels = fcluster(hierarchy,
                          self.fcluster_thresh,
                          criterion=self.fcluster_criterion
                          )
        labels_shaped = self.flat_notna_array_to_map(labels, 'labels')

        return (hierarchy, labels_shaped)

    def update_plots(self):
        # re-run correlation clustering and get the labels 2D array
        _, labels_shaped = self.corr_cluster()

        self.update_cluster_map(labels_shaped)

        # refresh canvas
        self.fig.canvas.draw()

    def update_cluster_map(self, labels_shaped):
        # compute and set a new norm based on the new labels
        # this ensures the colors are re-assigned to accommodate more clusters
        labels_min = np.nanmin(labels_shaped)
        labels_max = np.nanmax(labels_shaped)
        norm = Normalize(vmin=labels_min, vmax=labels_max)
        self.cluster_ax_image.set(norm=norm, cmap=self.cmap)

        # set the labels as the cluster plot image
        self.cluster_ax_image.set_data(labels_shaped)

    def linkage_method_radio_on_click(self, label):
        # update class attribute based on newly selected label in radio
        self.linkage_method = label
        # map will update itself using the new linkage method
        self.update_plots()

    def fcluster_criterion_radio_on_click(self, label):
        # update class attribute based on newly selected label in radio
        self.fcluster_criterion = label
        # map will update itself using the new fcluster criterion
        self.update_plots()

    def fcluster_thresh_textbox_on_submit(self, value):
        # update class attribute based on newly input text
        self.fcluster_thresh = float(value)
        # map will update itself using the new fcluster threshold
        self.update_plots()


class CorrelationMatrixViewer:

    def __init__(self, corr_mat, n_rows, n_cols, fig=None, cmap='rainbow'):
        # initialise class attributes for future reference
        self.corr_mat = corr_mat
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.cmap = cmap

        # default the location to analyse to the middle of the map
        self.corr_loc = [int(self.n_rows/2), int(self.n_cols/2)]

        # create a new figure or use an existing one if supplied
        self.fig = fig if fig else plt.figure()

        # initialise a bunch of plots and widgets
        self.init_corr_ax()
        self.init_linkage_method_radio_ax()
        self.init_fcluster_criterion_radio_ax()
        self.init_fcluster_thresh_textbox_ax()

        # set more class attributes from the values of widgets
        # like multiple selection and text input
        self.linkage_method = self.linkage_method_radio.value_selected
        self.fcluster_criterion = self.fcluster_criterion_radio.value_selected
        self.fcluster_thresh = self.fcluster_thresh_textbox.text

        # with the above can now perform correlation clustering
        # and initialise the correlation map
        self.init_cluster_ax()

        # refresh stuff
        self.update_plots()

    def init_corr_ax(self):

        # initialise the plot that shows correlation map for clicked point
        self.corr_ax = self.fig.add_subplot(231)

        # trick to ensure colorbar spands the range of valuese present in the
        # correlation matrix
        norm = Normalize(vmin=np.nanmin(self.corr_mat),
                         vmax=np.nanmax(self.corr_mat)
                         )

        # put the colorbar inside inset axes by the correlation map
        cax = self.corr_ax.inset_axes([1.04, 0, 0.05, 1])
        self.fig.colorbar(ScalarMappable(norm=norm, cmap=self.cmap),
                          ax=self.corr_ax,
                          cax=cax,
                          ticks=np.linspace(-1, 1, num=5, endpoint=True)
                          )

        # initialise the correlation map with zeroes in the expected map shape
        # values don't matter because we'll call update_corr_map()
        self.corr_ax_image = self.corr_ax.imshow(
            np.zeros([self.n_rows, self.n_cols]),
            origin='lower',
            norm=norm,
            cmap=self.cmap
        )

        # listen for click events only when mouse over corr_ax
        self.enter_corr_ax_cid = self.fig.canvas.mpl_connect(
            'axes_enter_event', self.enter_corr_ax_event)
        self.exit_corr_ax_cid = self.fig.canvas.mpl_connect(
            'axes_leave_event', self.leave_corr_ax_event)

    def init_linkage_method_radio_ax(self):
        # radio buttons for changing clustering method
        # create a new ax and set its title
        self.linkage_method_radio_ax = self.fig.add_subplot(234)
        self.linkage_method_radio_ax.set_title('Linkage method')

        # create the radio button with predefined labels
        self.linkage_method_radio = RadioButtons(
            self.linkage_method_radio_ax,
            ('single', 'complete', 'average', 'weighted', 'centroid',
             'median', 'ward'),
            active=1
        )

        # set the radio button to call this function when clicked
        self.linkage_method_radio.on_clicked(
            self.linkage_method_radio_on_click
        )

    def init_fcluster_criterion_radio_ax(self):
        # see init_linkage_method_radio_ax comments
        self.fcluster_criterion_radio_ax = self.fig.add_subplot(235)
        self.fcluster_criterion_radio_ax.set_title('Fcluster criterion')

        self.fcluster_criterion_radio = RadioButtons(
            self.fcluster_criterion_radio_ax,
            ('inconsistent', 'distance', 'maxclust', 'monocrit',
             'maxclust_monocrit'),
            active=1
        )

        self.fcluster_criterion_radio.on_clicked(
            self.fcluster_criterion_radio_on_click
        )

    def init_fcluster_thresh_textbox_ax(self):
        # create an ax and give it a title
        self.fcluster_thresh_textbox_ax = self.fig.add_subplot(236)
        self.fcluster_thresh_textbox_ax.set_title("Fcluster threshold")

        # create the textbox giving it an initial value
        self.fcluster_thresh_textbox = TextBox(
            ax=self.fcluster_thresh_textbox_ax,
            label='value:',
            initial='0.4'
        )

        # tell the textbox to call this function when new text is submitted
        self.fcluster_thresh_textbox.on_submit(
            self.fcluster_thresh_textbox_on_submit
        )

    def init_cluster_ax(self):
        # create new ax for the cluster map
        self.cluster_ax = self.fig.add_subplot(232)

        # run correlation clustering to get the 2D array of labels
        labels_shaped = self.corr_cluster()

        # map out clusters
        self.cluster_ax_image = self.cluster_ax.imshow(
            labels_shaped,
            origin='lower',
            cmap=self.cmap
        )

    def corr_cluster(self):
        # convert correlation matrix to pandas dataframe to drop nan rows/cols
        df = pd.DataFrame(self.corr_mat.data, index=None, columns=None)
        droppedna = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # form dataframe back into a correlation matrix (without nans)
        corr = np.array(droppedna)
        corr = np.reshape(corr, droppedna.shape)

        # corrections to reduce floating point errors
        corr = (corr + corr.T) / 2
        # np.fill_diagonal(corr, 1)

        # convert the correlation coefficients (higher is closer)
        # into distances (lower is closer)
        dissimilarity = 1 - corr

        # corrections to reduce floating point errors
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity, 0)

        # dissimilarity matrix needs to be in this form for hierarchical
        # clustering
        square = squareform(dissimilarity)

        # perform hierarchical clustering
        hierarchy = linkage(square, method=self.linkage_method)

        # flatten the hierarchy into usable clusters
        labels = fcluster(hierarchy,
                          self.fcluster_thresh,
                          criterion=self.fcluster_criterion
                          )

        # put the labels into the original dataframe that includes nans
        df.loc[
                df.index.isin(droppedna.index),
                'labels'
                ] = labels

        # get the labels back as a flat array, now including nans
        labels_flat = np.ma.masked_array(df['labels'])

        # shape the label back into a 2D array for plotting clusters on a map
        labels_shaped = np.reshape(labels_flat, (self.n_rows, self.n_cols))

        return labels_shaped

    def update_corr_loc_marker(self):
        # clear previous marker if present
        try:
            for handle in self.helper_point:
                handle.remove()
        except AttributeError:
            pass

        # add new star-shaped marker in clicked location
        try:
            x_pos, y_pos = self.corr_loc
            self.helper_point = self.corr_ax.plot(
                x_pos,
                y_pos,
                color='black',
                marker='*'
            )
        except AttributeError:
            pass

    def process_corr_ax_click(self, event):
        # ignore clicks outside the plots
        if not event.inaxes:
            return

        # get position of click
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)

        # save last clicked point
        self.corr_loc = [x_pos, y_pos]

        # update plots based on new location to be analysed
        self.update_plots()

        # refresh figure
        self.fig.canvas.draw()

    def update_plots(self):
        self.update_corr_map()
        self.update_corr_loc_marker()

    def update_corr_map(self):

        # unpack coords of last clicked point
        (x_pos, y_pos) = self.corr_loc

        # translate x-y coords of click to index of grid point in flat array
        flat_index = y_pos * self.n_rows + x_pos

        # get the corresponding row in the correlation map
        corr_array = self.corr_mat[flat_index]

        # shape this row into a 2D array and plot
        corr_map = np.reshape(corr_array,
                              [self.n_rows, self.n_cols]
                              )
        self.corr_ax_image.set_data(corr_map)

    def update_cluster_map(self):
        # re-run correlation clustering and get the labels 2D array
        labels_shaped = self.corr_cluster()

        # compute and set a new norm based on the new labels
        # this ensures the colors are re-assigned to accommodate more clusters
        labels_min = np.nanmin(labels_shaped)
        labels_max = np.nanmax(labels_shaped)
        norm = Normalize(vmin=labels_min, vmax=labels_max)
        self.cluster_ax_image.set(norm=norm, cmap=self.cmap)

        # set the labels as the cluster plot image
        self.cluster_ax_image.set_data(labels_shaped)

        # refresh canvas
        self.fig.canvas.draw()

    def linkage_method_radio_on_click(self, label):
        # update class attribute based on newly selected label in radio
        self.linkage_method = label
        # map will update itself using the new linkage method
        self.update_cluster_map()

    def fcluster_criterion_radio_on_click(self, label):
        # update class attribute based on newly selected label in radio
        self.fcluster_criterion = label
        # map will update itself using the new fcluster criterion
        self.update_cluster_map()

    def fcluster_thresh_textbox_on_submit(self, value):
        # update class attribute based on newly input text
        self.fcluster_thresh = value
        # map will update itself using the new fcluster threshold
        self.update_cluster_map()

    def enter_corr_ax_event(self, event):
        # check whether the entered ax is the correlation map
        if event.inaxes == self.corr_ax:
            # if so, listen for clicks
            self.click_corr_ax_cid = self.fig.canvas.mpl_connect(
                'button_press_event', self.process_corr_ax_click)

    def leave_corr_ax_event(self, event):
        # check whether the exited ax is the correlation map
        if event.inaxes == self.corr_ax:
            # if so, stop listening for clicks
            self.fig.canvas.mpl_disconnect(
                self.click_corr_ax_cid
            )


class CorrelationViewer(MultiSliceViewer, CorrelationMatrixViewer):

    def __init__(self, volume, colorbar=True,
                 cmap='rainbow', corr_mat_file='corr_mat.npy',
                 pval_mat_file='pval_mat.npy', pvalues=True):

        print("[i] Initialising CorrelationViewer")

        # create a new figure
        self.fig = plt.figure()
        # only the evolution plot is new, the rest are inherited
        self.evo_ax = self.fig.add_subplot()

        # call init of MultiSliceViewer parent, passing it the child's fig
        MultiSliceViewer.__init__(self, volume, colorbar=colorbar,
                                  legend=False, cmap=cmap, fig=self.fig)

        # obtain the correlation matrix
        corr_mat = self.get_corr_mat(pvalues,
                                     corr_mat_file=corr_mat_file,
                                     pval_mat_file=pval_mat_file
                                     )

        # still assuming data shape is t, z, y, x
        _, _, y, x = self.volume.shape

        # call init of CorrelationMatrixViewer parent,
        # passing it the child's fig
        CorrelationMatrixViewer.__init__(self, corr_mat, y, x, fig=self.fig)

        # arrange the plot to fit on screen
        self.layout_plots()

        # only plot left to refresh, since other were handled by parents' init
        self.update_evo_plot()

    def get_corr_mat(self, pvalues=False,
                     corr_mat_file="corr_mat.npy",
                     pval_mat_file="pval_mat.npy"):

        # shorthand names for data dimensions
        t, _, y, x = self.volume.shape

        # each row in evolutions is the R-age time series for a grid point
        evolutions = np.reshape(self.surface_slices, [t, x*y]).T

        # try reading correlation and p-value matrices from file
        if int(pvalues):
            try:
                pval_mat = np.load(pval_mat_file)
                print(f"[i] Read in {pval_mat_file}")
                corr_mat = np.load(corr_mat_file)
                print(f"[i] Read in {corr_mat_file}")

            # compute correlation now if not read from file
            except FileNotFoundError:

                # for p-values use pearson's r
                # initialise empty matrices to hold corr coefs and p-value
                pval_mat = np.empty([y*x, y*x])
                corr_mat = np.empty([y*x, y*x])

                # run Pearson's r for every possible pair of grid points
                for i, evo1 in enumerate(tqdm(evolutions,
                                              desc="[i] Computing Pearson's r \
                                              and p-values: ")):
                    # skip grid points missing data
                    if np.isnan(evo1).any():
                        continue

                    for j, evo2 in enumerate(tqdm(evolutions, leave=False)):

                        # skip grid points missing data
                        if np.isnan(evo2).any():
                            continue

                        # compute pearson's r and p-value and put in matrix
                        corr_coef, pval = pearsonr(evo1, evo2)
                        corr_mat[i, j] = corr_coef
                        pval_mat[i, j] = pval

                # save correlation analysis results to file
                np.save(pval_mat_file, pval_mat)
                np.save(corr_mat_file, corr_mat)

            # mask grid point where correlation not statistically significant
            pval_mask = (pval_mat > 0.05)
            corr_mat = np.ma.masked_array(corr_mat, mask=pval_mask,
                                          fill_value=np.nan)

        else:
            # if p-values not required, compute correlation with numpy
            # this is quick enough that there's no need to save to file
            corr_mat = np.ma.masked_array(np.corrcoef(evolutions))

        return corr_mat

    def layout_plots(self):

        # create GridSpec object with 3 rows and 4 columns
        gs = GridSpec(3, 4,
                      width_ratios=[1, 1, 0.5, 0.5],
                      height_ratios=[1, 1, 0.5],
                      figure=self.fig)

        # assign each ax a cell (or more) on the grid
        self.main_ax.set_position(gs[0].get_position(self.fig))
        self.helper_ax.set_position(gs[1].get_position(self.fig))
        self.fcluster_thresh_textbox_ax.set_position(
            gs[2].get_position(self.fig))
        self.corr_ax.set_position(gs[4].get_position(self.fig))
        self.cluster_ax.set_position(gs[5].get_position(self.fig))
        self.linkage_method_radio_ax.set_position(gs[6].get_position(self.fig))
        self.fcluster_criterion_radio_ax.set_position(
            gs[7].get_position(self.fig)
        )
        self.evo_ax.set_position(gs[8:].get_position(self.fig))

        # self.fig.set_constrained_layout(True)

    def change_slice(self, dimension, amount):
        # override MultiSliceViewer's change_slice method
        # to also update the time slice locator line on the evo plot
        super().change_slice(dimension, amount)
        self.update_evo_line()

    def update_plots(self):
        # override CorrelationMatrixViewer's update_plots  method
        # to also update the evo plot
        super().update_plots()
        self.update_evo_plot()

    def update_evo_line(self):
        # remove previous time slice locator line if exists
        try:
            for handle in self.evo_line:
                handle.remove()
        except AttributeError:
            pass

        # put new evo line based on current time slice
        try:
            evo_line_x = [self.index[0], self.index[0]]
            evo_line_y = [np.min(self.current_evo),
                          np.max(self.current_evo)]

            self.evo_line = self.evo_ax.plot(
                evo_line_x,
                evo_line_y,
                color='black'
            )
        except AttributeError:
            pass

    def update_evo_plot(self):
        # clear the evolution plot to draw R-age over time for new location
        self.evo_ax.clear()

        # unpack xy coords where user last clicked
        x_pos, y_pos = self.corr_loc

        # update attribute for future reference
        self.current_evo = self.surface_slices[:, y_pos, x_pos]

        # plot the new time series
        self.evo_ax.plot(range(self.surface_slices.shape[0]),
                         self.current_evo)


class DendrogramViewer(FclusterViewer):

    def __init__(self, corr_mat, map_shape, cmap='rainbow',
                 title='Dendrogram'):
        self.fig = plt.figure()
        self.fig.suptitle(title)
        self.cmap = cmap
        CorrelationMapperBase.__init__(self, corr_mat, map_shape, fig=self.fig)
        self.init_widgets()

        # with the above can now perform correlation clustering
        # and initialise the correlation map
        self.init_cluster_ax()
        self.init_dendro_ax()

        # call a first update to prepare the plots for viewing
        self.update_plots()

        plt.show()

    def init_dendro_ax(self):
        # add another axes for the dendrogram
        # i'm cheating and hardcoded the parent class to work with this layout
        # a gridspec (see CorrelationViewer) to override parent layout
        # would be preferable
        self.dendro_ax = self.fig.add_subplot(224)
        self.dendro_cid = self.fig.canvas.mpl_connect("button_press_event",
                                                      self.dendro_ax_on_click)

    def update_plots(self):
        hierarchy, labels_shaped = self.corr_cluster()

        cmap = get_cmap(self.cmap)
        labels_min = np.nanmin(labels_shaped)
        labels_max = np.nanmax(labels_shaped)
        norm = Normalize(vmin=labels_min, vmax=labels_max)
        colors = cmap(norm(np.arange(labels_max)+1))
        colors = [rgb2hex(c) for c in colors]
        set_link_color_palette(colors)

        self.update_dendrogram(hierarchy)
        self.update_cluster_map(labels_shaped)
        self.fig.canvas.draw()

    def update_dendrogram(self, hierarchy):
        self.dendro_ax.cla()
        self.dendro = dendrogram(hierarchy,
                                 color_threshold=self.fcluster_thresh,
                                 no_labels=True,
                                 ax=self.dendro_ax)
        self.dendro_ax.axhline(y=self.fcluster_thresh)
        self.dendro_ax.set_ylabel("cophenetic distance")
        self.dendro_ax.set_facecolor("lightgrey")

    def dendro_ax_on_click(self, event):
        if not event.inaxes == self.dendro_ax:
            return
        thresh = round(event.ydata, 3)
        self.fcluster_thresh = thresh
        self.fcluster_thresh_textbox.set_val(thresh)
        self.update_plots()
