import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats.stats import pearsonr
from slice_viewer import MultiSliceViewer
from tqdm import tqdm


class CorrelationViewer(MultiSliceViewer):

    def __init__(self, volume, title, colorbar=True, legend=False,
                 cmap='rainbow', corr_mat_file='corr_mat.npy',
                 pval_mat_file='pval_mat.npy', pvalues=True):

        print("[i] Initialising CorrelationViewer")
        try:
            # assuming volume is 4D of shape (t, z, y,x), take surface slice
            t, z, y, x = volume.shape
            self.surface_slice = volume[:, 0, :, :]
        except ValueError:
            # assumption of 4D was wrong -> volume is already a surface slice
            t, y, x = volume.shape
            self.surface_slice = volume

        # each row in evolutions is the R-age time series for a grid point
        evolutions = np.reshape(self.surface_slice, [t, x*y]).T

        # try reading correlation and p-value matrices from file
        if pvalues:
            try:
                pval_mat = np.load(pval_mat_file)
                print(f"[i] Read in {pval_mat_file}")
                corr_mat = np.load(corr_mat_file)
                print(f"[i] Read in {corr_mat_file}")
            # compute correlation now if not read from file
            except FileNotFoundError:
                # for p-values use pearson's r
                # intialise empty matrices to hold correlation coef and p-value
                pval_mat = np.empty([y*x, y*x])
                corr_mat = np.empty([y*x, y*x])
                # run Pearson's r for every possible pair of grid points
                for i, evo1 in enumerate(tqdm(evolutions,
                                              desc="[i] Computing Pearson's r \
                                              and p-values: ")):
                    # skip grid points missing data
                    if np.isnan(evo1).any():
                        continue
                    for j, evo2 in enumerate(
                        tqdm(evolutions, leave=False)
                    ):
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
            corr_mat = np.ma.masked_array(corr_mat, mask=pval_mask)
        else:
            # if p-values not required, compute correlation with numpy
            # this is quick enough that there's no need to save to file
            corr_mat = np.corrcoef(evolutions)

        # some handy attributes
        self.time_steps, self.n_cols, self.n_rows = t, y, x

        # save correlation matrix as attribute and run clustering
        self.corr_mat = corr_mat
        labels_shaped = self.corr_cluster()


        # call the init method of the MultiSliceViewer parent
        super().__init__(volume, title, colorbar=colorbar, legend=legend,
                         cmap=cmap)

        # listen for click events
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                     self.process_click)

        # map out clusters
        self.cluster_ax_image = self.cluster_ax.imshow(
            labels_shaped,
            origin='lower',
            cmap=cmap
        )

        # initialise correlation map with a gradient that
        # spans the range of r values in the corelation matrix
        dummy_corr_map = np.reshape(np.linspace(
                start=np.nanmin(self.corr_mat),
                stop=np.nanmax(self.corr_mat),
                num=x*y,
                endpoint=True
            ), [x, y])

        self.corr_ax_image = self.corr_ax.imshow(
            dummy_corr_map,
            origin='lower',
            cmap=cmap
        )

        # put a colorbar on the correlation map
        self.fig.colorbar(self.corr_ax_image, ax=self.corr_ax)

    def init_plots(self):
        # separate this call to plt.subplots for easy override in children
        # in this case we want 4 plots (2x2+1)
        self.fig = plt.figure()
        gs = GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 0.5],
                      figure=self.fig)
        self.main_ax = self.fig.add_subplot(gs[0])
        self.helper_ax = self.fig.add_subplot(gs[1])
        self.corr_ax = self.fig.add_subplot(gs[2])
        self.cluster_ax = self.fig.add_subplot(gs[3])
        self.evo_ax = self.fig.add_subplot(gs[4:])

    def process_click(self, event):
        # ignore clicks outside the plots
        if not event.inaxes:
            return

        # get position of click
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)

        # save last clicked point
        self.corr_loc = [x_pos, y_pos]

        # translate x-y coords of click to index of grid point in flat array
        flat_index = y_pos * self.n_rows + x_pos

        # clear the evolution plot and draw R-age over time for new location
        self.evo_ax.clear()
        self.current_evo = self.surface_slice[:, y_pos, x_pos]
        self.evo_ax.plot(range(self.surface_slice.shape[0]),
                         self.current_evo)

        # get the row in the correlation map corresponding
        # to the clicked grid point
        corr_array = self.corr_mat[flat_index]

        # shape this row into a 2D array and plot
        corr_map = np.reshape(corr_array,
                              [self.n_rows, self.n_cols]
                              )
        self.corr_ax_image.set_data(corr_map)

        # put a star on the location clicked by the user
        self.update_corr_loc_marker()

        # update figure
        self.fig.canvas.draw()

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
            self.helper_point = self.helper_ax.plot(
                x_pos,
                y_pos,
                color='black',
                marker='*'
            )
        except AttributeError:
            pass

    def change_slice(self, dimension, amount):
        super().change_slice(dimension, amount)
        self.update_evo_line()

    def update_evo_line(self):
        try:
            for handle in self.evo_line:
                handle.remove()
        except AttributeError:
            pass
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

    def corr_cluster(self, linkage_method='complete', fcluster_thresh=0.4,
                     fcluster_criterion='distance'):

        # correlation clustering
        # convert correlation matrix to pandas dataframe to drop nan rows/cols
        df = pd.DataFrame(self.corr_mat, index=None, columns=None)
        droppedna = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # form dataframe back into a correlation matrix (without nans)
        corr = np.array(droppedna)
        corr = np.reshape(corr, droppedna.shape)

        # corrections to reduce floating point errors
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)

        # convert the correlation coefficients (higher is closer)
        # into distances (lower is closer)
        dissimilarity = 1 - corr

        # dissimilarity matrix needs to be in this form for hierarchical
        # clustering
        square = squareform(dissimilarity)

        # perform hierarchical clustering
        hierarchy = linkage(square, method=linkage_method)

        # flatten the hierarchy into usable clusters
        # get the cluster label assigned to each grid point as a flat array
        labels = fcluster(hierarchy,
                          fcluster_thresh,
                          criterion=fcluster_criterion
                          )

        # put the labels into the whole dataframe (skipping nan rows/columns)
        df.loc[
                df.index.isin(droppedna.index),
                'labels'
                ] = labels

        # get the labels back as a flat array, now including nans
        labels_flat = np.ma.masked_array(df['labels'])

        # shape the label back into a 2D array for plotting clusters on a map
        labels_shaped = np.reshape(labels_flat, (self.n_rows, self.n_cols))

        return labels_shaped
