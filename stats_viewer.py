import matplotlib.pyplot as plt
import numpy as np

from scipy.stats.stats import pearsonr
from slice_viewer import MultiSliceViewer
from tqdm import tqdm


class CorrelationViewer(MultiSliceViewer):

    def __init__(self, volume, title, colorbar=True, legend=False,
                 cmap='rainbow', corr_mat_file='corr_mat.npy',
                 pval_mat_file='pval_mat.npy', pvalues=False):

        print("[i] Initialising CorrelationViewer")
        try:
            # assuming volume is 4D of shape (t, z, y,x), take surface slice
            t, z, y, x = volume.shape
            self.surface_slice = volume[:, 0, :, :]
        except ValueError:
            # assumption of 4D was wrong -> volume is already a surface slice
            t, y, x = volume.shape
            self.surface_slice = volume

        # each row is an array representing the evolution of a grid point
        evolutions = np.reshape(self.surface_slice, [t, x*y]).T

        # quick numpy correlation analysis, no p-values

        # try reading correlation and p-value matrices from file
        try:
            corr_mat = np.load(corr_mat_file)
            print(f"[i] Read in {corr_mat_file}")
            if pvalues:
                pval_mat = np.load(pval_mat_file)
                print(f"[i] Read in {pval_mat_file}")
        # compute correlation now if not read from file
        except FileNotFoundError:
            if not pvalues:
                # if p-values not required, compute correlation with numpy
                corr_mat = np.corrcoef(evolutions)
            else:
                # for p-values use pearson's r
                # intialise empty matrices to hold correlation coef and p-value
                pval_mat = np.empty([y*x, y*x])
                corr_mat = np.empty([y*x, y*x])
                # run Pearson's r for every possible pair of grid points
                for i, evo1 in enumerate(tqdm(evolutions)):
                    # skip grid points missing data
                    if np.isnan(evo1).any():
                        continue
                    for j, evo2 in enumerate(tqdm(evolutions, leave=False)):
                        if np.isnan(evo2).any():
                            continue
                        # compute pearson's r and p-value and put in matrix
                        corr_coef, pval = pearsonr(evo1, evo2)
                        corr_mat[i, j] = corr_coef
                        pval_mat[i, j] = pval
                # save correlation analysis results to file
                np.save(pval_mat_file, pval_mat)
            np.save(corr_mat_file, corr_mat)

        if pvalues:
            # mask grid point where correlation not statistically significant
            pval_mask = (pval_mat > 0.05)
            corr_mat = np.ma.masked_array(corr_mat, mask=pval_mask)

        # initialise some class attributes
        self.corr_mat = corr_mat
        self.time_steps, self.n_cols, self.n_rows = t, y, x

        # call the init method of the MultiSliceViewer parent
        super().__init__(volume, title, colorbar=colorbar, legend=legend,
                         cmap=cmap)

        # listen for click events
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                     self.process_click)

        # initialise correlation map with a gradient that
        # spans the range of r values in the corelation matrix
        dummy_corr_map = np.reshape(np.linspace(
                start=np.nanmin(self.corr_mat),
                stop=np.nanmax(self.corr_mat),
                num=self.n_rows*self.n_cols,
                endpoint=True
            ), [self.n_rows, self.n_cols])

        self.corr_ax_image = self.corr_ax.imshow(
            dummy_corr_map,
            origin='lower',
            cmap='rainbow'
        )

        # put a colorbar on the correlation map
        self.fig.colorbar(self.corr_ax_image, ax=self.corr_ax)

    def init_plots(self):
        # separate this call to plt.subplots for easy override in children
        # in this case we want 4 plots (2x2)
        self.fig, ((self.main_ax, self.helper_ax),
                   (self.evo_ax, self.corr_ax)) = plt.subplots(2, 2)

    def process_click(self, event):
        # ignore clicks outside the plots
        if not event.inaxes:
            return

        # get position of click
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)

        # save last clicked point
        self.corr_loc = [x_pos, y_pos]

        # clear the evolution plot and draw R-age over time for new location
        self.evo_ax.clear()
        self.evo_ax.plot(range(self.surface_slice.shape[0]),
                         self.surface_slice[:, y_pos, x_pos])

        # translate x-y coords of click to index of grid point in flat array
        flat_index = y_pos * self.n_rows + x_pos

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
