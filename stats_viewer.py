import matplotlib.pyplot as plt
import numpy as np

from scipy.stats.stats import pearsonr
from slice_viewer import MultiSliceViewer
from tqdm import tqdm


class CorrelationViewer(MultiSliceViewer):

    def __init__(self, volume, title, colorbar=True, legend=False,
                 cmap='rainbow', corr_mat_file='corr_mat.npy',
                 pval_mat_file='pval_mat.npy'):

        # assumes volume is 3D
        try:
            t, z, y, x = volume.shape
            self.surface_slice = volume[:, 0, :, :]
        except Exception as e:
            print(e)
            t, y, x = volume.shape
            self.surface_slice = volume

        evolutions = np.reshape(self.surface_slice, [t, x*y]).T

        # self.corr_mat = np.corrcoef(evolutions)
        try:
            corr_mat = np.load(corr_mat_file)
            pval_mat = np.load(pval_mat_file)
        except FileNotFoundError:

            corr_mat = np.empty([y*x, y*x])
            pval_mat = np.empty([y*x, y*x])

            for i, evo1 in enumerate(tqdm(evolutions)):
                if np.isnan(evo1).any():
                    continue
                for j, evo2 in enumerate(tqdm(evolutions, leave=False)):
                    if np.isnan(evo2).any():
                        continue
                    corr_coef, pval = pearsonr(evo1, evo2)
                    corr_mat[i, j] = corr_coef
                    pval_mat[i, j] = pval
            np.save(corr_mat_file, corr_mat)
            np.save(pval_mat_file, pval_mat)

        # debug
        print(f"min: {np.min(pval_mat)}")
        print(f"max: {np.max(pval_mat)}")

        self.corr_mat = corr_mat
        self.pval_mat = pval_mat
        self.corr_loc = [x/2, y/2]
        self.time_steps, self.n_cols, self.n_rows = t, y, x

        super().__init__(volume, title, colorbar=colorbar, legend=legend,
                         cmap=cmap)

        self.click_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                     self.process_click)
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
        self.fig.colorbar(self.corr_ax_image, ax=self.corr_ax)

    def init_plots(self):
        # separate this call to plt.subplots for easy override in children
        self.fig, ((self.main_ax, self.helper_ax),
                   (self.evo_ax, self.corr_ax)) = plt.subplots(2, 2)

    def process_click(self, event):
        if not event.inaxes:
            return

        x_pos = int(event.xdata)
        y_pos = int(event.ydata)
        self.corr_loc = [x_pos, y_pos]
        # print(f'data coords {x_pos}:{y_pos}')
        self.evo_ax.clear()
        self.evo_ax.plot(range(self.surface_slice.shape[0]),
                         self.surface_slice[:, y_pos, x_pos])

        flat_index = y_pos * self.n_rows + x_pos
        # print(f'flat index: {flat_index}')
        corr_array = self.corr_mat[flat_index]
        corr_map = np.reshape(corr_array,
                              [self.n_rows, self.n_cols]
                              )
        self.corr_ax_image.set_data(corr_map)
        self.update_corr_loc_marker()
        self.fig.canvas.draw()

    def update_corr_loc_marker(self):

        try:
            for handle in self.helper_point:
                handle.remove()
        except AttributeError:
            pass
        x_pos, y_pos = self.corr_loc
        self.helper_point = self.helper_ax.plot(
            x_pos,
            y_pos,
            color='black',
            marker='*'
        )
