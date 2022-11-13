import matplotlib.pyplot as plt
import numpy as np

from slice_viewer import MultiSliceViewer


class CorrelationViewer(MultiSliceViewer):

    def __init__(self, volume, corr_mat, title, colorbar=True, legend=False,
                 cmap='rainbow'):

        self.corr_mat = corr_mat

        self.time_steps, self.n_cols, self.n_rows = volume.shape

        self.corr_loc = [self.n_rows/2, self.n_cols/2]

        super().__init__(volume, title, colorbar=colorbar, legend=legend,
                         cmap=cmap)

        self.click_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                     self.process_click)
        dummy_corr_map = np.reshape(np.linspace(
                start=np.nanmin(corr_mat),
                stop=np.nanmax(corr_mat),
                num=self.n_rows*self.n_cols,
                endpoint=True
            ), [self.n_rows, self.n_cols])

        self.corr_ax_image = self.corr_ax.imshow(
            dummy_corr_map,
            origin='lower',
            cmap='cividis'
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
        self.evo_ax.plot(range(self.volume.shape[0]),
                         self.volume[:, 0, y_pos, x_pos])

        flat_index = y_pos * self.n_rows + x_pos
        # print(f'flat index: {flat_index}')
        corr_array = self.corr_mat[flat_index]
        corr_map = np.reshape(corr_array,
                              [self.n_rows, self.n_cols]
                              )
        self.corr_ax_image.set_data(corr_map)
        self.update_helper_plot()
        self.fig.canvas.draw()

    def update_helper_plot(self):
        try:
            self.helper_ax.lines.pop(0)
        except IndexError:
            pass
        x_pos, y_pos = self.corr_loc
        self.helper_point = self.helper_ax.plot(
            x_pos,
            y_pos,
            color='black',
            marker='*'
        )
