import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class MultiSliceViewer:

    def __init__(self, volume, title, colorbar=True, legend=False,
                 cmap='rainbow'):
        # if data has only 3 dimensions; assume it is missing the depth axis
        # reshape into 4D array with single depth level
        if len(volume.shape) == 3:
            t, y, x = volume.shape
            volume = np.reshape(volume, (t, 1, y, x))

        # init class attributes
        self.volume = volume
        self.surface_slices = volume[:, 0]
        self.index = [0, 0]

        self.init_plots()

        self.main_ax_image = self.main_ax.imshow(
            volume[
                self.index[0],
                self.index[1]
            ],
            cmap=cmap,
        )

        self.helper_ax_image = self.helper_ax.imshow(
            self.surface_slices[self.index[0]],
            origin='lower',
            cmap=cmap,
        )

        # put a colorbar next to main plot if requested
        if colorbar:
            self.fig.colorbar(self.main_ax_image, ax=self.main_ax)

        # initialise view perpendicular to z
        self.set_view('z')

        # tell figure to wait for key events
        self.keypress_cid = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.process_key)

        # title specified in function call
        plt.suptitle(title)

        if legend:
            values = np.unique(volume.ravel())
            im = self.main_ax_image
            colors = [im.cmap(im.norm(value)) for value in values]
            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i],
                                      label="{l}".format(l=values[i]))
                       for i in range(len(values))]
            # put those patched as legend-handles into the legend
            self.main_ax.legend(handles=patches,
                                bbox_to_anchor=(-0.25, 0.5),
                                loc="center left")

    def init_plots(self):
        # separate this call to plt.subplots for easy override in children
        self.fig, (self.main_ax, self.helper_ax) = plt.subplots(1, 2)

    def show(self):
        plt.show()

    def set_view(self, view):

        # if view not specified, default to z
        try:
            self.view
        except AttributeError:
            self.view = 'z'

        # define permutations between axes
        permutation_dict = {
            'x': [0, 2, 3, 1],
            'y': [0, 2, 1, 3],
            'z': [0, 1, 2, 3]
        }

        # move the axes to slice through the fist two (time and a space axis)
        self.volume = np.moveaxis(
            self.volume,
            permutation_dict[self.view],
            permutation_dict[view]
        )

        # make sure plot doesn't show upside down
        origin_dict = {
            'x': 'upper',
            'y': 'upper',
            'z': 'lower'
        }
        self.main_ax_image.origin = origin_dict[view]

        # assign the new view to the figure to be used when updating info text
        self.view = view

        # reset space slice to zero
        self.change_slice(1, -self.index[1])

    def process_key(self, event):
        """Define action to execute when certain keys are pressed."""
        # arrow key navigation
        if event.key == 'left':
            self.change_slice(0, -1)
        if event.key == 'right':
            self.change_slice(0, 1)
        if event.key == 'up':
            self.change_slice(1, 1)
        if event.key == 'down':
            self.change_slice(1, -1)
        if event.key == 'x':
            self.set_view('x')
        if event.key == 'y':
            self.set_view('y')
        if event.key == 'z':
            self.set_view('z')
        # update the plot
        self.fig.canvas.draw()

    def update_suptitle_text(self):
        time_step, depth_step = self.index
        max_time_steps, max_depth_steps, _, _ = self.volume.shape
        self.main_ax.title.set_text(f"time: "
                                    f"{time_step+1}/{max_time_steps}\n"
                                    f"{self.view}: "
                                    f"{depth_step+1}/{max_depth_steps}"
                                    )

    def change_slice(self, dimension, amount):
        # increment index (wrap around with modulo)
        self.index[dimension] += amount
        self.index[dimension] %= self.volume.shape[dimension]

        # set the slice to view based on the new index
        # self.main_ax.images[0].set_array(
        self.main_ax_image.set_data(
            self.volume[
                self.index[0],
                self.index[1]
            ]
        )

        # also update the surface view if time changed
        self.helper_ax_image.set_data(self.surface_slices[self.index[0]])

        self.update_suptitle_text()

        self.update_helper_plot()

    def update_helper_plot(self):
        try:
            self.helper_ax.lines.pop(0)
        except IndexError:
            pass

        if(self.view != 'z'):
            helper_line_x = [self.index[1], self.index[1]]
            helper_line_y = [0, self.volume.shape[3]]

            if(self.view == 'y'):
                helper_line_x, helper_line_y = helper_line_y, helper_line_x

            self.helper_line = self.helper_ax.plot(
                helper_line_x,
                helper_line_y,
                color='black'
            )


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
