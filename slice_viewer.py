import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize


class MultiSliceViewer:

    def __init__(self, volume, title="Viewer", colorbar=True, legend=False,
                 cmap='rainbow', fig=None):
        # if data has only 3 dimensions; assume it is missing the depth axis
        # reshape into 4D array with single depth level
        if len(volume.shape) == 3:
            t, y, x = volume.shape
            volume = np.reshape(volume, (t, 1, y, x))

        # init class attributes
        self.volume = volume
        self.surface_slices = volume[:, 0]
        self.index = [0, 0]
        self.cmap = cmap

        # debug
        self.norm = Normalize(vmin=np.nanmin(self.volume),
                         vmax=np.nanmax(self.volume))

        # the viewer can create a new figure or use an already existing figure
        self.fig = fig if fig else plt.figure()

        # initialise main plot and helper plot
        self.init_main_ax(colorbar, legend)
        self.init_helper_ax()

        # initialise view perpendicular to z
        self.set_view('z')

        # tell figure to wait for key events
        # save the callback id (cid) so it doesn't get garbage collected
        self.keypress_cid = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.process_key)
        # put a title at the top of the whole figure
        self.fig.suptitle(title)

    def init_main_ax(self, colorbar, legend):
        # create the main plot Axes
        self.main_ax = self.fig.add_subplot(121)

        # show the volume slice specified by self.index
        self.main_ax_image = self.main_ax.imshow(
            self.volume[
                self.index[0],
                self.index[1]
            ],
            cmap=self.cmap,
            norm=self.norm,
            origin='lower'
        )

        # optionally create inset axes to put the colorbar in
        # this stops the layout going haywire for more complex arrangements
        if colorbar:
            cax = self.main_ax.inset_axes([1.04, 0, 0.05, 1])
            self.fig.colorbar(self.main_ax_image, ax=self.main_ax, cax=cax)

        # optionally put on a legend (use with categorical data only!)
        if legend:
            # get the unique categories in our array
            values = np.unique(self.volume.ravel())

            # get a shorthand reference to the image on the main plot
            im = self.main_ax_image

            # transform the values into colors according to colormap
            colors = [im.cmap(im.norm(value)) for value in values]

            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i],
                                      label="{l}".format(l=values[i]))
                       for i in range(len(values))]
            # put those patches as legend-handles into the legend
            self.main_ax.legend(handles=patches,
                                bbox_to_anchor=(-0.25, 0.5),
                                loc="center left")

    def init_helper_ax(self):

        # create the helper plot
        self.helper_ax = self.fig.add_subplot(122)

        # initialise the surface slice as the helper image
        self.helper_ax_image = self.helper_ax.imshow(
            self.surface_slices[self.index[0]],
            origin='lower',
            cmap=self.cmap,
            norm=self.norm
        )

    def show(self):
        # convenience wrapper
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

        # shuffle the axes so that the first is time
        # and the second is the requested view
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
        # define action to execute when certain keys are pressed
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

    def update_main_ax_title(self):
        # change the title of the main plot to reflect the current time slice
        # and the space slice for the current view
        time_step, depth_step = self.index
        max_time_steps, max_depth_steps, _, _ = self.volume.shape
        self.main_ax.title.set_text(f"time: "
                                    f"{time_step+1}/{max_time_steps}; "
                                    f"{self.view}: "
                                    f"{depth_step+1}/{max_depth_steps}"
                                    )

    def change_slice(self, dimension, amount):
        # increment index (wrap around with modulo)
        self.index[dimension] += amount
        self.index[dimension] %= self.volume.shape[dimension]

        # set the current slice on the main plot based on the new index
        self.main_ax_image.set_data(
            self.volume[
                self.index[0],
                self.index[1]
            ]
        )

        # also update the surface view if time changed
        self.helper_ax_image.set_data(self.surface_slices[self.index[0]])

        # and the title on the main plot to reflect new time/depth step
        self.update_main_ax_title()

        # and the line on the helper plot to show new slice location
        self.update_slice_locator()

    def update_slice_locator(self):
        # remove previous line if exists
        try:
            for handle in self.helper_line:
                handle.remove()
        except (AttributeError, ValueError):
            pass

        # slice locator in current implementation only makes sense for x/y view
        if(self.view != 'z'):
            # figure out where to put the locator line
            helper_line_x = [self.index[1], self.index[1]]
            helper_line_y = [0, self.volume.shape[3]]

            # just a trick to make the line horizontal for y view
            if(self.view == 'y'):
                helper_line_x, helper_line_y = helper_line_y, helper_line_x

            # actually plot the slice locator line on the helper plot
            self.helper_line = self.helper_ax.plot(
                helper_line_x,
                helper_line_y,
                color='black'
            )
