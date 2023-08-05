import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize


class MultiSliceViewer:
    """A class that takes in a data array with one time dimension and 2 or 3
    spatial dimensions, and allows the user  to visualize the data as slices,
    navigating with the arrow keys (left-right for time, up-down for z axis).
    Pressing the x, y, z keys will change the cross-section to be perpendicular
    to the selected axis.
    """

    def __init__(self, volume, title=None, colorbar=True, legend=False,
                 cmap='rainbow', fig=None):
        # if data has only 3 dimensions; assume it is missing the depth axis
        # and reshape into 4D array with single depth level
        if len(volume.shape) == 3:
            t, y, x = volume.shape
            volume = np.reshape(volume, (t, 1, y, x))

        # save the data volume as class attribute
        self.volume = volume

        # start at time 0, depth 0
        self.index = [0, 0]

        # note the colormap for subsequent plotting
        self.cmap = cmap

        # take note of the surface slice at each time step.
        # we will need this when changing perspective because as we rotate
        # the axes of the data volume, we lose track of where the surface is
        self.surface_slices = volume[:, 0]

        # ensure colors span whole data range, not just initial slice
        self.norm = Normalize(
            vmin=np.nanmin(self.volume), vmax=np.nanmax(self.volume)
        )

        # the viewer can create a new figure or use an already existing figure
        self.fig = fig if fig else plt.figure()

        # initialise explorer plot and locator plot
        self.init_explorer_ax(colorbar, legend)
        self.init_locator_ax()

        # initialise view perpendicular to z
        self.set_view('z')

        # put a title at the top of the whole figure
        if title is not None:
            self.fig.suptitle(title)

        # tell figure to wait for key press events,
        # saving the callback id (cid) so it doesn't get garbage collected
        self.keypress_cid = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.process_key)

    def init_explorer_ax(self, colorbar, legend):
        # create the explorer plot Axes on the left
        self.explorer_ax = self.fig.add_subplot(121)

        # show the volume slice specified by self.index;
        # the index takes the form [time, depth] right after initialization.
        self.explorer_ax_image = self.explorer_ax.imshow(
            self.volume[
                self.index[0],
                self.index[1]
            ],
            # remember we've set the colormap and norm based on the input data
            cmap=self.cmap,
            norm=self.norm,
            # prevents map showing upside down with our axes convention
            origin='lower'
        )

        # optional colorbar
        if colorbar:
            # create inset axes to put the colorbar in.
            cax = self.explorer_ax.inset_axes([1.04, 0, 0.05, 1])
            # by specifying the explorer_ax_image as ScalarMappable to pass
            # to colorbar, we ensure the colorbar range matches the data
            self.fig.colorbar(self.explorer_ax_image,
                              ax=self.explorer_ax, cax=cax)

        # optionally put on a legend (use with categorical data only!)
        if legend:
            # get the unique categories in our array
            values = np.unique(self.volume.ravel())

            # get a shorthand reference to the image on the explorer plot
            im = self.explorer_ax_image

            # transform the values into colors according to colormap
            colors = [im.cmap(im.norm(value)) for value in values]

            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i],
                                      label="{l}".format(l=values[i]))
                       for i in range(len(values))]
            # put those patches as legend-handles into the legend
            self.explorer_ax.legend(handles=patches,
                                    bbox_to_anchor=(-0.25, 0.5),
                                    loc="center left")

    def init_locator_ax(self):
        # create the locator plot on the right
        self.locator_ax = self.fig.add_subplot(122)

        # initialise the surface slice at the current time step
        # as the locator image
        self.locator_ax_image = self.locator_ax.imshow(
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
        self.explorer_ax_image.origin = origin_dict[view]

        # assign the new view to the figure to be used when updating info text
        self.view = view

        # reset space slice to zero
        self.change_slice(1, -self.index[1])

    def process_key(self, event):
        # define actions to execute when certain keys are pressed.
        # arrow keys to navigate time (left/right) and space (up/down)
        if event.key == 'left':
            self.change_slice(0, -1)
        if event.key == 'right':
            self.change_slice(0, 1)
        if event.key == 'up':
            self.change_slice(1, 1)
        if event.key == 'down':
            self.change_slice(1, -1)
        # keys x, y, z to change cross-section direction
        if event.key == 'x':
            self.set_view('x')
        if event.key == 'y':
            self.set_view('y')
        if event.key == 'z':
            self.set_view('z')

        # update the plot
        self.fig.canvas.draw()

    def update_explorer_ax_title(self):
        # change the title of the explorer plot to reflect current time slice
        # and the space slice and the current view direction
        time_step, depth_step = self.index
        max_time_steps, max_depth_steps, _, _ = self.volume.shape
        self.explorer_ax.title.set_text(f"time: "
                                        f"{time_step+1}/{max_time_steps}; "
                                        f"{self.view}: "
                                        f"{depth_step+1}/{max_depth_steps}"
                                        )

    def change_slice(self, dimension, amount):
        # increment index (wrap around with modulo)
        self.index[dimension] += amount
        self.index[dimension] %= self.volume.shape[dimension]

        # set the current slice on the explorer plot based on the new index
        self.explorer_ax_image.set_data(
            self.volume[
                self.index[0],
                self.index[1]
            ]
        )

        # also update the surface view if time changed
        self.locator_ax_image.set_data(self.surface_slices[self.index[0]])

        # and the title on the explorer plot to reflect new time/depth step
        self.update_explorer_ax_title()

        # and the line on the locator plot to show new slice location
        self.update_slice_locator()

    def update_slice_locator(self):
        # remove previous line if exists
        try:
            for handle in self.locator_line:
                handle.remove()
        except (AttributeError, ValueError):
            pass

        # slice locator in current implementation only makes sense for x/y view
        if(self.view != 'z'):
            # figure out where to put the locator line
            locator_line_x = [self.index[1], self.index[1]]
            locator_line_y = [0, self.volume.shape[3]]

            # just a trick to make the line horizontal for y view
            if(self.view == 'y'):
                locator_line_x, locator_line_y = locator_line_y, locator_line_x

            # actually plot the slice locator line on the locator plot
            self.locator_line = self.locator_ax.plot(
                locator_line_x,
                locator_line_y,
                color='black'
            )
