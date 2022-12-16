import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib_venn import venn2


class ClusterMatcher:

    def __init__(self, palette='viridis'):
        self.cmap = get_cmap(palette)

    def __set_norm(self):
        # create a norm object for coloring purposes
        # scaling the range of labels in our data to the interval [0, 1]
        all_data = np.concatenate([self.labels_left.values.flatten(),
                                   self.labels_right.values.flatten()])
        self.norm = Normalize(vmin=np.nanmin(all_data),
                              vmax=np.nanmax(all_data))

    def labels_from_data_arrays(self, labels_left, labels_right):
        self.labels_left = labels_left
        self.labels_right = labels_right
        self.__set_norm()

    def labels_from_file(self, save_path_1, save_path_2):
        self.labels_left = xr.open_dataarray(save_path_1)
        self.labels_right = xr.open_dataarray(save_path_2)
        self.__set_norm()

    def save_labels(self, save_path_1, save_path_2):
        self.labels_left.to_netcdf(save_path_1)
        self.labels_right.to_netcdf(save_path_2)

    def compare_maps(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.labels_left.values, origin='lower')
        ax1.set_title(self.labels_left.long_name)
        ax2.imshow(self.labels_right.values, origin='lower')
        ax2.set_title(self.labels_right.long_name)
        plt.show()

    def overlap(self):
        try:
            # 2D array that is True where the left and right maps agree
            overlap_mask = np.equal(self.labels_left.values,
                                    self.labels_right.values)
        except ValueError:
            print("[!] Can't overlap maps with different sizes! Try regrid.")

        # initialise the figure and axes and define hatches pattern
        fig = plt.figure()
        self.ax1 = plt.subplot(221)
        self.ax2 = plt.subplot(222)

        hatches = ["", "/\\/\\/\\/\\"]

        # show left map and hatch based on overlap mask
        self.ax1.imshow(self.labels_left.values,
                        origin='lower', cmap=self.cmap)
        self.ax1.contourf(overlap_mask, 1, hatches=hatches, alpha=0)

        # show right map and hatch based on overlap mask
        self.ax2.imshow(self.labels_right.values,
                        origin='lower', cmap=self.cmap)
        self.ax2.contourf(overlap_mask, 1, hatches=hatches, alpha=0)

        # listen for clicks
        self.cid = fig.canvas.mpl_connect('button_press_event',
                                          self.__venn_clicked_label)

        # show figure
        plt.show()

    def __venn_clicked_label(self, event):
        # get click location
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)

        # set data to left or right map depending on where the mouse was
        if event.inaxes == self.ax1:
            data = self.labels_left.values
        if event.inaxes == self.ax2:
            data = self.labels_right.values

        # set the label from its index in the map data array
        label = data[y_pos, x_pos]

        # get the color corresponding to the label
        label_color = self.cmap(self.norm(label))

        # get the sets of flat indices with this label
        (idx,) = np.where(self.labels_left.values.flatten() == label)
        left_set = set(idx)
        (idx,) = np.where(self.labels_right.values.flatten() == label)
        right_set = set(idx)

        # get a reference to the venn diagram ax and clear previous drawing
        ax3 = plt.subplot(212)
        ax3.clear()

        # create new venn diagram based on the sets corresponding to the label
        venn = venn2(subsets=[left_set, right_set],
                     set_colors=[label_color, label_color],
                     ax=ax3)

        # use black edges on the venn diagram
        for patch in venn.patches:
            patch.set(edgecolor='black')

        # update the figure
        event.canvas.draw()

    def regrid_left_to_right(self):
        # create regridder object with the input coords of left map
        # and output coords of right map
        # nearest source to destination algorithm avoids interpolation
        # (since we want integer labels)
        regridder = xe.Regridder(self.labels_left,
                                 self.labels_right,
                                 'nearest_s2d'
                                 )
        # regridding deletes the attributes for some reason so save a copy
        attrs = self.labels_left.attrs
        # regrid the left map
        self.labels_left = regridder(self.labels_left)
        # put the attributes back in
        self.labels_left.attrs = attrs

    def match_labels(self):
        # flatten labels into 1D array
        labels_left_flat = self.labels_left.values.flatten()
        labels_right_flat = self.labels_right.values.flatten()

        # the labels are 0 to n-1
        n_clusters = int(np.nanmax(self.labels_left.values)) + 1
        if n_clusters != int(np.nanmax(self.labels_right.values)) + 1:
            print("[!] The maps have a different number of clusters")
            exit()

        # initialise empty arrays
        array_of_sets_left = []
        array_of_sets_right = []

        # for every cluster (label), get all the indices that have
        # the current label and put them in a set
        for cluster in range(0, n_clusters):
            (indices_1, ) = np.where(labels_left_flat == cluster)
            array_of_sets_left.append(set(indices_1))
            (indices_2, ) = np.where(labels_right_flat == cluster)
            array_of_sets_right.append(set(indices_2))

        # reorder lists to have biggest sets first
        array_of_sets_left.sort(key=lambda x: -len(x))
        array_of_sets_right.sort(key=lambda x: -len(x))

        # quantify overlap between pairs of sets between the two label versions
        pairings_scores = []
        for i, left_set in enumerate(array_of_sets_left):
            for j, right_set in enumerate(array_of_sets_right):
                union_size = len(left_set.union(right_set))
                intersection_size = len(left_set.intersection(right_set))
                overlap = intersection_size/union_size
                # print(f'{i} vs. {j}: {overlap:.2%}')
                pairings_scores.append((i, j, overlap))

        # start with the highest overlap scores
        pairings_scores.sort(reverse=True, key=lambda x: x[2])

        # prepare an empty array to hold the 1:1 matching between
        # labels on the left and right
        translation = np.full(n_clusters, np.nan)

        # put the pairings with the highest scores in the translation array
        # skipping when an entry already exists
        for pair in pairings_scores:
            left, right, score = pair
            if right not in translation and np.isnan(translation[left]):
                translation[left] = right

        # relabel the sets to match based on our translation function
        for i, left_set in enumerate(array_of_sets_left):
            for idx in left_set:
                labels_left_flat[idx] = translation[i]

        for i, right_set in enumerate(array_of_sets_right):
            for idx in right_set:
                labels_right_flat[idx] = i

        # reshape translated 1D arrays and update attributes
        shape = self.labels_left.values.shape
        self.labels_left.values = np.reshape(labels_left_flat, shape)
        self.labels_right.values = np.reshape(labels_right_flat, shape)
