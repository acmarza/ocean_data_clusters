import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import xarray as xr
import xesmf as xe

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, Greys
from matplotlib_venn import venn2
from matplotlib.widgets import Slider
from nccluster.workflows import dRWorkflow
from scipy.stats import gaussian_kde


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
        # and output coords of right map;
        # "nearest source to destination" algorithm avoids interpolation
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


class DdR_Histogram:

    def __init__(self, config_path1, config_path2, labels_savefile):
        # initialise workflows (letting them compute surface ocean dR)
        wf1 = dRWorkflow(config_path1)
        wf2 = dRWorkflow(config_path2)

        # load in the subcluster assignments
        labels_ds = nc.DataSet(labels_savefile)

        # regrid everything to second workflow's dataset
        wf1.regrid_to_ds(wf2._ds)

        # use nearest neighbor (not interpolate!) to regrid integer labels
        labels_ds.regrid(wf2._ds, method="nn")

        # initialise the attributes we'll need
        self.dR1 = wf1.ds_var_to_array('dR')
        self.dR2 = wf2.ds_var_to_array('dR')
        self.labels = labels_ds.to_xarray()['labels'].values
        self.sublabels = labels_ds.to_xarray()['sublabels'].values
        self.time1 = 0
        self.time2 = 0
        self.__set_DdR()

        # plot the subclustesr on a map
        cmap = Greys
        cmap.set_bad('tan')
        self.fig = plt.figure()
        self.map_ax = self.fig.add_subplot(221)
        self.map_ax.imshow(make_subclusters_map(self.labels, self.sublabels),
                           origin='lower', cmap=cmap)

        # listen for clicks on the subcluster map
        self.__cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.__process_click)

        # init the DdR map
        self.dR_ax = self.fig.add_subplot(223)
        self.dR_map = self.dR_ax.imshow(self.DdR, origin='lower')

        # init the histogram plot
        self.hist_ax = self.fig.add_subplot(224)

        # sliders for adjusting timeslices
        slider1_ax = self.fig.add_subplot(422)
        slider2_ax = self.fig.add_subplot(424)

        self.time1_slider = Slider(ax=slider1_ax,
                                   label='Dataset 1 time slice',
                                   valmin=0,
                                   valmax=self.dR1.shape[0]-1,
                                   valstep=1)

        self.time2_slider = Slider(ax=slider2_ax,
                                   label='Dataset 2 time slice',
                                   valmin=0,
                                   valmax=self.dR2.shape[0]-1,
                                   valstep=1)

        self.time1_slider.on_changed(self.__set_time1)
        self.time2_slider.on_changed(self.__set_time2)

        plt.show()

    def __set_time1(self, value=0):
        self.time1 = value
        self.__set_DdR()
        self.__hists()
        self.dR_map.set_data(self.DdR)

    def __set_time2(self, value=0):
        self.time2 = value
        self.__set_DdR()
        self.__hists()
        self.dR_map.set_data(self.DdR)

    def __process_click(self, event):
        if event.inaxes != self.map_ax:
            return
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)

        self.current_label = self.labels[y_pos, x_pos]
        self.current_sublabel = self.sublabels[y_pos, x_pos]

        self.__hists()

    def __set_DdR(self):
        self.DdR = self.dR1[self.time1, 0] - self.dR2[self.time2, 0]

    def __hists(self):

        # shorthand
        DdR = self.DdR

        # boolean array that are True for points in current (sub)cluster
        label_match = self.labels == self.current_label
        sublabel_match = self.sublabels == self.current_sublabel

        # note the extra ~ in front of the conditions
        # because the mask should be False where we show a point (True to mask)
        intrasub = np.ma.masked_where(~(label_match & sublabel_match), DdR)
        intra = np.ma.masked_where(~label_match, DdR)
        extra = np.ma.masked_where(label_match, DdR)

        # remove previous overlay
        try:
            for handle in self.subclust_overlay.collections:
                handle.remove()
        except AttributeError:
            pass
        # color in the clicked subcluster
        self.subclust_overlay = self.map_ax.contourf(~np.isnan(intrasub))

        # intrasubcluster average
        DdR_k = np.nanmean(intrasub)

        # remove nans, flatten and subtract subcluster average
        intrasub, intra, extra = list(map(
            lambda a: a[~np.isnan(a)].flatten() - DdR_k,
            [intrasub, intra, extra]
        ))

        # compute densities and plot
        densities = [gaussian_kde(data)
                     for data in [intrasub, intra, extra]]
        span = range(-750, 750)
        self.hist_ax.cla()
        for dens in densities:
            self.hist_ax.plot(span, dens(span), linewidth=2, alpha=0.5)
        self.hist_ax.axvline(0)
        self.hist_ax.legend(['intrasub', 'intra', 'extra'])
        self.fig.canvas.draw()


def make_subclusters_map(labels, sublabels):
    subclust_map = np.full_like(labels, np.nan)
    subclust_sizes = make_subclust_sizes(labels, sublabels)
    for (yi, xi) in np.argwhere(~np.isnan(labels)):
        label = labels[yi, xi]
        sublabel = sublabels[yi, xi]
        size = subclust_sizes[int(label)]
        subclust_map[yi, xi] = get_sublabel_colorval(label, sublabel, size)
    return subclust_map


def make_subclust_sizes(labels, sublabels):
    n_clusters = int(np.nanmax(labels)) + 1
    subclust_sizes = []
    for label in range(n_clusters):
        subclust_size = np.nanmax(sublabels[labels == label])
        subclust_sizes.append(subclust_size)
    return subclust_sizes


def get_sublabel_colorval(label, sublabel, subclust_size):
    interval = 0.5
    offset = sublabel * (interval / (subclust_size - 1))
    colorval = label - interval/2 + offset
    return colorval


def show_map(map, cmap='viridis'):
    plt.figure()
    plt.imshow(map, origin='lower', cmap=cmap)
    plt.show()
