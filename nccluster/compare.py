import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import xarray as xr
import xesmf as xe
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, Greys, ScalarMappable
from matplotlib_venn import venn2
from nccluster.ts import TimeSeriesWorkflowBase
from nccluster.utils import make_subclusters_map, locate_medoids, ts_from_locs
from numpy.linalg import norm
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
            # 2D array that is True where the left and right maps disagree
            overlap_mask = np.not_equal(self.labels_left.values,
                                        self.labels_right.values)
            overlap_mask[np.isnan(self.labels_left.values)] = False
            overlap_mask[np.isnan(self.labels_right.values)] = False
        except ValueError:
            print("[!] Can't overlap maps with different sizes! Try regrid.")

        # initialise the figure and axes and define hatches pattern

        plt.rcParams['figure.constrained_layout.use'] = True
        plt.figure()
        self.left_map = plt.subplot(221)
        self.right_map = plt.subplot(222)

        hatches = ["", "/\\/\\/\\/\\"]

        # show left map and hatch based on overlap mask
        self.left_map.imshow(self.labels_left.values,
                             origin='lower', cmap=self.cmap)
        self.left_map.contourf(overlap_mask, 1, hatches=hatches, alpha=0)

        # show right map and hatch based on overlap mask
        self.right_map.imshow(self.labels_right.values,
                              origin='lower', cmap=self.cmap)
        self.right_map.contourf(overlap_mask, 1, hatches=hatches, alpha=0)

        n_clusters = self.__get_n_clusters()
        for clust in range(n_clusters):
            ax = plt.subplot(2, n_clusters, n_clusters + clust + 1)
            self.plot_venn(clust, ax)

        equal = np.equal(self.labels_left.values, self.labels_right.values)
        n_equal = equal.sum()
        notnan = self.labels_left.values[~np.isnan(self.labels_left.values)]
        n_notnan = notnan.size
        overlap = int(n_equal) / n_notnan
        title = f'Total label overlap: {overlap:.0%}'
        plt.suptitle(title, fontsize='14')

        self.left_map.set_title(self.labels_left.long_name)
        self.right_map.set_title(self.labels_right.long_name)

        # show figure
        plt.show()

    def plot_venn(self, label, ax):

        # get the color corresponding to the label
        label_color = self.cmap(self.norm(label))

        # get the sets of flat indices with this label
        (idx,) = np.where(self.labels_left.values.flatten() == label)
        left_set = set(idx)
        (idx,) = np.where(self.labels_right.values.flatten() == label)
        right_set = set(idx)
        total = len(right_set.union(left_set))
        # create venn diagram based on the sets corresponding to the label
        venn = venn2(subsets=[left_set, right_set],
                     subset_label_formatter=lambda x: f"{(x/total):1.0%}",
                     set_colors=[label_color, label_color],
                     set_labels=["", ""],
                     ax=ax)
        # number the plot
        ax.text(0.5, 0, int(label+1),
                horizontalalignment='center',
                verticalalignment='center',
                rotation='horizontal',
                transform=ax.transAxes)

        # use black edges on the venn diagram
        try:
            for patch in venn.patches:
                patch.set(edgecolor='black')
        except AttributeError:
            pass

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

    def __get_n_clusters(self, left=True):
        labels = self.labels_left if left else self.labels_right
        return int(np.nanmax(labels.values)) + 1

    def match_labels(self):
        # flatten labels into 1D array
        labels_left_flat = self.labels_left.values.flatten()
        labels_right_flat = self.labels_right.values.flatten()

        # the labels are 0 to n-1
        n_clusters = self.__get_n_clusters()
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

    def __init__(self, config_path, config_original, labels_savefile):
        # initialise workflows (letting them compute surface ocean R-ages)
        wf = TimeSeriesWorkflowBase(config_path)
        wf_orig = TimeSeriesWorkflowBase(config_original)

        # load in the subcluster assignments
        labels_ds = nc.DataSet(labels_savefile)

        # use nearest neighbor (not interpolate!) to regrid integer labels
        labels_ds.regrid(wf._ds, method="nn")

        # also regrid the original timeseries from which labels extracted
        wf_orig.regrid_to_ds(wf._ds)

        # shorthands for frequently used arrays
        self.R_target = wf.ds_var_to_array('R_age')[:, 0, :, :]
        self.R_target_df = wf._make_df('R_age')
        self.avg_R = np.nanmean(self.R_target, axis=(-1, -2))
        self.labels = labels_ds.to_xarray()['labels'].values
        self.sublabels = labels_ds.to_xarray()['sublabels'].values

        # locate the subcluster medoids in the original time series
        age_array = wf_orig.ds_var_to_array('R_age')[:, 0]
        self.locations_dict = locate_medoids(
            self.labels, self.sublabels, age_array)

        # note the time series corresponding to medoids in target dataset
        self.centers_dict = ts_from_locs(self.locations_dict,
                                         self.R_target)
        self.map_all_cosines()

        # initialise main figure
        self.fig = plt.figure()

        # plot the subclusters in greyscale on map, continents in beige
        cmap = Greys
        cmap.set_bad('tan')
        self.map_ax = self.fig.add_subplot(121)
        self.map_ax.set_title('Subcluster map')
        self.map_ax.imshow(make_subclusters_map(self.labels, self.sublabels),
                           origin='lower', cmap=cmap)

        # location of colorbar to the right of subcluster map
        self.cos_cax = self.map_ax.inset_axes([1.04, 0, 0.05, 1])

        # listen for clicks on the subcluster map
        self.__cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.__process_click)

        # init the histogram plot
        self.hist_ax = self.fig.add_subplot(222)
        self.ts_ax = self.fig.add_subplot(224)

        # show the figure
        plt.tight_layout()
        plt.show()

    def __get_mask(self, label, sublabel=None):
        # construct a condition to extract the subcluster data points
        label_match = self.labels == label
        if sublabel:
            sublabel_match = self.sublabels == sublabel
            mask = ~(label_match & sublabel_match)
        else:
            mask = ~label_match
        return mask

    def __put_cosines_in_df(self, df, mask, benchmark):

        # flatten mask to match up with dataframe rows
        mask = mask.flatten()

        # extract the rows in the R dataframe corresponding to subcluster
        subclust_df = df.loc[~mask].copy()

        # drop any pre-existing cosines column to get only the time series
        try:
            subclust_df = subclust_df.drop(columns='cosine')
        except KeyError:
            pass

        # convert to array of time series
        subclust_tss = np.array(subclust_df)

        # column vectors
        A = subclust_tss
        B = benchmark

        # for each subcluster member, compute cosine similarity to center
        cosines = np.dot(A, B)/(norm(A, axis=1)*norm(B))

        # put the cosines as a new column on the R dataframe
        df.loc[df.index.isin(subclust_df.index), 'cosine'] = cosines

    def map_all_cosines(self):

        fig, axes = plt.subplots(nrows=1, ncols=3)
        cax = axes[1].inset_axes([-1, -0.5, 3, 0.1])
        titles = ['vs. global surface mean',
                  'vs. respective cluster medoid',
                  'vs. respective subcluster medoid']

        dfs = [self.R_target_df.copy() for i in range(3)]

        y = self.R_target.shape[1]
        x = self.R_target.shape[2]

        # first map compared to global average
        benchmark = self.avg_R
        mask = np.isnan(self.R_target[0])
        self.__put_cosines_in_df(dfs[0], mask=mask, benchmark=benchmark)

        # second map compared to cluster medoid location
        for label, center in enumerate(self.centers_dict['clusters']):
            mask = self.__get_mask(label=label)
            benchmark = center
            self.__put_cosines_in_df(dfs[1], mask=mask, benchmark=benchmark)

        # third map compared to subcluster medoid location
        for label, centers in enumerate(self.centers_dict['subclusters']):
            for sublabel, center in enumerate(centers):
                mask = self.__get_mask(label=label, sublabel=sublabel)
                benchmark = center
                self.__put_cosines_in_df(dfs[2], mask=mask,
                                         benchmark=benchmark)

        # convert to array and reshape into 2D map
        cosines_flat = [np.ma.masked_array(df['cosine']) for df in dfs]
        cosines_shaped = [np.reshape(flat, (y, x)) for flat in cosines_flat]

        norm = Normalize(vmin=np.nanmin(np.array(cosines_shaped)), vmax=1)
        cmap = 'cividis'
        mappable = ScalarMappable(norm=norm, cmap=cmap)

        for ax, data, title in zip(axes, cosines_shaped, titles):
            ax.imshow(data, origin='lower', cmap=cmap, norm=norm)
            ax.set_title(title)

        # add colorbar for the similarity levels
        fig.colorbar(mappable, ax=ax, cax=cax, orientation='horizontal')
        plt.show()

    def __plot_subclust_cosines(self, mask, subclust_center):

        # note the mask shape before flattening
        y, x = mask.shape

        # copy the time series of the target dataset
        df = self.R_target_df.copy()
        self.__put_cosines_in_df(df, mask, subclust_center)

        # convert to array and reshape into 2D map
        cosines_flat = np.ma.masked_array(df['cosine'])
        cosines_shaped = np.reshape(cosines_flat, (y, x))

        # remove previous overlay
        try:
            for handle in self.subclust_overlay.collections:
                handle.remove()
        except AttributeError:
            pass

        # color in the clicked subcluster
        self.subclust_overlay = self.map_ax.contourf(cosines_shaped,
                                                     cmap='coolwarm')

        # add colorbar for the similarity levels
        self.fig.colorbar(self.subclust_overlay, ax=self.map_ax,
                          cax=self.cos_cax, cmap='coolwarm')

        # remove previous marker
        try:
            for handle in self.marker:
                handle.remove()
        except AttributeError:
            pass

        # put a star on the grid point with highest similarity
        idx_flat = np.nanargmax(cosines_shaped)
        marker_x, marker_y = np.unravel_index(idx_flat, cosines_shaped.shape)
        self.marker = self.map_ax.plot(marker_y, marker_x,
                                       color='magenta',
                                       marker="*")

        return

    def __process_click(self, event):
        # ignore clicks outside subcluster map
        if event.inaxes != self.map_ax:
            return

        # unpack the click position
        x_pos = int(event.xdata)
        y_pos = int(event.ydata)

        # update attributes based on clicked data point
        label = int(self.labels[y_pos, x_pos])
        sublabel = int(self.sublabels[y_pos, x_pos])
        mask = self.__get_mask(label, sublabel)

        # extract and plot the (sub)cluster center
        cluster_center = self.centers_dict['clusters'][label]
        subclust_center = self.centers_dict['subclusters'][label][sublabel]

        # re-compute the density plots
        self.__density_plots(mask, cluster_center, subclust_center)

        # identify the subcluster member that best approximates the centroid
        self.__plot_subclust_cosines(mask, subclust_center)

        # update figure
        self.fig.canvas.draw()

    def to_density_function(self, data, benchmark):
        t, y, x = data.shape
        diff = data - benchmark.reshape(t, 1, 1).repeat(y, 1).repeat(x, 2)
        diff_not_masked = diff[~diff.mask]
        diff_abs = np.abs(diff_not_masked)
        diff_notnan = diff_abs[~np.isnan(diff_abs)]
        density = gaussian_kde(diff_notnan)

        return density

    def __density_plots(self, mask, cluster_center, subclust_center):

        # clear the time series plot
        self.ts_ax.cla()

        # get a handle on array shapes
        t, y, x = self.R_target.shape

        # repeat the masking condition for every time slice
        mask = np.repeat(np.array([mask]), repeats=t, axis=0)

        # mask R array to show current subcluster
        intrasub = np.ma.masked_where(mask, self.R_target)

        # plot time series for each subcluster member
        subclust_tss = intrasub[~intrasub.mask]
        for ts in np.reshape(subclust_tss, (t, -1)).T:
            self.ts_ax.plot(ts, color='grey', linewidth=0.5)

        # plot the benchmarks
        subclust_line, = self.ts_ax.plot(subclust_center)
        cluster_line, = self.ts_ax.plot(cluster_center)
        avg_line, = self.ts_ax.plot(self.avg_R)

        # add labels
        self.ts_ax.legend([subclust_line, cluster_line, avg_line],
                          ['subcluster center R-age',
                           'cluster center R-age',
                           'surface mean R-age'])

        # time series to compare subcluster time series to
        benchmarks = [subclust_center, cluster_center, self.avg_R]

        # get the density functions of the differences between
        # subcluster members and benchmarks
        densities = [self.to_density_function(intrasub, benchmark)
                     for benchmark in benchmarks]

        span = range(0, 750)
        self.hist_ax.cla()
        for dens in densities:
            self.hist_ax.plot(span, dens(span), linewidth=2, alpha=0.5)
        self.hist_ax.legend(['DR_subclust_center',
                             'DR_clust_center',
                             'DR_global_average'])
