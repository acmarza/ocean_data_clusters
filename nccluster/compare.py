import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import xarray as xr
import xesmf as xe
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, Greys
from matplotlib_venn import venn2
from nccluster.workflows import dRWorkflow
from nccluster.utils import make_subclusters_map, construct_barycenters
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
            # 2D array that is True where the left and right maps agree
            overlap_mask = np.equal(self.labels_left.values,
                                    self.labels_right.values)
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
        plt.suptitle(title)

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

        # create venn diagram based on the sets corresponding to the label
        venn = venn2(subsets=[left_set, right_set],
                     set_colors=[label_color, label_color],
                     ax=ax)

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

    def __init__(self, config_path, labels_savefile):
        # initialise workflow (letting it compute surface ocean dR)
        wf = dRWorkflow(config_path)

        # load in the subcluster assignments
        labels_ds = nc.DataSet(labels_savefile)

        # use nearest neighbor (not interpolate!) to regrid integer labels
        labels_ds.regrid(wf._ds, method="nn")

        # subtract surface mean R-age from centers to get their dRs
        avgR = wf.ds_var_to_array('avgR')

        # shorthand for frequently used arrays
        self.dR = wf.ds_var_to_array('dR')[:, 0, :, :]
        self.labels = labels_ds.to_xarray()['labels'].values
        self.sublabels = labels_ds.to_xarray()['sublabels'].values

        # get the barycenter of each cluster and subcluster
        # and convert to dR by subtracting surface mean
        ts = wf._make_ts('local_age')
        self.centers_dict = construct_barycenters(self.labels,
                                                  self.sublabels,
                                                  ts)
        self.centers_dict['clusters'] -= avgR
        for label in range(int(np.nanmax(self.labels) + 1)):
            self.centers_dict['subclusters'][label] -= avgR

        # save the dR time series (subtract average R from raw) for later
        self.dR_df = wf._make_df('dR')

        # the timeslice for the dR map
        self.time = 0

        # initialise main figure
        self.fig = plt.figure()

        # plot the subclusters on a map
        cmap = Greys
        cmap.set_bad('tan')
        self.map_ax = self.fig.add_subplot(121)
        self.map_ax.set_title('Subcluster map')
        self.map_ax.imshow(make_subclusters_map(self.labels, self.sublabels),
                           origin='lower', cmap=cmap)
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

    def __get_mask(self, label, sublabel):
        # construct a condition to extract the subcluster data points
        label_match = self.labels == label
        sublabel_match = self.sublabels == sublabel
        mask = ~(label_match & sublabel_match)

        return mask

    def __similarity_to_subclust_center(self, mask, subclust_center):

        y, x = mask.shape
        mask_flat = mask.flatten()

        # extract the rows in the dR dataframe corresponding to subcluster
        subclust_df = self.dR_df.loc[~mask_flat]

        # convert to array of time series
        subclust_tss = np.array(subclust_df)

        # column vectors
        A = subclust_tss
        B = subclust_center[:, 0]

        # for each subcluster member, compute cosine similarity to center
        cosines = np.dot(A, B)/(norm(A, axis=1)*norm(B))

        # put the cosines as a new column on the dR dataframe
        df = self.dR_df.copy()
        df.loc[df.index.isin(subclust_df.index), 'cosine'] = cosines

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
        self.__similarity_to_subclust_center(mask, subclust_center)

        # update figure
        self.fig.canvas.draw()

    def __density_plots(self, mask, cluster_center, subclust_center):

        t, y, x = self.dR.shape

        # repeat the condition for every time slice
        mask = np.repeat(np.array([mask]), repeats=t, axis=0)

        # mask dR array to show current subcluster
        intrasub = np.ma.masked_where(mask, self.dR)

        # remove previous overlay
        # try:
        #    for handle in self.subclust_overlay.collections:
        #        handle.remove()
        # except AttributeError:
        #    pass

        # color in the clicked subcluster
        # self.subclust_overlay = self.map_ax.contourf(
        #    ~np.isnan(intrasub[self.time]))

        # clear the time series plot
        self.ts_ax.cla()

        # plot time series for each subcluster member
        subclust_tss = intrasub[~intrasub.mask]
        for ts in np.reshape(subclust_tss, (t, -1)).T:
            self.ts_ax.plot(ts, color='grey')

        subclust_line, = self.ts_ax.plot(subclust_center)

        # extract and plot the cluster center
        cluster_line, = self.ts_ax.plot(cluster_center)

        # compute and plot global mean dR over time
        # note to self the computation could be moved to init
        avg_dR = np.nanmean(self.dR, axis=(1, 2))
        avg_line, = self.ts_ax.plot(avg_dR)

        # consistent y range
        self.ts_ax.set_ylim([-1500, 1500])

        # add labels
        self.ts_ax.legend([subclust_line, cluster_line, avg_line],
                          ['subcluster center dR',
                           'cluster center dR',
                           'surface mean dR'])

        DdR_subclust_center, DdR_clust_center, DdR_global_avg = list(map(
            lambda benchmark:
            intrasub - benchmark.reshape(t, 1, 1).repeat(y, 1).repeat(x, 2),
            [subclust_center, cluster_center, avg_dR]
        ))

        # drop NaNs, take absolute value, compute densities and plot
        densities = list(map(
            lambda arr: gaussian_kde(np.abs(arr[~arr.mask].flatten())),
            [DdR_subclust_center, DdR_clust_center, DdR_global_avg]
        ))
        span = range(0, 750)
        self.hist_ax.cla()
        for dens in densities:
            self.hist_ax.plot(span, dens(span), linewidth=2, alpha=0.5)
        self.hist_ax.legend(['DdR_subclust_center',
                             'DdR_clust_center',
                             'DdR_global_average'])
