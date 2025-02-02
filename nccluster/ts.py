import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from kneed import KneeLocator
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from nccluster.radiocarbon import RadioCarbonWorkflow
from nccluster.utils import make_subclusters_map, make_subclust_sizes,\
    get_sublabel_colorval, make_xy_coords, reorder_labels
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from tqdm import tqdm
from tslearn.barycenters import euclidean_barycenter
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, to_sktime_dataset


class TimeSeriesWorkflowBase(RadioCarbonWorkflow):
    '''Base class for workflows involving time series.'''
    def _checkers(self):
        super()._checkers()

    def _setters(self):
        super()._setters()
        self.mask = None

    def run(self):
        self.plot_all_ts(ylabel="R-age (yrs)")
        plt.show()

    def plot_all_ts(self, var='R_age', ylabel=None):
        # plot evolution of every grid point over time
        # i.e. plot all the time series on one figure
        # initialise the figure, axis and get the data to plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ts_array = self._make_ts_array(var)

        # plot each time series
        for ts in tqdm(ts_array,
                       desc="[i] Plotting combined time series"):
            ax.plot(range(0, len(ts)), ts, linewidth='0.2')

        # some information
        ax.set_xlabel('time step')
        if not ylabel:
            ylabel = var
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' over time')

        fig.show()

    def _make_ts_array(self, var='R_age'):
        data_arr = self.ds_var_to_array(var)
        if len(data_arr.shape) == 4:
            # remember shape = (t, z, y, x)
            # we want ts_array.shape = (n, t) at z=0 with n=x*y
            # note the -1 in np.reshape tells it to figure out n on its own
            data_arr = self.ds_var_to_array(var)[:, 0]

        ts_array = np.ma.masked_array(data_arr, self.mask)
        ts_array = np.reshape(ts_array,
                              newshape=(ts_array.shape[0], -1)).T
        return ts_array

    def _make_df(self, var='R_age'):
        # convenience function to get time series as dataframe
        ts_array = self._make_ts_array(var)
        df = pd.DataFrame(ts_array)
        return df

    def _make_ts(self, var='R_age'):
        # convenience function to get a tslearn-formatted time series array
        df = self._make_df(var)
        ts_droppedna_array = np.array(df.dropna())
        ts = to_time_series_dataset(ts_droppedna_array)
        return ts


class TimeSeriesClusteringWorkflow(TimeSeriesWorkflowBase):
    '''A workflow to find clusters in the surface ocean based on the
    radiocarbon age time series at each grid point.'''
    def _checkers(self):
        super()._checkers()
        self.__check_cluster_on()
        self.__check_n_clusters()
        self.__check_max_iter()
        self.__check_n_init()
        self.__check_clustering_method()
        self.__check_scaling_bool()
        self.__check_labels_long_name()
        self.__check_palette()
        self.__check_metrics_savefile()

    def _setters(self):
        super()._setters()
        self._set_ts()

    def run(self):

        # get the model with label assignments and barycenters
        self.cluster()

        # plot and show results
        self.view_results()

    def cluster(self):
        self._fit_model()
        self.labels = self._make_labels_shaped()

    def _set_ts(self, mask=None):
        # wrapper for setting the time series attribute of this class
        target_var = self.config['timeseries']['cluster_on']
        self.ts = self._make_ts(target_var)

    def __check_cluster_on(self):
        self._check_config_option(
            'timeseries', 'cluster_on',
            missing_msg="[!] You have not specified what variable to cluster.",
            input_msg="[>] Enter which var in the dataset to cluster on: ",
            confirm_msg='[i] Will form clusters based on: '
        )
    def __check_n_clusters(self):
        self._check_config_option(
            'timeseries', 'n_clusters',
            missing_msg="[!] You have not specified the number of clusters.",
            input_msg="[>] Enter number of clusters for time series k-means: ",
            confirm_msg='[i] n_clusters: '
        )

    def __check_clustering_method(self):
        self._check_config_option(
            'timeseries', 'method',
            missing_msg="[!] You have not specified a clustering method.",
            input_msg="[>] Clustering method (k-means/k-medoids): ",
            confirm_msg="[i] Proceeding with clustering method = "
        )

    def __check_scaling_bool(self):
        self._check_config_option(
            'timeseries', 'scaling',
            default=False,
            isbool=True,
            confirm_msg="[i] Proceeding with scaling bool = "
        )

    def __check_labels_long_name(self):
        self._check_config_option(
            'default', 'labels_long_name',
            required=True,
            default='Time series clustering results',
            confirm_msg='[i] Labels variable long name: '
        )

    def __check_palette(self):
        self._check_config_option(
            'default', 'palette',
            required=True,
            default='rainbow',
            confirm_msg='[i] Palette for plots: '
        )

    def __check_max_iter(self):
        self._check_config_option(
            'timeseries', 'max_iter',
            required=True,
            default=10,
            confirm_msg="[i] Max iterations for clustering algorithm: "
        )

    def __check_n_init(self):
        self._check_config_option(
            'timeseries', 'n_init',
            required=True,
            default=10,
            confirm_msg="[i] Number of initializaitons for clustering: "
        )

    def __check_metrics_savefile(self):
        self._check_config_option(
            'timeseries', 'metrics_savefile_path',
            missing_msg="[!] You have not specified a savefile for metrics.",
            input_msg="[>] Metrics savefile path: ",
            confirm_msg="[i] Using metrics savefile path: "
        )

    def _fit_model(self, n_clusters=None, dataset=None, quiet=False):
        if n_clusters is None:
            n_clusters = self.config['timeseries'].getint('n_clusters')

        # define the keyword arguments to pass to the model
        kwargs = {
            'n_clusters': n_clusters,
            'max_iter': int(self.config['timeseries']['max_iter']),
            'n_init': int(self.config['timeseries']['n_init']),
            'metric': 'euclidean'
        }

        if dataset is None:
            dataset = self.__make_dataset()
        # initialise model using desired clustering method
        if self.config['timeseries']['method'] == 'k-means':
            if not quiet:
                print("[i] Initialising k-means model")
            self.model = TimeSeriesKMeans(**kwargs)
        elif self.config['timeseries']['method'] == 'k-medoids':
            if not quiet:
                print("[i] Initialising k-medoids model")
            self.model = TimeSeriesKMedoids(**kwargs)
            dataset = to_sktime_dataset(dataset)
        else:
            print('[fatal] Unrecognised clustering method specified')
            exit()
        if not quiet:
            print("[i] Fitting model, please stand by")

        # actually fit the model
        self.model.fit(dataset)

        # reorder labels for consistent results
        self.model.labels_ = reorder_labels(self.model.labels_,
                                            self._make_ts_array())

    def __make_dataset(self):
        dataset = self.ts

        # optionally scale the data (aids in shape detection
        # but loses amplitude information)
        if self.config['timeseries'].getboolean('scaling'):
            print("[i] Normalising time series")
            dataset = TimeSeriesScalerMeanVariance().fit_transform(dataset)

        return dataset

    def view_results(self, combined_ax_bool=True):
        self.fig = plt.figure()
        # plt.rcParams['figure.constrained_layout.use'] = True
        self._plot_ts_clusters(combined_ax_bool=combined_ax_bool,
                               var_to_plot=self.config['timeseries']['cluster_on'],
                               units='fraction')
        self._map_clusters(combined_ax_bool=combined_ax_bool)
        plt.show()

    def _plot_ts_clusters(self, combined_ax_bool=True, var_to_plot="R-age",
                          units='no units'):
        # get predictions for our time series from trained model
        # i.e. to which cluster each time series belongs
        print("[i] Plotting time series by cluster")
        labels = self.labels.flatten()
        labels = labels[~np.isnan(labels)]

        n_clusters = int(np.nanmax(labels) + 1)
        norm = Normalize(vmin=0, vmax=n_clusters-1)
        cmap = get_cmap(self.config['default']['palette'])

        # for each cluster/label
        if combined_ax_bool:
            combined_ax = self.fig.add_subplot(2, 2, 4)
            combined_ax.set_title(f"Combined {var_to_plot} histories")
            combined_ax.set_xlabel("time step")
            combined_ax.yaxis.set_label_position("right")
            combined_ax.yaxis.tick_right()

        for label in range(0, n_clusters):
            # create a subplot in a table with n_cluster rows and 1 column
            # this subplot is number label+1 because we're counting from 0
            ax = self.fig.add_subplot(n_clusters, 2,
                                      2 * n_clusters - 2 * label - 1)
            if label == n_clusters - 1:
                ax.set_title(f"Cluster {var_to_plot} histories")
                ax.set_ylabel(f"{var_to_plot} ({units})")
            color = cmap(norm(label))
            # for every time series that has been assigned this label
            cluster_tss = self.ts[labels == label]
            for ts in cluster_tss:
                # plot with a thin transparent line
                ax.plot(ts.ravel(), color=color, alpha=.2)
                if combined_ax_bool:
                    combined_ax.plot(ts.ravel(), color=color, alpha=.1,
                                     linewidth=0.4)
            # plot the cluster barycenter
            ax.plot(euclidean_barycenter(cluster_tss).ravel(), "r-")
            if label > 0:
                ax.set_xticks([])
            else:
                ax.set_xlabel('time step')

            # put the cluster number on the right of the current ax
            ax.text(1.05, 0.5, n_clusters - label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation='horizontal',
                    transform=ax.transAxes)

    def _map_clusters(self, combined_ax_bool=True):
        print("[i] Mapping out clusters")

        # view the clusters on a map
        if combined_ax_bool:
            ax = self.fig.add_subplot(222)
        else:
            ax = self.fig.add_subplot(122)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(self.labels, origin='lower',
                  cmap=self.config['default']['palette'])
        ax.set_title(self.config['default']['labels_long_name'])

    def _make_labels_shaped(self):
        # get the timeseries as a dataframe and append labels to non-empty rows
        df = self._make_df(self.config['timeseries']['cluster_on'])
        df.loc[
            df.index.isin(df.dropna().index),
            'labels'] = self.model.labels_

        # convert to array of labels and reshape into 2D for map
        labels_flat = np.ma.masked_array(df['labels'])
        _, _, y, x = self._age_array.shape
        labels_shaped = np.reshape(labels_flat, [y, x])
        return labels_shaped

    def _make_labels_data_array(self):

        # get the raw labels array
        labels_shaped = self._make_labels_shaped()
        coords = make_xy_coords(self._ds)
        long_name = self.config['default']['labels_long_name']
        # create the data array from our labels and
        # the x-y coords copied from the original dataset
        data_array = xr.DataArray(data=labels_shaped,
                                  coords=coords,
                                  name='labels',
                                  attrs={'long_name': long_name}
                                  )
        return data_array

    def save_labels(self, filename):
        darray = self._make_labels_data_array()
        darray.to_netcdf(filename)
        print(f"[i] Saved labels to {filename}")

    def save_centroids(self, filename):
        # if no normalisation, can save the centers as they are
        if not self.config['timeseries'].getboolean('scaling'):
            centers = self.model.cluster_centers_
        # otherwise, the original amplitudes are irrecoverable
        # need to get the timeseries for each cluster and compute centers
        else:
            centers = self._make_centers()

        with open(filename, 'wb') as file:
            pickle.dump(centers, file)

    def _make_centers(self):
        centers = []
        for label in range(self.model.n_clusters):
            cluster_tss = self.ts[self.model.labels_ == label]
            cluster_center = euclidean_barycenter(cluster_tss)
            centers.append(cluster_center)
        return centers

    def get_knee(self, n_min=2, n_max=10):
        inertias = []
        dataset = self.__make_dataset()
        K_range = range(n_min, n_max + 1)

        for K in tqdm(K_range, desc='[i] locating knee: ', leave=True,
                      position=1):
            self._fit_model(n_clusters=K, dataset=dataset, quiet=True)
            inertias.append(self.model.inertia_)

        kn = KneeLocator(x=K_range, y=inertias, curve='convex',
                         direction='decreasing')
        return kn.knee

    def export_clustering_metrics(self, n_min=2, n_max=10):

        csv_string = "K,inertia,sil_score,ch_index,db_index\n"

        # get the time series in the correct format
        dataset = self.__make_dataset()

        # the range of K (number of clusters)
        K_range = range(n_min, n_max + 1)

        # for every value of K (with a nice progress bar)
        for K in tqdm(K_range, desc='[i] metrics: ', leave=True, position=1):

            # fit the model and get the labels
            self._fit_model(n_clusters=K, dataset=dataset, quiet=True)
            labels = self.model.labels_

            inertia = self.model.inertia_
            sil = silhouette_score(dataset, labels, metric='euclidean')
            ch = calinski_harabasz_score(dataset[:, :, 0], labels)
            db = davies_bouldin_score(dataset[:, :, 0], labels)

            csv_string += f"{K},{inertia},{sil},{ch},{db}\n"

        savefile = self.config['timeseries']['metrics_savefile_path']
        with open(savefile, 'w+') as f:
            f.write(csv_string)

        print(f"[i] Exported clustering metrics to {savefile}")

    def load_labels_from_file(self, filename):
        print("[i] Loading labels from " + filename)
        labels_data_arr = xr.open_dataarray(filename)
        labels_shaped = labels_data_arr.values
        self.labels = labels_shaped


class TwoStepTimeSeriesClusterer(TimeSeriesClusteringWorkflow):

    def _setters(self):
        super()._setters()
        self.centroids_dict = {}
        t, z, y, x = self._age_array.shape
        self.labels2step = np.zeros((y, x, 2))

    def run(self):
        print("[i] This workflow will override the config setting for scaling")
        self.cluster()
        self.view_results()

    def cluster(self):
        # set scaling on to form shape-based clusters
        self.config['timeseries']['scaling'] = 'True'
        self._fit_model()
        self.centroids_dict['clusters'] = self._make_centers()

        # set scaling off to form amplitude-based subclusters
        self.config['timeseries']['scaling'] = 'False'
        self.labels2step = self.__make_subclusters()

    def __make_subclusters(self):
        # scaling should be off to make amplitude-based subclusters
        # but this is not enforced in case shape-based subclustering is desired
        if self.config['timeseries'].getboolean('scaling'):
            print("[!] Scaling has not been turned off for subclusters")

        # some useful variables
        labels = self._make_labels_shaped()
        n_clusters = int(self.config['timeseries']['n_clusters'])
        _, _, y, x = self._age_array.shape

        # we'll save the cluster and subcluster assignments to this array
        labels2step = np.full((y, x, 2), np.nan)

        # index 0 = step one = cluster labels
        labels2step[:, :, 0] = labels

        # for each shape cluster
        for label in tqdm(range(0, n_clusters),
                          leave=True, position=0,
                          desc='[i] subdividing cluster: '):

            self.__mask_cluster(labels, label)

            # run algorithm again on these points, without normalisation
            elbow = self.get_knee()
            self._fit_model(n_clusters=elbow, quiet=True)
            self.centroids_dict['cluster_' + str(label)] =\
                self.model.cluster_centers_

            # get the labels for current subcluster
            sublabels = self._make_labels_shaped()

            # for every grid point that is not nan
            for arg in np.argwhere(~np.isnan(sublabels)):
                # unpack the 2D index
                yi, xi = arg

                # set the subcluster label in the two-step map (second column)
                labels2step[yi, xi, 1] = sublabels[yi, xi]

        # reset the mask to its original state now we're done with subclusters
        self.mask = None
        self._set_ts()

        # return the results
        return labels2step

    def __mask_cluster(self, labels, cluster):
        time_steps, *_ = self._age_array.shape

        # mask all the points not belonging to this cluster
        cluster_mask = (labels != cluster)

        # repeat the mask for each time step and set it
        cluster_mask = np.array([cluster_mask])
        cluster_mask = np.repeat(cluster_mask, time_steps, axis=0)
        self.mask = cluster_mask
        # re-buld the time series dataset (now restricted to this cluster)
        self._set_ts()

    def _map_clusters(self, combined_ax_bool=True):
        # override parent method
        subclusters_map = make_subclusters_map(self.labels2step[:, :, 0],
                                               self.labels2step[:, :, 1])
        if combined_ax_bool:
            ax = self.fig.add_subplot(222)
        else:
            ax = self.fig.add_subplot(122)
        ax.imshow(subclusters_map, origin='lower',
                  cmap=self.config['default']['palette'])
        ax.set_title(self.config['default']['labels_long_name'])
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_ts_clusters(self, combined_ax_bool = True):
        sublabels = self.labels2step[:, :, 1]
        sublabels = sublabels[~np.isnan(sublabels)].flatten()
        labels = self.labels2step[:, :, 0]
        labels = labels[~np.isnan(labels)].flatten()

        subclust_sizes = make_subclust_sizes(self.labels2step[:, :, 0],
                                             self.labels2step[:, :, 1])
        n_clusters = len(subclust_sizes)

        interval = 0.5
        norm = Normalize(vmin=0-interval/2, vmax=n_clusters-1+interval/2)
        cmap = get_cmap(self.config['default']['palette'])

        # for each cluster/label
        if combined_ax_bool:
            combined_ax = self.fig.add_subplot(224)
            combined_ax.set_title("Combined R-age histories")
            combined_ax.set_xlabel("time step")
            combined_ax.yaxis.set_label_position("right")
            combined_ax.yaxis.tick_right()

        # for each cluster/label
        for label in range(n_clusters):
            # create a subplot in a table with n_cluster rows and 1 column
            # this subplot is number label+1 because we're counting from 0
            ax = self.fig.add_subplot(n_clusters, 2,
                                      2 * (n_clusters - label) - 1)

            if label == n_clusters - 1:
                ax.set_title("Cluster R-age histories")
                ax.set_ylabel("R-age (yrs)")
            if label == 0:
                ax.set_xlabel('time step')

            barycenters = []
            subclust_size = subclust_sizes[label]
            for sublabel in range(subclust_size):
                colorval = get_sublabel_colorval(label, sublabel,
                                                 subclust_size)
                color = cmap(norm(colorval))

                label_match = labels == label
                sublabel_match = sublabels == sublabel
                subcluster_tss = self.ts[label_match & sublabel_match]
                for ts in subcluster_tss:
                    ax.plot(ts.ravel(), color=color)
                    if combined_ax_bool:
                        combined_ax.plot(ts.ravel(), color=color, alpha=.1,
                                            linewidth=0.2)
                barycenter = euclidean_barycenter(subcluster_tss)
                barycenters.append(barycenter)
            # need to plot these last else they'd be covered by subcluster ts
            for barycenter in barycenters:
                ax.plot(barycenter.ravel(), color='black', linewidth=0.5)

            # put the cluster number on the right of the current ax
            ax.text(1.05, 0.5, n_clusters - label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation='horizontal',
                    transform=ax.transAxes)

    def save_labels(self, filename):
        labels_darray = self._make_labels_data_array(step=0)
        sublabels_darray = self._make_labels_data_array(step=1)
        coords = make_xy_coords(self._ds)
        dataset = xr.Dataset({'labels': labels_darray,
                              'sublabels': sublabels_darray},
                             coords=coords
                             )

        dataset.to_netcdf(filename)
        print(f"[i] Saved labels and sublabels to {filename}")

    def load_labels_from_file(self, filename):

        # load in the subcluster assignments
        labels_ds = xr.load_dataset(filename)
        self.labels2step[:, :, 0] = labels_ds['labels'].values
        self.labels2step[:, :, 1] = labels_ds['sublabels'].values

    def save_centroids(self, filename):
        # bug: does not match reordered labeld
        centroids_dict = self.centroids_dict

        # fix inconsistent dimensions between clustering algorithms
        # in this case ensure output is (n_clusters, n_time_steps, n_dims)
        # i.e. force tslearn format over sktime
        # code block untested!
        for key in centroids_dict.keys():
            if centroids_dict[key].shape[1] == 1:
                centroids_dict[key] = np.moveaxis(
                    centroids_dict[key], 1, 2)

        with open(filename, 'wb') as file:
            pickle.dump(centroids_dict, file)

    def _make_labels_data_array(self, step=0):
        # override
        # get the raw labels array
        labels_shaped = self.labels2step[:, :, step]
        coords = make_xy_coords(self._ds)
        option = 'labels_long_name' if step == 0 else 'sublabels_long_name'
        long_name = self.config['default'][option]
        # create the data array from our labels and
        # the x-y coords copied from the original dataset
        data_array = xr.DataArray(data=labels_shaped,
                                  coords=coords,
                                  name='labels',
                                  attrs={'long_name': long_name}
                                  )
        return data_array


class dRWorkflow(TimeSeriesWorkflowBase):

    def _preprocess_ds(self):
        # compute local age from radiocarbon
        super()._preprocess_ds()

        # restrict analysis to surface
        self._ds.top()

        # define new surface variable, dR = R_age - mean surface age
        self.__compute_avgR()
        self.__compute_dR()

    def __compute_dR(self):

        # compute R-age difference from surface mean
        self._ds.assign(dR=lambda x: x.R_age - x.avgR)

    def __compute_avgR(self):

        # compute mean surface R-age
        self._ds.assign(avgR=lambda x: spatial_mean(x.R_age))
