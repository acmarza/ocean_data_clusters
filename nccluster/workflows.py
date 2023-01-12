import atexit
import configparser
# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import pandas as pd
import pickle
import xarray as xr

from kneed import KneeLocator
from math import log
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
#                            silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from tqdm import tqdm
from tslearn.barycenters import euclidean_barycenter
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, to_sktime_dataset

from nccluster.multisliceviewer import MultiSliceViewer
from nccluster.corrviewer import CorrelationViewer, DendrogramViewer
from nccluster.utils import make_subclusters_map, make_subclust_sizes,\
    get_sublabel_colorval

nc.options(lazy=False)


class Workflow:
    '''Base class for defining a workflow that operates on a netCDF dataset'''
    def __init__(self, config_path):
        # save the config_path as an attribute, read config and
        # be ready to save config changes  when script exits
        self.config_path = config_path
        self.__read_config_file()
        atexit.register(self.__offer_save_config)

        # make sure all required fields are defined in the config
        # before loading the data from file and applying any preprocessing
        self._checkers()
        self.__load_ds()
        self._preprocess_ds()

        # initialise any other required attributes
        self._setters()

    def __read_config_file(self):
        # parse config file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        print(f"[i] Using config: {self.config_path}")

    def __offer_save_config(self):
        # will only ask to save if a new options is added to config that's
        # not already in the config file
        # first read in the original config again
        original_config = configparser.ConfigParser()
        original_config.read(self.config_path)

        # then for each config section and option, check for additions
        for section in dict(self.config.items()).keys():
            for option in dict(self.config[section]).keys():
                if not original_config.has_option(section, option):
                    print(f"[i] The script is about to exit, but you have \
unsaved changes to {self.config_path}.")
                    yn = input("[>] Save modified config to file? (y/n): ")
                    if yn == 'y':
                        with open(self.config_path, 'w') as file:
                            self.config.write(file)
                    return

    def _check_config_option(self, section, option,
                             missing_msg="[!] Missing option in config",
                             input_msg="[>] Please enter value: ",
                             confirm_msg="[i] Got value: ",
                             default=None,
                             required=True,
                             isbool=False,
                             force=False):
        # convenience function to check if an option is defined in the config;
        # if not, interactively get its value and add to config
        try:
            # don't attempt to read from config if option is forced
            # move straight to except block where default should be applied
            if force:
                raise KeyError
            # otherwise try to read value from config
            value = self.config[section][option]
        except KeyError:
            # do nothing if option missing from config and not required
            if not required:
                return

            # apply default or ask user to input a value
            if default is not None:
                value = default

            else:
                # ask the user for the value
                print(missing_msg)
                value = input(input_msg)

                # for bools expect a y/n answer
                if isbool:
                    value = (value.lower() == 'y')

            # may need to create a new section
            if not self.config.has_section(section):
                self.config.add_section(section)

            value = str(value)
            # set the option we've just read in
            self.config[section][option] = value

        # either way echo the value to the user for sanity checking
        print(confirm_msg + value)

    def __check_nc_files(self):
        self._check_config_option(
            'default', 'nc_files',
            missing_msg='[i] Data file path not provided',
            input_msg='[>] Please type the path of the netCDF file to use: ',
            confirm_msg='[i] NetCDF file(s): '
        )

    def __check_vars_subset(self):
        self._check_config_option(
            'default', 'subset',
            required=False,
            confirm_msg="[i] Data will be subset keeping only variables: "
        )

    def __check_surface_only(self):
        self._check_config_option(
            'default', 'surface_only',
            default=False, isbool=True,
            confirm_msg='[i] Restrict analysis to ocean surface: '
        )

    def __check_timesteps_subset(self):
        self._check_config_option(
            'default', 'timesteps_subset',
            required=False,
            confirm_msg='[i] Data will be subset to the following timesteps: '
        )

    def __load_ds(self):
        print("[i] Loading data")

        # get a list of file paths and create the DataSet object
        nc_files = self.config['default']['nc_files'].split(",")
        self._ds = nc.DataSet(nc_files)

    def _preprocess_ds(self):
        # optionally limit analysis to a subset of variables
        if self.config.has_option('default', 'vars_subset'):
            subset = self.config['default']['vars_subset'].split(",")
            print("[i] Subsetting data")
            self._ds.subset(variable=subset)

        # optionally limit analysis to the ocean surface (level 0):
        if self.config['default'].getboolean('surface_only'):
            self._ds.top()

        # optionally limit analysis to an interval of time steps
        if self.config.has_option('default', 'timesteps_subset'):
            str_opt = self.config['default']['timesteps_subset']
            start, end = str_opt.strip('[]').split(",")
            self._ds.subset(timesteps=range(int(start), int(end)))

        # merge datasets if multiple files specified in config
        try:
            print("[i] Merging datasets")
            self._ds.merge()
        except Exception:
            print("[fatal] You have specified multiple netCDF files to use",
                  "but these could not be merged for further analysis.",
                  "Consider subsetting the data as shown in the example",
                  "config, keeping only variables of interest. Or merge your",
                  "datasets externally.")
            exit()

    def _checkers(self):
        self.__check_nc_files()
        self.__check_vars_subset()
        self.__check_surface_only()
        self.__check_timesteps_subset()

    def _setters(self):
        print("[i] All attributes have been initialized")

    def regrid_to_ds(self, target_ds):
        # interpolate/extrapolate the data to fit the grid of a target dataset
        self._ds.regrid(target_ds)
        # features and time series need to be re-created for new grid
        self._setters()

    def run(self):
        print("[!] Nothing to run. Did you forget to override self.run()?")

    def list_plottable_vars(self):
        # get an alphabetical list of plottable variables
        plottable_vars = self._ds.variables

        # easy reference to a dataframe containing useful info on variables
        df = self._ds.contents

        # list plottable variables for the user to inspect
        for i, var in enumerate(plottable_vars):
            try:
                # print extra info if possible
                long_name = df.loc[df['variable'] == var].long_name.values[0]
                print(f"{i}. {var}\n\t{long_name}")
            except Exception:
                # just print the variable name
                print(f"{i}. {var}")

    def plot_var(self, var_to_plot):
        try:
            # get the data as an array
            ds_tmp = self._ds.copy()
            ds_tmp.subset(variables=var_to_plot)
            var_xr = ds_tmp.to_xarray()
            data_to_plot = var_xr[var_to_plot].values

            try:
                # add more info in plot title if possible
                plot_title = var_to_plot + " ("\
                    + var_xr[var_to_plot].long_name + ") "\
                    + var_xr[var_to_plot].units
            except AttributeError:
                # stick to basic info if the above goes wrong
                plot_title = var_to_plot

            # pass on the data array to interactive viewer
            MultiSliceViewer(data_to_plot, plot_title).show()

        except IndexError:
            print(f"[!] {var_to_plot} not found; check spelling")

    def ds_var_to_array(self, var_name):
        # some  manipulation to isolate age data and cast it into a useful form
        # make a copy of the original dataset to subset to one variable
        ds_tmp = self._ds.copy()
        ds_tmp.subset(variables=var_name)

        # convert to xarray and extract the numpy array of values
        as_xr = ds_tmp.to_xarray()
        as_array = as_xr[var_name].values

        return as_array


class RadioCarbonWorkflow(Workflow):
    '''Base for workflows involving dc14 and radiocarbon age calculations.'''
    def _preprocess_ds(self):
        # for computing radiocarbon ages, will use nctoolkit's DataSet.assign,
        # however this will require knowing the variable names in advance;
        # to avoid confusion, rename the variables used in this computation
        # to something simple and consistent
        super()._preprocess_ds()
        print("[i] Preprocessing data")
        self.__construct_rename_dict(['dc14', 'dic', 'di14c'])
        self.__rename_vars()

        # compute dc14 if not present in dataset and we have the necessary vars
        if 'dic' in self._ds.variables and\
                'di14c' in self._ds.variables and\
                'dc14' not in self._ds.variables:
            self.__compute_dc14()

        # compute radiocarbon ages if dc14 exists in dataset
        if 'dc14' in self._ds.variables:
            self.__check_mean_radiocarbon_lifetime()
            self.__check_atm_dc14()
            self.__compute_local_age()

    def __construct_rename_dict(self, vars):
        # get the name of each variable as it appears in the dataset
        self.__rename_dict = {}
        for var in vars:
            try:
                # try to read from config
                self.__rename_dict[var] = self.config['radiocarbon'][var]
            except KeyError:
                # if option not defined, assume user does not need it
                # and continue without asking
                pass

    def __rename_vars(self):
        # rename variables to be used in calculations for easy reference
        for key, value in self.__rename_dict.items():
            self._ds.rename({value: key})
            print(f"[i] Renamed variable {value} to {key}")

    def __check_mean_radiocarbon_lifetime(self):
        self._check_config_option(
            'radiocarbon', 'mean_radiocarbon_lifetime',
            missing_msg="[!] Mean lifetime of radiocarbon was not provided",
            input_msg="[>] Enter mean radiocarbon lifetime \
                                        (Cambridge=8267, Libby=8033): ",
            confirm_msg="[i] Mean lifetime of radiocarbon: ")

    def __check_atm_dc14(self):
        self._check_config_option(
            'radiocarbon', 'atm_dc14',
            missing_msg="[!] Atmospheric dc14 was not provided",
            input_msg="[>] Type the atmospheric dc14 as integer in per mil: ",
            confirm_msg="[i] Proceeding with atmospheric dc14 = "
        )

    def __compute_local_age(self):
        # using mean radiocarbon lifetime, dc14, atm_dc14
        mean_radio_life =\
            self.config['radiocarbon'].getint('mean_radiocarbon_lifetime')
        atm_dc14 = self.config['radiocarbon'].getint('atm_dc14')
        self._ds.assign(local_age=lambda x: -mean_radio_life*log(
            (x.dc14/1000 + 1)/(atm_dc14/1000 + 1))
                       )
        print("[i] Converted dc14 to age")

    def __compute_dc14(self):
        # from dic and di14c
        self._ds.assign(dc14=lambda x:
                        (x.di14c/x.dic-1)*1000)
        print("[i] Computed dc14 from di14c and dic")

    def _setters(self):
        self._age_array = self.ds_var_to_array('local_age')


class TimeSeriesWorkflowBase(RadioCarbonWorkflow):
    '''Base class for workflows involving time series.'''
    def _checkers(self):
        super()._checkers()
        self.__check_plot_all_ts_bool()

    def _setters(self):
        super()._setters()
        self.mask = None

    def run(self):
        # this base class simply plots all the time series if asked to
        if self.config['timeseries'].getboolean('plot_all_ts'):
            self.plot_all_ts()

    def __check_plot_all_ts_bool(self):
        missing_msg = '[!] You have not specified whether to show\
 a plot of all the R-age time series'
        self._check_config_option(
            'timeseries', 'plot_all_ts', missing_msg=missing_msg,
            input_msg='[>] Show a plot of all the R-age time series? (y/n): ',
            confirm_msg='[i] Plot all time series bool: ',
            isbool=True
        )

    def plot_all_ts(self):
        # plot evolution of every grid point over time
        # i.e. plot all the time series on one figure
        # initialise the figure, axis and get the data to plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ts_array = self._make_ts_array()

        # plot each time series
        for ts in tqdm(ts_array,
                       desc="[i] Plotting combined time series"):
            ax.plot(range(0, len(ts)), ts)

        # some information
        ax.set_xlabel('time step')
        ax.set_ylabel('age')
        ax.set_title('age over time')

        fig.show()

    def _make_ts_array(self):
        # remember age_array.shape = (t, z, y, x)
        # we want ts_array.shape = (n, t) at z=0 with n=x*y
        # note the -1 in np.reshape tells it to figure out n on its own
        ts_array = np.ma.masked_array(self._age_array[:, 0], self.mask)
        ts_array = np.reshape(ts_array,
                              newshape=(ts_array.shape[0], -1)).T
        return ts_array

    def _make_df(self):
        # convenience function to get time series as dataframe
        ts_array = self._make_ts_array()
        df = pd.DataFrame(ts_array)
        return df

    def _make_ts(self):
        # convenience function to get a tslearn-formatted time series array
        df = self._make_df()
        ts_droppedna_array = np.array(df.dropna())
        ts = to_time_series_dataset(ts_droppedna_array)
        return ts


class CorrelationWorkflow(TimeSeriesWorkflowBase):
    '''A workflow for visualising correlations between time series at each
    grid point in the surface ocean.'''
    def _checkers(self):
        super()._checkers()

        # only worry about save files if required to use p-values
        self.__check_pvalues_bool()
        if self.config['correlation'].getboolean('pvalues'):
            self.__check_corr_mat_file()
            self.__check_pval_mat_file()

    def run(self):
        # run parent workflow (time series plot)
        TimeSeriesWorkflowBase.run(self)

        # keyword arguments to be passed to CorrelationViewer
        kwargs = self.config['correlation']

        # initialise and show CorrelationViewer
        self.corrviewer = CorrelationViewer(self._age_array,
                                            'R-ages', **kwargs)
        plt.show()

    def __check_pvalues_bool(self):
        self._check_config_option(
            'correlation', 'pvalues',
            missing_msg="[!] You have not specified the p-values boolean.",
            input_msg="[>] Mask out grid points with insignificant \
                    ( p > 0.05 ) correlation? (y/n): ",
            confirm_msg="[i] P-values boolean: "
        )

    def __check_corr_mat_file(self):
        self._check_config_option(
            'correlation', 'corr_mat_file',
            missing_msg="[!] Correlation matrix save file not provided",
            input_msg="[>] Enter file path to save/read correlation matrix: ",
            confirm_msg='[i] Correlation matrix savefile: '
        )

    def __check_pval_mat_file(self):
        self._check_config_option(
            'correlation', 'pval_mat_file',
            missing_msg="[!] P-value matrix save file not provided",
            input_msg="[>] Enter file path to save/read p-value matrix : ",
            confirm_msg='[i] P-value matrix savefile: '
        )


class TimeSeriesClusteringWorkflow(TimeSeriesWorkflowBase):
    '''A workflow to find clusters in the surface ocean based on the
    radiocarbon age time series at each grid point.'''
    def _checkers(self):
        super()._checkers()
        self.__check_n_clusters()
        self.__check_clustering_method()
        self.__check_scaling_bool()
        self.__check_labels_long_name()
        self.__check_cmap()

    def _setters(self):
        super()._setters()
        self._set_ts()

    def run(self):
        # run the parent workflow (plotting all time series)
        TimeSeriesWorkflowBase.run(self)

        # get the model with label assignments and barycenters
        self.cluster()

        # plot and show results
        self.view_results()

    def cluster(self):
        self._fit_model()

    def _set_ts(self, mask=None):
        # wrapper for setting the time series attribute of this class
        self.ts = self._make_ts()

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
            required=False,
            default='Time series clustering results',
            confirm_msg='[i] Labels variable long name: '
        )

    def __check_cmap(self):
        self._check_config_option(
            'default', 'palette',
            required=True,
            default='rainbow',
            confirm_msg='[i] Palette for plots: '
        )

    def _fit_model(self, n_clusters=None, dataset=None, quiet=False):
        if n_clusters is None:
            n_clusters = self.config['timeseries'].getint('n_clusters')

        # define the keyword arguments to pass to the model
        kwargs = {
            'n_clusters': n_clusters,
            'max_iter': 10,
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

    def __make_dataset(self):
        dataset = self.ts

        # optionally scale the data (aids in shape detection
        # but loses amplitude information)
        if self.config['timeseries'].getboolean('scaling'):
            print("[i] Normalising time series")
            dataset = TimeSeriesScalerMeanVariance().fit_transform(dataset)

        return dataset

    def view_results(self):
        self.fig = plt.figure()
        plt.rcParams['figure.constrained_layout.use'] = True
        self._plot_ts_clusters()
        self._map_clusters()
        plt.show()

    def _plot_ts_clusters(self):
        # get predictions for our time series from trained model
        # i.e. to which cluster each time series belongs
        labels = self.model.labels_
        n_clusters = self.model.n_clusters
        norm = Normalize(vmin=0, vmax=n_clusters-1)
        cmap = get_cmap(self.config['default']['palette'])
        # for each cluster/label
        for label in range(0, n_clusters):
            # create a subplot in a table with n_cluster rows and 1 column
            # this subplot is number label+1 because we're counting from 0
            ax = self.fig.add_subplot(n_clusters, 2, 2 * label + 1)
            color = cmap(norm(label))
            # for every time series that has been assigned this label
            cluster_tss = self.ts[labels == label]
            for ts in cluster_tss:
                # plot with a thin transparent line
                ax.plot(ts.ravel(), color=color, alpha=.2)
            # plot the cluster barycenter
            ax.plot(self.model.cluster_centers_[label, 0], "k-")
            ax.plot(euclidean_barycenter(cluster_tss).ravel(), "b:")

    def _map_clusters(self):
        # get the 2D labels array
        labels_shaped = self._make_labels_shaped()

        # view the clusters on a map
        ax = self.fig.add_subplot(122)
        ax.imshow(labels_shaped, origin='lower',
                  cmap=self.config['default']['palette'])

    def _make_labels_shaped(self):
        # get the timeseries as a dataframe and append labels to non-empty rows
        df = self._make_df()
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
        coords = self._make_xy_coords()
        long_name = self.config['default']['labels_long_name']
        # create the data array from our labels and
        # the x-y coords copied from the original dataset
        data_array = xr.DataArray(data=labels_shaped,
                                  coords=coords,
                                  name='labels',
                                  attrs={'long_name': long_name}
                                  )
        return data_array

    def _make_xy_coords(self):

        # copy the coords of the original dataset, but keep only x and y
        all_coords = self._ds.to_xarray().coords
        coords = {}
        for key in all_coords:
            try:
                if all_coords[key].axis in ('X', 'Y'):
                    coords[key] = all_coords[key]
            except AttributeError:
                pass
        # arcane magic to put the coordinates in reverse order
        # because otherwise DataArray expects the transpose of what we have
        coords = dict(reversed(list(coords.items())))
        return coords

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

    def clustering_metrics(self, n_min=2, n_max=10, show=True):
        inertias = []

        if show:
            sil_scores = []
            ch_scores = []
            db_scores = []
        dataset = self.__make_dataset()
        n_range = range(n_min, n_max+1)
        for n in tqdm(n_range, desc='[i] metrics: ', leave=True, position=1):
            self._fit_model(n_clusters=n, dataset=dataset, quiet=True)
            labels = self.model.labels_
            inertias.append(self.model.inertia_)
            if show:
                sil_scores.append(
                    silhouette_score(dataset, labels, metric='euclidean'))
                ch_scores.append(calinski_harabasz_score(dataset[:, :, 0],
                                                         labels))
                db_scores.append(davies_bouldin_score(dataset[:, :, 0],
                                                      labels))

        kn = KneeLocator(x=n_range,
                         y=inertias,
                         curve='convex',
                         direction='decreasing')
        if not show:
            return kn.knee

        fig, axes = plt.subplots(4, 1)
        scores = [inertias, sil_scores, ch_scores, db_scores]
        titles = ['Sum of squared errors, choose elbow point',
                  'Silhouette Score, higher is better',
                  'Calinski-Harabasz Index, higher is better',
                  'Davies-Bouldin Index, lower is better']
        for (ax, score, title) in zip(axes, scores, titles):
            ax.plot(n_range, score)
            ax.set_title(title)
        axes[0].axvline(kn.knee)

        plt.show()

        return kn.knee


class TwoStepTimeSeriesClusterer(TimeSeriesClusteringWorkflow):

    def _setters(self):
        super()._setters()
        self.centroids_dict = {}

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
            elbow = self.clustering_metrics(show=False)
            self._fit_model(n_clusters=elbow, quiet=True)
            self.centroids_dict['cluster_' + str(label)] =\
                self.model.cluster_centers_

            # get the labels for current subcluster
            sublabels = self._make_labels_shaped()
            sublabels = self.__reorder_sublabels(sublabels)

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

    def __reorder_sublabels(self, labels):

        # prepare an empty array to hold the average variance of each cluster
        n_clusters = int(np.nanmax(labels) + 1)
        order_scores = np.zeros(n_clusters)
        for label in range(0, n_clusters):
            # assume at this point self.mask is still set for this cluster
            # so can grab the timeseries array right away
            # and zip it with the labels
            cluster_tss = zip(self._make_ts_array(), labels.flatten())

            # the zip above lets us this neat list comprehension
            # to retrieve just the time series with the current label
            subcluster_tss = [ts for (ts, ll) in cluster_tss if ll == label]

            order_scores[label] = np.mean(np.array(subcluster_tss))

        idx = np.argsort(order_scores)
        orig = np.arange(n_clusters)
        mapping = dict(zip(idx, orig))

        ordered_labels = np.copy(labels)
        for key in mapping:
            ordered_labels[labels == key] = mapping[key]

        return ordered_labels

    def _map_clusters(self):
        # override parent method
        subclusters_map = make_subclusters_map(self.labels2step[:, :, 0],
                                               self.labels2step[:, :, 1])
        ax = self.fig.add_subplot(122)
        ax.imshow(subclusters_map, origin='lower', cmap='twilight')

    def _plot_ts_clusters(self):
        sublabels = self.labels2step[:, :, 1]
        sublabels = sublabels[~np.isnan(sublabels)].flatten()
        labels = self.labels2step[:, :, 0]
        labels = labels[~np.isnan(labels)].flatten()

        subclust_sizes = make_subclust_sizes(self.labels2step[:, :, 0],
                                             self.labels2step[:, :, 1])
        n_clusters = len(subclust_sizes)

        interval = 0.5
        norm = Normalize(vmin=0-interval/2, vmax=n_clusters-1+interval/2)
        cmap = get_cmap('twilight')

        # for each cluster/label
        for label in range(n_clusters):
            # create a subplot in a table with n_cluster rows and 1 column
            # this subplot is number label+1 because we're counting from 0
            ax = self.fig.add_subplot(n_clusters, 2, 2 * label + 1)

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
                barycenter = euclidean_barycenter(subcluster_tss)
                barycenters.append(barycenter)
            # need to plot these last else they'd be covered by subcluster ts
            for barycenter in barycenters:
                ax.plot(barycenter.ravel(), color='magenta', linewidth=0.5)

    def save_labels(self, filename):
        labels_darray = self._make_labels_data_array(step=0)
        sublabels_darray = self._make_labels_data_array(step=1)
        coords = self._make_xy_coords()
        dataset = xr.Dataset({'labels': labels_darray,
                              'sublabels': sublabels_darray},
                             coords=coords
                             )

        dataset.to_netcdf(filename)
        print(f"[i] Saved labels and sublabels to {filename}")

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
        coords = self._make_xy_coords()
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

        # define new surface variable, dR = local_age - mean surface age
        self.__compute_avgR()
        self.__compute_dR()

    def __compute_dR(self):

        # compute R-age difference from surface mean
        self._ds.assign(dR=lambda x: x.local_age - x.avgR)

    def __compute_avgR(self):

        # compute mean surface R-age
        self._ds.assign(avgR=lambda x: spatial_mean(x.local_age))


class DendrogramWorkflow(RadioCarbonWorkflow):

    def run(self):
        t, _, y, x = self._age_array.shape
        evolutions = np.reshape(self._age_array[:, 0], [t, x*y]).T
        corr_mat = np.corrcoef(evolutions)
        DendrogramViewer(corr_mat, (y, x))


class KMeansWorkflowBase(Workflow):

    def _checkers(self):
        self.__check_n_init()
        self.__check_max_iter()

    def _setters(self):
        self.__set_selected_vars()
        self.__set_pipeline()
        self.__set_features()

    def __set_selected_vars(self):
        # get parameters to run k-means for from file, or interactively
        try:
            selected_vars = self.config['k-means']['selected_vars'].split(",")
        except KeyError:
            # let the user view and plot available variables
            self.list_plottable_vars()
            self.__interactive_var_plot()
            # finally ask user which vars to use
            selected_vars = input(
                "\n[>] Type the variables to use in kmeans " +
                "separated by commas (no whitespaces): ")
            selected_vars = selected_vars.split(",")

        print(f"[i] Selected variables for k-means: {selected_vars}")
        self.selected_vars = selected_vars

    def __interactive_var_plot(self):
        try:
            while True:
                # get the name of the variable the user wants to plot
                var_to_plot = input(
                    "[>] Type a variable to plot or Ctrl+C to"
                    " proceed to choosing k-means parameters: "
                )
                self.plot_var(var_to_plot)

        except KeyboardInterrupt:
            # Ctrl+C to exit loop
            pass

    def __set_pipeline(self):
        # construct the data processing pipeline including scaling and k-means
        preprocessor = Pipeline(
            [("scaler", MinMaxScaler())]
        )

        clusterer = Pipeline(
            [("kmeans",
              KMeans(
                  init="k-means++",
                  n_init=self.config['k-means'].getint('n_init'),
                  max_iter=self.config['k-means'].getint('max_iter')
              )
              )]
        )

        self.pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("clusterer", clusterer)
            ])

    def __check_n_init(self):
        self._check_config_option(
            'k-means', 'n_init',
            default=10,
            confirm_msg="[i] Proceeding with n_init = "
        )

    def __check_max_iter(self):
        self._check_config_option(
            'k-means', 'max_iter',
            default=300,
            confirm_msg="[i] Proceeding with max_iter = "
        )

    def __set_features(self):
        # some data manipulation to cast it in a useful xarray form
        ds_tmp = self._ds.copy()
        ds_tmp.subset(variables=self.selected_vars)
        nc_data = ds_tmp.to_xarray()

        # find out if any of the selected vars are 2D
        selected_vars_dims = [len(nc_data[var].shape)
                              for var in self.selected_vars]
        n_spatial_dims = min(selected_vars_dims) - 1

        # to do: make this more concise
        # restrict analysis to ocean surface if some parameters are 2D
        if n_spatial_dims == 2:
            # obtain the data arrays,
            # taking care to slice 3D+time arrays at the surface
            selected_vars_arrays = [nc_data[var].values
                                    if len(nc_data[var].shape) == 3
                                    else nc_data[var].values[:, 0, :, :]
                                    for var in self.selected_vars]
        else:
            # if all data has depth information, use as is
            selected_vars_arrays = [nc_data[var].values
                                    for var in self.selected_vars]

        # take note of the original array shapes before flattening
        self.shape_original = selected_vars_arrays[0].shape
        selected_vars_flat = [array.flatten()
                              for array in selected_vars_arrays]

        # construct the feature vector with missing data
        features = np.ma.masked_array(selected_vars_flat).T

        # convert to pandas dataframe to drop NaN entries, and back to array
        self.df = pd.DataFrame(features)
        self.df.columns = self.selected_vars

    def _make_features(self):
        return np.array(self.df.dropna())


class KMeansMetricsWorkflow(RadioCarbonWorkflow, KMeansWorkflowBase):
    # WORK IN PROGRESS NOT FUNCTIONAL
    def _init_remaining_attrs(self):
        KMeansWorkflowBase._init_remaining_attrs()
        self.__check_max_clusters()

    def __check_max_clusters(self):
        self._check_config_option(
            'k-means', 'max_clusters',
            missing_msg="[!] You have not specified \
            the maximum number of clusters",
            input_msg="[>] Max clusters for metrics: ",
            confirm_msg="[i] Proceeding with max_clusters = "
        )

    def run(self):
        # iterate over different cluster sizes to find "optimal" k
        print("[i] Now computing clustering metrics")
        # initialise empty arrrays for our four tests in search of optimal k
        sse = []
        # silhouettes = []
        calinski_harabasz = []
        davies_bouldin = []

        # run k-means with increasing number of clusters
        for i in tqdm(range(2, self.max_clusters),
                      desc='k-means run: '):
            # set the number of clusters for kmeans
            self.pipe['clusterer']['kmeans'].n_clusters = i

            # actually run kmeans
            self.pipe.fit(self.features)

            # handy variables for computing scores
            scaled_features =\
                self.pipe['preprocessor'].transform(self.features)
            labels = self.pipe['clusterer']['kmeans'].labels_

            # compute various scores for current k
            sse.append(self.pipe['clusterer']['kmeans'].inertia_)
            # silhouettes.append(silhouette_score(scaled_features, labels))
            calinski_harabasz.append(
                calinski_harabasz_score(scaled_features, labels))
            davies_bouldin.append(
                davies_bouldin_score(scaled_features, labels))

        # plot the various scores versus number of clusters
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        ax1.scatter(range(2, self.max_clusters), sse)
        ax1.title.set_text('Sum of squared errors, choose elbow point')

        # ax2.scatter(range(2, max_clusters), silhouettes)
        # ax2.title.set_text('Silhouette Score, higher is better')

        ax3.scatter(range(2, self.max_clusters), calinski_harabasz)
        ax3.title.set_text('Calinski-Harabasz Index, higher is better')

        ax4.scatter(range(2, self.max_clusters), davies_bouldin)
        ax4.title.set_text('Davies-Bouldin Index, lower is better')

        plt.show()


class KMeansWorkflow(RadioCarbonWorkflow, KMeansWorkflowBase):

    def _checkers(self):

        super()._checkers()
        self.__check_palette()
        self.__check_n_clusters()

    def _setters(self):
        super()._setters()
        self.__set_n_clusters()

    def run(self):
        self.set_kmeans_labels()
        self._map_clusters()

    def set_kmeans_labels(self):
        print("[i] Running k-means, please stand by...")
        self.pipe.fit(self._make_features())
        self.__append_labels_to_df()

    def __check_palette(self):
        self._check_config_option(
            'k-means', 'palette',
            missing_msg="[!] You have not specified a palette for the viewer",
            input_msg="[>] Choose a palette (e.g. rainbow|tab10|viridis): ",
            confirm_msg="[i] Proceeding with color palette: "
        )

    def __check_n_clusters(self):
        self._check_config_option(
            'k-means', 'n_clusters',
            missing_msg="[!] You have not specified n_clusters",
            input_msg="[>] Enter number of clusters to use for k-means: ",
            confirm_msg="[i] Proceeding with n_clusters = "
            )

    def __set_n_clusters(self):
        # set the number of clusters of the KMeans object
        self.pipe['clusterer']['kmeans'].n_clusters =\
            self.config['k-means'].getint('n_clusters')

    def __append_labels_to_df(self):
        # add the labels as a new column to our pandas dataframe
        labels = self.pipe['clusterer']['kmeans'].labels_
        self.df.loc[
            self.df.index.isin(self.df.dropna().index),
            'labels'
        ] = labels

    def _map_clusters(self):
        # retrieve the labels column including missing vals as a 1D array
        labels_flat = np.ma.masked_array(self.df['labels'])

        # then reshape to the original 3D/2D form
        labels_shaped = np.reshape(labels_flat, self.shape_original)

        # put some useful info in plot title
        plot_title = f'Kmeans results based on {self.selected_vars}'

        # colors specified in config
        palette = self.config['k-means']['palette']

        # view the labels in 3D
        MultiSliceViewer(volume=labels_shaped, title=plot_title,
                         colorbar=False, legend=True, cmap=palette).show()

# labels_colors = cmap(np.linspace(0, 1, num=n_clusters))

# rescale the centroids back to original data ranges
# centroids = kmeans.cluster_centers_
# centroids = pipe['preprocessor'].inverse_transform(centroids)

# scatter plot of the centroids for each selected variable
# n_params = len(selected_vars)
# centroids_fig, centroids_axes = plt.subplots(1, n_params)
# for i, ax in enumerate(centroids_axes):
#    ax.set_xticks([1], [selected_vars[i]])
#    for j in range(0, centroids.shape[0]):
#        ax.scatter(1, centroids[j, i], color=labels_colors[j])
# centroids_fig.show()
