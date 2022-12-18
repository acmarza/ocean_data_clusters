import atexit
import configparser
# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import pandas as pd
import xarray as xr

from math import log
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
#                            silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, to_sktime_dataset

from nccluster.multisliceviewer import MultiSliceViewer
from nccluster.corrviewer import CorrelationViewer

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
                             isbool=False):
        # convenience function to check if an option is defined in the config;
        # if not, interactively get its value and add to config
        try:
            # ideally the value can be read from config, confirming it exists
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

    def __check_subset(self):
        self._check_config_option(
            'default', 'subset',
            required=False,
            confirm_msg="[i] Data will be subset keeping only variables: "
        )

    def __load_ds(self):
        print("[i] Loading data")

        # get a list of file paths and create the DataSet object
        nc_files = self.config['default']['nc_files'].split(",")
        self.ds = nc.DataSet(nc_files)

    def _preprocess_ds(self):
        # optionally limit analysis to a subset of variables
        if self.config.has_option('default', 'subset'):
            subset = self.config['default']['subset'].split(",")
            print("[i] Subsetting data")
            self.ds.subset(variable=subset)

        # merge datasets if multiple files specified in config
        try:
            print("[i] Merging datasets")
            self.ds.merge()
        except Exception:
            print("[fatal] You have specified multiple netCDF files to use",
                  "but these could not be merged for further analysis.",
                  "Consider subsetting the data as shown in the example",
                  "config, keeping only variables of interest. Or merge your",
                  "datasets externally.")
            exit()

    def _checkers(self):
        self.__check_nc_files()
        self.__check_subset()

    def _setters(self):
        print("[i] All attributes have been initialized")

    def regrid_to_ds(self, tarset_ds):
        # interpolate/extrapolate the data to fit the grid of a target dataset
        self.ds.regrid(tarset_ds)
        # features and time series need to be re-created for new grid
        self._setters()

    def run(self):
        print("[!] Nothing to run. Did you forget to override self.run()?")

    def list_plottable_vars(self):
        # get an alphabetical list of plottable variables
        plottable_vars = self.ds.variables

        # easy reference to a dataframe containing useful info on variables
        df = self.ds.contents

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
            ds_tmp = self.ds.copy()
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
        if 'dic' in self.ds.variables and\
                'di14c' in self.ds.variables and\
                'dc14' not in self.ds.variables:
            self.__compute_dc14()

        # compute radiocarbon ages if dc14 exists in dataset
        if 'dc14' in self.ds.variables:
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
            self.ds.rename({value: key})
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
        self.ds.assign(local_age=lambda x: -mean_radio_life*log(
                           (x.dc14/1000 + 1)/(atm_dc14/1000 + 1))
                       )
        print("[i] Converted dc14 to age")

    def __compute_dc14(self):
        # from dic and di14c
        self.ds.assign(dc14=lambda x:
                       (x.di14c/x.dic-1)*1000)
        print("[i] Computed dc14 from di14c and dic")


class TimeseriesWorkflowBase(RadioCarbonWorkflow):
    '''Base class for workflows involving time series.'''
    def _checkers(self):
        super()._checkers()
        self.__check_plot_all_ts_bool()

    def _setters(self):
        self.mask = None
        self.__set_age_array()

    def run(self):
        # this base class simply plots all the time series if asked to
        if self.config['timeseries'].getboolean('plot_all_ts'):
            self.plot_all_ts()

    def __set_age_array(self):
        # some  manipulation to isolate age data and cast it into a useful form
        # make a copy of the original dataset to subset to one variable
        ds_tmp = self.ds.copy()
        ds_tmp.subset(variable='local_age')

        # convert to xarray and extract the numpy array of values
        age_xr = ds_tmp.to_xarray()
        age_array = age_xr['local_age'].values

        self.age_array = age_array

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
        ts_array = np.ma.masked_array(self.age_array[:, 0], self.mask)
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


class CorrelationWorkflow(TimeseriesWorkflowBase):
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
        TimeseriesWorkflowBase.run(self)

        # keyword arguments to be passed to CorrelationViewer
        kwargs = self.config['correlation']

        # initialise and show CorrelationViewer
        self.corrviewer = CorrelationViewer(self.age_array, 'R-ages', **kwargs)
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


class TSClusteringWorkflow(TimeseriesWorkflowBase):
    '''A workflow to find clusters in the surface ocean based on the
    radiocarbon age time series at each grid point.'''
    def _checkers(self):
        super()._checkers()
        self.__check_n_clusters()
        self.__check_clustering_method()
        self.__check_scaling_bool()

    def _setters(self):
        super()._setters()
        self._set_ts()

    def run(self):
        # run the parent workflow (plotting all time series)
        TimeseriesWorkflowBase.run(self)

        # get the model with label assignments and barycenters
        self.fit_model()

        # plot and show results
        self.view_results()

        plt.show()

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

    def fit_model(self):
        # define the keyword arguments to pass to the model
        kwargs = {
            'n_clusters': self.config['timeseries'].getint('n_clusters'),
            # 'max_iter': 10,
            'metric': 'euclidean'

        }

        dataset = self.ts

        # optionally scale the data (aids in shape detection
        # but loses amplitude information)
        if self.config['timeseries'].getboolean('scaling'):
            print("[i] Normalising time series")
            dataset = TimeSeriesScalerMeanVariance().fit_transform(dataset)

        # initialise model using desired clustering method
        if self.config['timeseries']['method'] == 'k-means':
            print("[i] Initialising k-means model")
            self.model = TimeSeriesKMeans(**kwargs)
        elif self.config['timeseries']['method'] == 'k-medoids':
            print("[i] Initialising k-medoids model")
            self.model = TimeSeriesKMedoids(**kwargs)
            dataset = to_sktime_dataset(dataset)

        print("[i] Fitting model, please stand by")

        # actually fit the model
        self.model.fit(dataset)

    def view_results(self):

        self.fig = plt.figure()
        self._plot_ts_clusters()
        self._map_clusters()
        plt.show()

    def _plot_ts_clusters(self):
        # get predictions for our time series from trained model
        # i.e. to which cluster each time series belongs
        labels = self.model.labels_
        n_clusters = self.model.n_clusters

        # for each cluster/label
        for label in range(n_clusters):
            # create a subplot in a table with n_cluster rows and 1 column
            # this subplot is number yi+1 because we're counting from 0
            ax = self.fig.add_subplot(n_clusters, 2, 2 * label + 1)
            # for every time series that has been assigned label yi
            for xx in self.ts[labels == label]:
                # plot with a thin transparent line
                ax.plot(xx.ravel(), "k-", alpha=.2)
            # plot the cluster barycenter
            ax.plot(self.model.cluster_centers_[label].ravel(), "r-")
            # label the cluster
            ax.set_title(f'Cluster {label}')

    def _map_clusters(self):
        # get the 2D labels array
        labels_shaped = self._make_labels_shaped()

        # view the clusters on a map
        ax = self.fig.add_subplot(122)
        ax.imshow(labels_shaped, origin='lower')

    def _make_labels_shaped(self):
        # get the timeseries as a dataframe and append labels to non-empty rows
        df = self._make_df()
        df.loc[
            df.index.isin(df.dropna().index),
            'labels'] = self.model.labels_

        # convert to array of labels and reshape into 2D for map
        labels_flat = np.ma.masked_array(df['labels'])
        _, _, y, x = self.age_array.shape
        labels_shaped = np.reshape(labels_flat, [y, x])
        return labels_shaped

    def make_labels_data_array(self,
                               long_name='time series clustering results'):
        # get the raw labels array
        labels_shaped = self._make_labels_shaped()

        # copy the coords of the original dataset, but keep only x and y
        all_coords = self.ds.to_xarray().coords
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

        # create the data array from our labels and
        # the x-y coords copied from the original dataset
        data_array = xr.DataArray(data=labels_shaped,
                                  coords=coords,
                                  name='labels',
                                  attrs={'long_name': long_name}
                                  )
        return data_array

    def save_labels_data_array(self, filename, long_name):
        da = self.make_labels_data_array(long_name)
        da.to_netcdf(filename)
        print(f"[i] Saved labels to {filename}")


class TwoStepTimeSeriesClusterer(TSClusteringWorkflow):

    def run(self):
        print("[i] This workflow will override the config setting for scaling")
        self.config['timeseries']['scaling'] = 'True'
        self.fit_model()
        # self.view_results()
        self.config['timeseries']['scaling'] = 'False'
        self.form_subclusters()

    def form_subclusters(self):
        if self.config['timeseries'].getboolean('scaling'):
            print("[!] Scaling has not been turned off for subclusters")

        labels = self._make_labels_shaped()
        n_clusters = int(self.config['timeseries']['n_clusters'])
        time_steps, _, y, x = self.age_array.shape

        labels_two_step = np.full((y, x, 2), np.nan)
        labels_two_step[:, :, 0] = labels

        subclust_sizes = []

        # for each shape cluster construct a time series with just those points
        for label in range(0, n_clusters):

            # mask all the points not belonging to this cluster
            cluster_mask = (labels != label)

            # repeat the mask for each time step
            cluster_mask = np.array([cluster_mask])
            cluster_mask = np.repeat(cluster_mask, time_steps, axis=0)

            # set the cluster mask and re-build time series object
            self.mask = cluster_mask

            self._set_ts()
            # run algorithm again on these points, without normalisation
            self.fit_model()

            # view results
            # self.view_results()

            sublabels = self._make_labels_shaped()
            subclust_sizes.append(np.nanmax(sublabels) + 1)

            for arg in np.argwhere(~np.isnan(sublabels)):
                yi, xi = arg
                labels_two_step[yi, xi, 1] = sublabels[yi, xi]

        subclusters_map = np.full((y, x), np.nan)
        for arg in np.argwhere(~np.isnan(labels)):
            yi, xi = arg
            label, sublabel = labels_two_step[yi, xi]
            # prev_clust_sizes = subclust_sizes[:int(label)-1]
            # offset = np.sum(prev_clust_sizes)
            interval = 0.4
            offset = sublabel * (interval / (subclust_sizes[int(label)]-1))

            subclusters_map[yi, xi] = label - interval/2 + offset

        plt.figure()
        plt.imshow(subclusters_map, origin='lower')
        plt.show()


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
        ds_tmp = self.ds.copy()
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
