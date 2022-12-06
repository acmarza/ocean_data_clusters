import configparser
# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import pandas as pd
import pickle

from math import log
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
#                            silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from nccluster.multisliceviewer import MultiSliceViewer
from nccluster.corrviewer import CorrelationViewer


class Workflow:

    def __init__(self, config_path=None):

        self.config_path = config_path

        self.read_config_file()
        self.get_nc_files()
        self.load_ds()
        self.preprocess_ds()

    def read_config_file(self):
        # parse config file if present
        try:
            self.config = configparser.ConfigParser()
            self.config.read(self.config_path)
            print(f"[i] Read config: {self.config_path}")
        except FileNotFoundError:
            print('[!] Config file not passed via commandline')

    def get_nc_files(self):
        # load netCDF data from file,
        # asking user to specify the path(s) if not provided in config
        try:
            self.nc_files = self.config['default']['nc_files']
        except (NameError, KeyError):
            print('[i] Data file path not provided')
            self.nc_files = input(
                "[>] Please type the path of the netCDF file to use: ")
        self.nc_files = self.nc_files.split(",")
        print(f"[i] NetCDF file(s): {self.nc_files}")

    def load_ds(self):
        # load in data, optionally limiting analysis to a subset of variables
        print("[i] Loading data")
        self.ds = nc.DataSet(self.nc_files)
        try:
            subset = self.config['default']['subset'].split(",")
            self.ds.subset(variable=subset)
        except (NameError, KeyError):
            pass
        try:
            self.ds.merge()
        except Exception:
            print("[fatal] You have specified multiple netCDF files to use",
                  "but these could not be merged for further analysis.",
                  "Consider subsetting the data as shown in the example",
                  "config, keeping only variables of interest. Or merge your",
                  "datasets externally.")
            exit()

    def preprocess_ds(self):
        print("[!] No preprocessing routine defined")

    def run(self):
        print("[!] Nothing to run. Did you forget to override self.run()?")


class RadioCarbonWorkflow(Workflow):

    def preprocess_ds(self):
        # for computing radiocarbon ages, will use nctoolkit's DataSet.assign,
        # however this will require knowing the variable names in advance;
        # to avoid confusion, rename the variables used in this computation
        # to something simple and consistent
        print("[i] Preprocessing data")
        self.construct_rename_dict(['dc14', 'dic', 'di14c'])
        self.rename_vars()

        if 'dic' in self.ds.variables and\
                'di14c' in self.ds.variables and\
                'dc14' not in self.ds.variables:
            self.compute_dc14()

        if 'dc14' in self.ds.variables:
            self.get_mean_radiocarbon_lifetime()
            self.compute_local_age()

    def construct_rename_dict(self, vars):
        # get the name of each variable as it appears in the dataset
        # from config or interactively
        self.rename_dict = {}
        for var in vars:
            try:
                self.rename_dict[var] = self.config['radiocarbon'][var]
            except NameError:
                print(f"[!] Name of the {var} variable was not provided")
                self.rename_dict[var] = input("[>] Enter {var} variable name \
                                        as it appears in the dataset: ")
            except KeyError:
                pass

    def rename_vars(self):
        # rename variables to be used in calculations for easy reference
        for key, value in self.rename_dict.items():
            self.ds.rename({value: key})
            print(f"[i] Renamed variable {value} to {key}")
        self.ds.run()

    def get_mean_radiocarbon_lifetime(self):
        # get the mean radiocarbon lifetime from config or interactively
        try:
            self.mean_radio_life =\
                int(self.config['radiocarbon']['mean_radiocarbon_lifetime'])
        except (NameError, KeyError):
            print("[!] Mean lifetime of radiocarbon was not provided")
            self.mean_radio_life = int(input(
                "[>] Enter mean radiocarbon lifetime \
                                        (Cambridge=8267, Libby=8033): "))

    def compute_local_age(self):
        # using mean radiocarbon lifetime and dc14
        self.ds.assign(local_age=lambda x:
                       -self.mean_radio_life*log((1000+x.dc14)/1000))
        print("[i] Converted dc14 to age",
              f"using mean radioC lifetime {self.mean_radio_life}")

    def compute_dc14(self):
        # from dic and di14c
        self.ds.assign(dc14=lambda x:
                       (x.di14c/x.dic-1)*1000)
        self.ds.run()
        print("[i] Computed dc14 from di14c and dic")


class TimeseriesWorkflowBase(RadioCarbonWorkflow):
    def __init__(self, config_path):
        print("[i] Starting timeseries analysis workflow")

        # call the parent init
        RadioCarbonWorkflow.__init__(self, config_path)

        # extend parent init by computing and reading in additional attributes
        self.get_ts_array()
        self.get_plot_all_ts_bool()

    def run(self):
        # this base class simply plots all the timeseries if asked to
        if self.plot_all_ts_bool:
            self.plot_all_ts()

    def get_ts_array(self):
        # some data manipulation to cast it into a useful form
        ds_tmp = self.ds.copy()
        ds_tmp.subset(variable='local_age')
        age_xr = ds_tmp.to_xarray()
        age_array = age_xr['local_age'].__array__()
        # offset ages to make more sense (else they'd be negative)
        min_age = np.nanmin(age_array)
        age_array -= min_age
        self.age_array = age_array
        # produce an array containing the R-age time series at each grid point
        t, z, y, x = self.shape_original = age_array.shape

        # the result is an array of timeseries
        self.ts_array = np.reshape(age_array[:, 0], [t, x*y]).T

    def get_plot_all_ts_bool(self):
        # find out from config or interactively whether to show
        # all time series on one plot
        try:
            self.plot_all_ts_bool =\
                self.config['timeseries'].getboolean('plot_all_ts')
        except (NameError, KeyError):
            yn = input("[>] Show a plot of all the R-age timeseries? (y/n): ")
            self.plot_all_ts_bool = (yn == 'y')

    def plot_all_ts(self):
        # plot evolution of every grid point over time
        # i.e. plot all the timeseries on one figure
        fig = plt.figure()
        ax = fig.add_subplot()
        for ts in tqdm(self.ts_array,
                       desc="[i] Plotting combined time series"):
            ax.plot(range(0, len(ts)), ts)
        ax.set_xlabel('time step')
        ax.set_ylabel('age')
        ax.set_title('age over time')


class CorrelationWorkflow(TimeseriesWorkflowBase):

    def __init__(self, config_path):
        # call parent init
        TimeseriesWorkflowBase.__init__(self, config_path)

        # extend parent init by sorting out whether we need to use
        # p-values and savefiles
        self.get_pvalues_bool()

        # keyword arguments to be passed to CorrelationViewer
        self.kwargs = {
            'volume':  self.age_array,
            'title': "R-ages",
            'pvalues': self.pvalues
        }
        if self.pvalues:
            self.get_corr_mat_file()
            self.get_corr_mat_file()
            self.kwargs['corr_mat_file'] = self.corr_mat_file
            self.kwargs['pval_mat_file'] = self.pval_mat_file

    def run(self):
        # run parent workflow (timeseries plot)
        TimeseriesWorkflowBase.run(self)

        # initialise and show CorrelationViewer
        self.corrviewer = CorrelationViewer(**self.kwargs)
        plt.show()

    def get_pvalues_bool(self):
        # find out whether to mask using p-values, from config or interactively
        try:
            self.pvalues = self.config['correlation'].getboolean('pvalues')
        except (NameError, KeyError):
            yn = input("[>] Mask out grid points with insignificant \
                    ( p > 0.05 ) correlation? (y/n): ")
            self.pvalues = (yn == 'y')

    def get_corr_mat_file(self):
        # get the savefile path for the correlation matrix
        try:
            self.corr_mat_file = self.config['correlation']['corr_mat_file']
        except (KeyError, NameError):
            print("[!] Correlation matrix save file not provided")
            self.corr_mat_file = input(
                "[>] Enter file path to save/read correlation matrix now: "
            )

    def get_pval_mat_file(self):
        # get the savefile path for the p-value matrix
        try:
            self.pval_mat_file = self.config['correlation']['pval_mat_file']
        except (KeyError, NameError):
            print("[!] P-value matrix save file not provided")
            self.pval_mat_file = input(
                "[>] Enter file path to save/read p-value matrix now: "
            )


class TSClusteringWorkflow(TimeseriesWorkflowBase):
    def __init__(self, config_path):
        # run the parent init
        TimeseriesWorkflowBase.__init__(self, config_path)

        # compute and read additional attributes
        self.get_ts()
        self.get_n_clusters()
        self.get_model_save_path()

    def run(self):
        # run the parent workflow (plotting all timeseries)
        TimeseriesWorkflowBase.run(self)

        # read model from file or train a new one
        try:
            self.read_model_save_file()
        except FileNotFoundError:
            self.train_new_model()
            self.save_model()

        # plot and show results
        self.plot_ts_clusters()
        self.map_clusters()
        plt.show()

    def get_ts(self):
        # convert array of time series to pandas dataframe to drop NaN entries,
        # then back to array, then to time series dataset for use with
        # tslearn
        self.df = pd.DataFrame(self.ts_array)
        ts = np.array(self.df.dropna())
        self.ts = to_time_series_dataset(ts)

    def get_n_clusters(self):
        # get the number of clusters for the KMeans model
        try:
            self.n_clusters = int(self.config['timeseries']['n_clusters'])
        except (KeyError, ValueError):
            print("[!] You have not specified the number of clusters to use")
            self.n_clusters = int(input("[>] Enter number of clusters for \
                                        timeseries k-means: "))

    def get_model_save_path(self):
        # get the path of the pickle save file from config or interactively
        try:
            self.model_save_path = self.config['timeseries']['pickle']
        except (KeyError, NameError):
            print("[!] Model save file not provided")
            self.model_save_path = input(
                "[>] Enter a file path to save/read pickled model now: "
            )

    def read_model_save_file(self):
        # load in the trained model
        with open(self.model_save_path, 'rb') as file:
            self.km = pickle.load(file)
            print(f"[i] Read in {self.model_save_path}")

    def train_new_model(self):
        # initialise model
        self.km = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric='euclidean',
            max_iter=10,
            n_jobs=-1
        )
        print("[i] Fitting k-means model, please stand by")

        # actually fit the model
        self.km.fit(self.ts)

    def save_model(self):
        # write k-means model object to file
        with open(self.model_save_path, 'wb') as file:
            pickle.dump(self.km, file)
            print(f"[i] Saved model to {self.model_save_path}")

    def plot_ts_clusters(self):
        # get predictions for our timeseries from trained model
        # i.e. to which cluster each timeseries belongs
        # consider changing this to labels
        # y_pred = self.km.predict(self.ts)
        y_pred = self.km.labels_

        # plot each cluster members and their barycenter
        # initialise figure
        clusters_fig = plt.figure()

        # for each cluster/label
        for yi in range(self.n_clusters):
            # create a subplot in a table with n_cluster rows and 1 column
            # this subplot is number yi+1 because we're counting from 0
            plt.subplot(self.n_clusters, 1, yi + 1)
            # for every timeseries that has been assigned label yi
            for xx in self.ts[y_pred == yi]:
                # plot with a thin transparent line
                plt.plot(xx.ravel(), "k-", alpha=.2)
            # plot the cluster barycenter
            plt.plot(self.km.cluster_centers_[yi].ravel(), "r-")
            # label the cluster
            plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
        # add a title at the top of the figure
        clusters_fig.suptitle("k-means results")
        # finally show the figure
        clusters_fig.show()

    def map_clusters(self):
        # assign predicted labels to the original dataframe
        self.df.loc[
            self.df.index.isin(self.df.dropna().index),
            'labels'] = self.km.labels_

        # convert to array of labels and reshape into 2D for map
        labels_flat = np.ma.masked_array(self.df['labels'])
        _, _, y, x = self.shape_original
        labels_shaped = np.reshape(labels_flat, [y, x])

        # finally view the clusters on a map
        clusters_fig = plt.figure()
        ax = clusters_fig.add_subplot()
        ax.imshow(labels_shaped, origin='lower')
        clusters_fig.show()


class KMeansWorkflow(RadioCarbonWorkflow):

    def run(self):
        self.get_selected_vars()
        self.get_metrics_mode_bool()

        if self.metrics_mode:
            self.run_metrics()
        else:
            self.run_kmeans()

    def run_kmeans(self):
        self.construct_pipeline()
        self.construct_features()
        self.get_n_clusters()

        kmeans = self.pipe['clusterer']['kmeans']
        kmeans.n_clusters = self.n_clusters
        print("[i] Running k-means, please stand by...")
        self.pipe.fit(self.features)
        self.labels = kmeans.labels_
        self.labels += 1

        # now to reshape the 1D array of labels into a plottable 2D form
        # first add the labels as a new column  to our pandas dataframe
        self.df.loc[
            self.df.index.isin(self.df.dropna().index),
            'labels'
        ] = self.labels

        # then retrieve the labels column including missing vals as a 1D array
        labels_flat = np.ma.masked_array(self.df['labels'])

        # then reshape to the original 3D/fD form
        labels_shaped = np.reshape(labels_flat, self.shape_original)

        # map out the clusters each with its own color
        plot_title = f"Kmeans results with {self.n_clusters} clusters"
        plot_title += f" based on {self.selected_vars}"

        # read color palette from file or default to rainbow
        try:
            palette = config['default']['palette']
        except (NameError, KeyError):
            palette = 'rainbow'
        # cmap = cm.get_cmap(palette)

        MultiSliceViewer(labels_shaped, title=plot_title, colorbar=False,
                         legend=True, cmap=palette).show()

    def get_selected_vars(self):
        # get parameters to run k-means for from file, or interactively
        try:
            selected_vars = self.config['k-means']['selected_vars'].split(",")
        except KeyError:
            self.list_plottable_vars()
            self.interactive_var_plot()
            # finally ask user which vars to use
            selected_vars = input(
                "\n[>] Type the variables to use in kmeans " +
                "separated by commas: ")
            selected_vars = selected_vars.split(",")

        print(f"[i] Selected variables for k-means: {selected_vars}")
        self.selected_vars = selected_vars

    def list_plottable_vars(self):
        # get an alphabetical list of plottable variables
        plottable_vars = self.ds.variables
        df = self.ds.contents

        # list plottable variables for the user to inspect
        for i, var in enumerate(plottable_vars):
            # print in a nice format if possible
            try:
                long_name = df.loc[df['variable'] == var].long_name.values[0]
                print(f"{i}. {var}\n\t{long_name}")
            except Exception:
                print(f"{i}. {var}")

    def plot_var(self, var_to_plot):
        try:
            # get the data as an array
            ds_tmp = self.ds.copy()
            ds_tmp.subset(variables=var_to_plot)
            var_xr = ds_tmp.to_xarray()
            data_to_plot = var_xr[var_to_plot].__array__()

            # add more info in plot title if possible
            try:
                plot_title = var_to_plot + " ("\
                    + var_xr[var_to_plot].long_name + ") "\
                    + var_xr[var_to_plot].units
            except AttributeError:
                plot_title = var_to_plot

            MultiSliceViewer(data_to_plot, plot_title).show()

        except IndexError:
            print(f"[!] {var_to_plot} not found; check spelling")

    def interactive_var_plot(self):
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

    def construct_pipeline(self):

        # construct the data processing pipeline including scaling and k-means
        preprocessor = Pipeline(
            [("scaler", MinMaxScaler())]
        )

        try:
            n_init = int(self.config['k-means']['n_init'])
        except (KeyError, NameError):
            # will not ask but use the kmeans default
            n_init = 10
        try:
            max_iter = int(self.config['k-means']['max_iter'])
        except (KeyError, NameError):
            # will not ask but use the kmeans default
            max_iter = 300
        print("[i] K-means hyperparameters: ",
              f"n_init = {n_init}, max_iter = {max_iter}")
        clusterer = Pipeline([("kmeans", KMeans(
                        init="k-means++",
                        n_init=n_init,
                        max_iter=max_iter
                    )
                )])

        self.pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("clusterer", clusterer)
            ])

    def run_metrics(self):

        self.construct_pipeline()
        self.construct_features()
        self.get_max_clusters()

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

    def get_metrics_mode_bool(self):

        try:
            self.metrics_mode =\
                self.config['k-means'].getboolean('metrics_mode')
        except (KeyError, NameError):
            print("[!] You have not specified whether to run in metrics mode")
            yn = input("[>] Evaluate clustering metrics? (y/n): ")
            self.metrics_mode = (yn == 'y')

    def get_n_clusters(self):
        try:
            self.n_clusters = int(self.config['k-means']['n_clusters'])
        except (KeyError, ValueError):
            print("[!] You have not specified the number of clusters to use")
            self.n_clusters = int(input("[>] Enter number of clusters: "))

    def get_max_clusters(self):
        try:
            max_clusters = int(self.config['k-means']['max_clusters'])
        except KeyError:
            print("[!] You have not specified the maximum number of clusters")
            max_clusters = int(input("[>] Max clusters for metrics: "))
        self.max_clusters = max_clusters

    def construct_features(self):
        selected_vars = self.selected_vars
        ds_tmp = self.ds.copy()
        ds_tmp.subset(variables=selected_vars)
        nc_data = ds_tmp.to_xarray()

        selected_vars_dims = [len(nc_data[var].shape)
                              for var in selected_vars]
        n_spatial_dims = min(selected_vars_dims) - 1

        # restrict analysis to ocean surface if some parameters are x-y only
        if n_spatial_dims == 2:
            # flatten the data arrays,
            # taking care to slice 4D ones at the surface
            selected_vars_arrays = [nc_data[var].__array__()
                                    if len(nc_data[var].shape) == 3
                                    else nc_data[var].__array__()[:, 0, :, :]
                                    for var in selected_vars]
        else:
            # if all data has depth information, use as is
            selected_vars_arrays = [nc_data[var].__array__()
                                    for var in selected_vars]

        # take note of the original array shapes before flattening
        self.shape_original = selected_vars_arrays[0].shape

        selected_vars_flat = [array.flatten()
                              for array in selected_vars_arrays]

        # construct the feature vector with missing data
        features = np.ma.masked_array(selected_vars_flat).T

        # convert to pandas dataframe to drop NaN entries, and back to array
        self.df = pd.DataFrame(features)
        self.df.columns = selected_vars
        self.features = np.array(self.df.dropna())

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
