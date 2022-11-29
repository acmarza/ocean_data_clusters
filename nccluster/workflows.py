# multi slice viewer adapted from
# https://www.datacamp.com/tutorial/matplotlib-3d-volumetric-data

# kmeans pipeline based on
# https://realpython.com/k-means-clustering-python/
import configparser
# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
#                            silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from nccluster.multisliceviewer import MultiSliceViewer


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
        except Exception:
            print('[i] Config file not passed via commandline')

    def get_nc_files(self):
        # load data from file,
        # asking user to specify a path if not provided in config
        try:
            self.nc_files = self.config['default']['nc_files']
        except (NameError, KeyError):
            print('[i] Data file path not provided')
            self.nc_files = input(
                "[>] Please type the path of the netCDF file to use: ")
        self.nc_files = self.nc_files.split(",")
        print(f"[i] NetCDF file(s): {self.nc_files}")

    def load_ds(self):
        print("[i] Loading data")
        self.ds = nc.DataSet(self.nc_files)

    def preprocess_ds(self):
        print("[i] Preprocessing data")


class KMeansWorkflow(Workflow):

    def run(self):
        self.get_selected_vars()

        if self.get_metrics_mode_bool():
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
        except NameError:
            # will not ask but use the kmeans default
            n_init = 10
        try:
            max_iter = int(self.config['k-means']['max_iter'])
        except NameError:
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

        metrics_mode = self.config['k-means'].getboolean('metrics_mode')
        if metrics_mode is None:
            print("[!] You have not specified whether to run in metrics mode")
            yn = input("[>] Evaluate clustering metrics? (y/n): ")
            metrics_mode = (yn == 'y')
        print(metrics_mode)
        return metrics_mode

    def get_n_clusters(self):
        try:
            n_clusters = int(self.config['k-means']['n_clusters'])
        except (KeyError, ValueError):
            print("[!] You have not specified the number of clusters to use")
            n_clusters = int(input("[>] Enter number of clusters: "))
        self.n_clusters = n_clusters

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
