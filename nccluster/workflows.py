# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from nccluster.base import Workflow
from nccluster.multisliceviewer import MultiSliceViewer
from nccluster.radiocarbon import RadioCarbonWorkflow


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
            self.interactive_var_plot()
            # finally ask user which vars to use
            selected_vars = input(
                "\n[>] Type the variables to use in kmeans " +
                "separated by commas (no whitespaces): ")
            selected_vars = selected_vars.split(",")

        print(f"[i] Selected variables for k-means: {selected_vars}")
        self.selected_vars = selected_vars

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
