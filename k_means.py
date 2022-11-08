# multi slice viewer adapted from
# https://www.datacamp.com/tutorial/matplotlib-3d-volumetric-data

# kmeans pipeline based on
# https://realpython.com/k-means-clustering-python/
import argparse
import configparser
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score,\
                            davies_bouldin_score,\
                            silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def multi_slice_viewer(volume, title, colorbar=True, view="z"):
    """Move through 2D slices of a 4D data array using the arrow keys.
    Assumes array is given with shape (time, z, y, x)"""

    # if data has only 3 dimensions; assume it is missing the depth axis
    # reshape into 4D array with single depth level
    if len(volume.shape) == 3:
        t, y, x = volume.shape
        volume = np.reshape(volume, (t, 1, y, x))

    # initialise the figure and handy vars, showing the first time/depth slice
    # and wait for key presses
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = [0, 0]
    ax.pos = ax.imshow(
        volume[ax.index[0], ax.index[1]],
        cmap='rainbow',
    )

    # default is to draw colorbar
    if colorbar:
        fig.colorbar(ax.pos, ax=ax)

    # initialise view perpendicular to z
    change_view(ax, 'z')

    # tell figure to wait for key events
    fig.canvas.mpl_connect('key_press_event', process_key)

    # title specified in function call
    plt.suptitle(title)

    # finally show the figure
    plt.show()


def change_view(ax, view):

    # if view not specified, default to z
    try:
        ax.view
    except AttributeError:
        ax.view = 'z'

    # define permutations between axes
    z_order = [0, 1, 2, 3]
    x_order = [0, 2, 3, 1]
    y_order = [0, 2, 1, 3]
    permutation_dict = {
        'x': x_order,
        'y': y_order,
        'z': z_order
    }

    # move the axes to slice through the fist two (time and a space axis)
    ax.volume = np.moveaxis(
        ax.volume,
        permutation_dict[ax.view],
        permutation_dict[view]
    )

    # make sure plot doesn't show upside down
    origin_dict = {
        'x': 'upper',
        'y': 'upper',
        'z': 'lower'
    }
    ax.pos.origin = origin_dict[view]

    # assign the new view to the figure to be used when updating info text
    ax.view = view

    # reset space slice to zero
    change_slice(ax, 1, -ax.index[1])


def process_key(event):
    """Define action to execute when certain keys are pressed."""
    # get a handle on the axis
    fig = event.canvas.figure
    ax = fig.axes[0]

    # arrow key navigation
    if event.key == 'left':
        change_slice(ax, 0, -1)
    if event.key == 'right':
        change_slice(ax, 0, 1)
    if event.key == 'up':
        change_slice(ax, 1, -1)
    if event.key == 'down':
        change_slice(ax, 1, 1)
    if event.key == 'x':
        change_view(ax, 'x')
    if event.key == 'y':
        change_view(ax, 'y')
    if event.key == 'z':
        change_view(ax, 'z')
    # update the plot
    fig.canvas.draw()


def update_suptitle_text(ax):
    time_step, depth_step = ax.index
    max_time_steps, max_depth_steps, _, _ = ax.volume.shape
    ax.title.set_text(f"time: {time_step+1}/{max_time_steps}\n"
                      f"{ax.view}: {depth_step+1}/{max_depth_steps}")


def change_slice(ax, dimension, amount):
    """Change the index of the current slice by amount for given dimension."""
    # get a handle on the 4D data array
    volume = ax.volume
    # increment index (wrap around with modulo)
    ax.index[dimension] = (ax.index[dimension] + amount)\
        % volume.shape[dimension]
    # set the slice to view based on the new index
    ax.images[0].set_array(volume[ax.index[0], ax.index[1]])
    update_suptitle_text(ax)


# parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from,\
                     if parameters not supplied interactively")
args = parser.parse_args()

# parse config file if present
try:
    config = configparser.ConfigParser()
    config.read(args.config)
except Exception:
    print('[i] Config file not passed via commandline')

# load data from file, asking user to specify a path if not provided in config
try:
    nc_file = config['default']['nc_file']
except NameError:
    print('[i] Data file path not provided in config')
    nc_file = input("[>] Please type the path of the netCDF file to use: ")
print(f"[i] NetCDF file: {nc_file}")

# load data
nc_data = nc.Dataset(nc_file)

# get parameters to run k-means for from file, or interactively
try:
    selected_vars = config['default']['selected_vars'].split(",")
except KeyError:

    # get an alphabetical list of plottable variables
    plottable_vars = list(nc_data.variables.keys())
    plottable_vars.sort()

    # list plottable variables for the user to inspect
    for i, var in enumerate(plottable_vars):
        # print in a nice format if possible
        try:
            long_name = nc_data.variables[var].long_name
            dims = nc_data.variables[var].dimensions
            print(f"{i}. {var}\n\t{dims}\n\t{long_name}")
        except Exception:
            print(f"{i}. {var}\n{nc_data[var]}")

    # offer to visualise a variable of choice
    view_var_plots = input(
        "[>] Would you like to visualise any of these variables? (y/n): ")
    if view_var_plots.lower() == 'y':
        try:
            while True:
                # get the name of the variable the user wants to plot
                string_var_to_plot = input(
                    "[>] Type which variable to plot or Ctrl+C to"
                    " proceed to choosing k-means parameters: "
                )

                try:
                    # get the data as an array
                    var_to_plot = nc_data[string_var_to_plot]
                    data_to_plot = var_to_plot.__array__()
                    # view over time and depth
                    plot_title = string_var_to_plot + " ("\
                        + var_to_plot.long_name + ") " + var_to_plot.units
                    multi_slice_viewer(data_to_plot, plot_title, view='x')
                except IndexError:
                    print("[!] {string_var_to_plot} not found; check spelling")
        except KeyboardInterrupt:
            # the user is done viewing plots, continue to kmeans routine
            pass

    # finally ask user which vars to use
    selected_vars = input(
        "[>] Type the variables to use in kmeans separated by spaces: ")
    selected_vars = selected_vars.split(" ")
print(f"[i] Selected variables for k-means: {selected_vars}")

# restrict analysis to ocean surface if some parameters are 2D-only (+time)
selected_vars_dims = [len(nc_data[var].shape) for var in selected_vars]
n_spatial_dims = min(selected_vars_dims) - 1

# construct the data processing pipeline including scaling and k-means
preprocessor = Pipeline(
    [("scaler", MinMaxScaler())]
)

try:
    n_init = int(config['default']['n_init'])
except NameError:
    # will not ask but use the kmeans default
    n_init = 10
try:
    max_iter = int(config['default']['max_iter'])
except NameError:
    # will not ask but use the kmeans default
    max_iter = 300
print(f"[i] K-means hyperparameters: n_init = {n_init}, max_iter = {max_iter}")
clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                init="k-means++",
                n_init=n_init,
                max_iter=max_iter
            )
        )
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)


# if the number of clusters is not supplied, will evaluate clustering
# performance over a range of k with different metrics
# otherwise proceed to k-means
try:
    optimal_k = config['default']['optimal_k']
    metrics_mode = False
except KeyError:
    try:
        optimal_k = int(input(
            "[>] Enter number of clusters now, or leave blank"
            " to compute clustering metrics for a range of k: "
             ))
        metrics_mode = False
    except ValueError:
        print("[i] Now computing clustering metrics")
        print("[i] When finished, re-run script with your choice of k")
        metrics_mode = True
        try:
            max_clusters = int(config['default']['max_clusters'])
        except KeyError:
            max_clusters = int(input("[>] Max clusters: "))

if n_spatial_dims == 2:
    # flatten the data arrays taking care to slice 4D ones at the surface
    selected_vars_arrays = [nc_data[var].__array__()
                            if len(nc_data[var].shape) == 3
                            else nc_data[var].__array__()[:, 0, :, :]
                            for var in selected_vars]
else:
    # if all data has depth information, use as is
    selected_vars_arrays = [nc_data[var].__array__()
                            for var in selected_vars]

# take note of the original array shapes before flattening
shape_original = selected_vars_arrays[0].shape
selected_vars_flat = [array.flatten()
                      for array in selected_vars_arrays]

# construct the feature vector with missing data
# do not use np.array because it fills numerical values and confuses scaling
features = np.ma.masked_array(selected_vars_flat).T

# convert to pandas dataframe to drop NaN entries, and back to array
df = pd.DataFrame(features)
df.columns = selected_vars
features = np.array(df.dropna())

# iterate over different cluster sizes to find optimal k, if not specified
if metrics_mode:
    # initialise empty arrrays for our four tests in search of optimal k
    sse = []
    silhouettes = []
    calinski_harabasz = []
    davies_bouldin = []

    # run k-means with increasing number of clusters
    for i in tqdm(range(2, max_clusters), desc='k-means run'):
        # set the number of clusters for kmeans
        pipe['clusterer']['kmeans'].n_clusters = i

        # actually run kmeans
        pipe.fit(features)

        # handy variables for computing scores
        scaled_features = pipe['preprocessor'].transform(features)
        labels = pipe['clusterer']['kmeans'].labels_

        # compute various scores for current k
        sse.append(pipe['clusterer']['kmeans'].inertia_)
        silhouettes.append(silhouette_score(scaled_features, labels))
        calinski_harabasz.append(
            calinski_harabasz_score(scaled_features, labels))
        davies_bouldin.append(
            davies_bouldin_score(scaled_features, labels))

    # plot the various scores versus number of clusters
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.scatter(range(2, max_clusters), sse)
    ax1.title.set_text('Sum of squared errors, choose elbow point')

    ax2.scatter(range(2, max_clusters), silhouettes)
    ax2.title.set_text('Silhouette Score, higher is better')

    ax3.scatter(range(2, max_clusters), calinski_harabasz)
    ax3.title.set_text('Calinski-Harabasz Index, higher is better')

    ax4.scatter(range(2, max_clusters), davies_bouldin)
    ax4.title.set_text('Davies-Bouldin Index, lower is better')

    plt.show()

else:
    # metrics mode off
    # run k-means with chosen k
    kmeans = pipe['clusterer']['kmeans']
    kmeans.n_clusters = optimal_k
    pipe.fit(features)
    labels = kmeans.labels_

    # now to reshape the 1D array of labels into a plottable 2D form
    # first add the labels as a new column  to our pandas dataframe
    df.loc[
        df.index.isin(df.dropna().index),
        'labels'
    ] = labels

    # then retrieve the labels column including missing vals as a 1D array
    labels_flat = np.ma.masked_array(df['labels'])

    # then reshape to the original 3D/fD form
    labels_shaped = np.reshape(labels_flat, shape_original)

    # map out the clusters each with its own color
    plot_title =\
        f"Kmeans result with {optimal_k} clusters based on {selected_vars}"
    multi_slice_viewer(labels_shaped,
                       title=plot_title,
                       colorbar=False
                       )
# to do
#   show info about each cluster on hover, based on centroids
#       look into plotly or just write that information elsewhere
