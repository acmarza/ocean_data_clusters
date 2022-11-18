import argparse
import configparser
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import pandas as pd
import pickle

from math import log
from stats_viewer import CorrelationViewer
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

print("[i] Starting R-ages analysis workflow")

# define and parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from, \
                    if parameters not supplied interactively")
args = parser.parse_args()

# parse config file if present
try:
    config = configparser.ConfigParser()
    config.read(args.config)
    print(f"[i] Read config: {args.config}")
except Exception:
    print('[!] Config file not passed via commandline')

# load data from file, asking user to specify a path if not provided
try:
    nc_file = config['default']['nc_file']
except (KeyError, NameError):
    print('[i] Data file path not provided')
    nc_file = input("[>] Type the path of the netCDF file to use: ")
print(f"[i] NetCDF file: {nc_file}")

# load in the data
ds = nc.open_data(nc_file)

# get the name of the Delta14C variable in dataset from config or interactively
try:
    dc14_var_name = config['radiocarbon']['dc14']
except (NameError, KeyError):
    print("[!] Name of the Delta14Carbon variable was not provided")
    dc14_var_name = input("[>] Enter Delta14Carbon variable name \
                          as it appears in the dataset: ")
# rename Delta14C variable to dc14 for easy reference
ds.rename({dc14_var_name: 'dc14'})
print(f"[i] Renamed variable {dc14_var_name} to dc14")

# get the mean radiocarbon lifetime from config or interactively
try:
    mean_radio_life = int(config['radiocarbon']['mean_radiocarbon_lifetime'])
except (NameError, KeyError):
    print("[!] Mean lifetime of radiocarbon was not provided")
    mean_radio_life = int(input("[>] Enter mean radiocarbon lifetime \
                                (Cambrdige=8267, Libby=8033): "))

# compute local age from Delta14C
ds.assign(local_age=lambda x: -mean_radio_life*log((1000+x.dc14)/1000))
print(
    f"[i] Converted dc14 to age using mean radioC lifetime {mean_radio_life}"
)

# get the raw data array of local ages
xr_ds = ds.to_xarray()
age_array = xr_ds['local_age'].__array__()

# slice before anomaly specific to our data
# age_array = age_array[:78]

# offset ages to make more sense (else they'd be negative)
min_age = np.nanmin(age_array)
age_array -= min_age

# produce an array containing the R-age time series at each grid point
t, z, y, x = age_array.shape
evolutions = np.reshape(age_array[:, 0], [t, x*y]).T

# find out from config or interactively whether user wants to plot all time
# series on one plot
try:
    plot_all_evo = config['timeseries'].getboolean('plot_all_evo')
except (NameError, KeyError):
    yn = input("[>] Show a plot of all the R-age timeseries? (y/n): ")
    plot_all_evo = (yn == 'y')

# draw plot if yes
if plot_all_evo:
    # plot evolution of every grid point over time
    all_evo_fig = plt.figure()
    for point in tqdm(range(0, x*y), desc="[i] Plotting combined time series"):
        plt.plot(range(0, t), evolutions[point, :])
    plt.xlabel('time step')
    plt.ylabel('age')
    plt.title('age over time')
    all_evo_fig.show()

try:
    run_ts = config['timeseries'].getboolean('run')
except (KeyError, NameError):
    print("[!] You have not specified whether to run timeseries clustering")
    yn = input("[>] Run timeseries clustering? (y/n): ")
    run_ts = (yn == 'y')

if run_ts:
    # convert array of time series to pandas dataframe to drop NaN entries
    # then back to array, then to time series dataset for use with tslearn
    df = pd.DataFrame(evolutions)
    evolutions = np.array(df.dropna())
    ts = to_time_series_dataset(evolutions)

    # get the number of clusters for k-means and plotting, from config or
    # interactively
    try:
        n_clusters = int(config['timeseries']['n_clusters'])
    except (NameError, KeyError):
        n_clusters = int(input(
            "[>] Enter number of clusters for timeseries k-means: "
        ))
    print(f"[i] Number of clusters for time series analysis: {n_clusters}")

    # get the path of the pickle save file from config or interactively
    try:
        pickle_file = config['timeseries']['pickle']
    except (KeyError, NameError):
        print("[!] Pickle save file not provided")
        pickle_file = input(
            "[>] Enter pickle file path to save/read now: "
        )

    # assume save file exists and try to load trained k-means model from it
    try:
        with open(pickle_file, 'rb') as file:
            km = pickle.load(file)
            print(f"[i] Read in {pickle_file}")

    # no previous model saved, need to fit new model
    except FileNotFoundError:
        # initialise model
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean",
                              max_iter=10, n_jobs=-1)
        print("[i] Fitting k-means model, please stand by...")
        # actually fit the model
        km.fit(ts)

        # write k-means model object to file
        with open(pickle_file, 'wb') as file:
            pickle.dump(km, file)
        print(f"[i] Saved model to {pickle_file}")

    # get predictions for our timeseries from trained model
    y_pred = km.predict(ts)

    # plot each cluster members and their barycenter
    clusters_fig = plt.figure()
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, yi + 1)
        for xx in ts[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
    clusters_fig.suptitle("k-means results")
    clusters_fig.show()

    # assign predicted labels to the original dataframe
    df.loc[
        df.index.isin(df.dropna().index),
        'labels'] = km.labels_

    # convert to array of labels and reshape into 2D for map
    labels_flat = np.ma.masked_array(df['labels'])
    labels_shaped = np.reshape(labels_flat, [x, y])

    # finally view the clusters on a map
    plt.figure()
    plt.imshow(labels_shaped, origin='lower')
    plt.show()


try:
    run_corr = config['correlation'].getboolean('run')
except (KeyError, NameError):
    print("[!] You have not specified whether to run correlation clustering")
    yn = input("[>] Run correlation clustering? (y/n): ")
    run_corr = (yn == 'y')

if run_corr:

    # find out from config or interactively whether to mask using p-values
    try:
        pvalues = config['correlation'].getboolean('pvalues')
    except (NameError, KeyError):
        yn = input("[>] Mask out grid point with insignificant \
                ( p > 0.05 ) correlation? (y/n): ")
        pvalues = (yn == 'y')

    # visualize
    viewer = CorrelationViewer(age_array, 'Ages', pvalues=pvalues)
    plt.show()
