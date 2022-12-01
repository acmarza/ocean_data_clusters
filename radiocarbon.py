import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from nccluster.corrviewer import CorrelationViewer
from nccluster.workflows import RadioCarbonWorkflow

# define and parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from, \
                    if parameters not supplied interactively")
args = parser.parse_args()

radio_c_workflow = RadioCarbonWorkflow(args.config)

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
    # i.e. to which cluster each timeseries belongs
    y_pred = km.predict(ts)

    # plot each cluster members and their barycenter
    # initialise figure
    clusters_fig = plt.figure()

    # for each cluster/label
    for yi in range(n_clusters):
        # create a subplot in a table with n_cluster rows and 1 column
        # this subplot is number yi+1 because we're counting from 0
        plt.subplot(n_clusters, 1, yi + 1)
        # for every timeseries in the dataset that has been assigned label yi
        for xx in ts[y_pred == yi]:
            # plot with a thin transparent line
            plt.plot(xx.ravel(), "k-", alpha=.2)
        # plot the cluster barycenter
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        # label the cluster
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
    # add a title at the top of the figure
    clusters_fig.suptitle("k-means results")
    # finally show the figure
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

if run_corr:
    # find out whether to mask using p-values, from config or interactively
    try:
        pvalues = config['correlation'].getboolean('pvalues')
    except (NameError, KeyError):
        yn = input("[>] Mask out grid points with insignificant \
                ( p > 0.05 ) correlation? (y/n): ")
        pvalues = (yn == 'y')

    # define keyword arguments for CorrelationViewer
    kwargs = {'pvalues': pvalues}

    if pvalues:
        # if taking into account p-values, will use the slower scipy pearsonr
        # so it's better to work with save files
        # their names are read from config or user input
        try:
            corr_mat_file = config['correlation']['corr_mat_file']
        except (KeyError, NameError):
            print("[!] Correlation matrix save file not provided")
            corr_mat_file = input(
                "[>] Enter file path to save/read correlation matrix now: "
            )
        try:
            pval_mat_file = config['correlation']['pval_mat_file']
        except (KeyError, NameError):
            print("[!] P-value matrix save file not provided")
            pval_mat_file = input(
                "[>] Enter file path to save/read p-value matrix now: "
            )
        # add the save file paths to keyword arguments for CorrelationViewer
        kwargs['corr_mat_file'] = corr_mat_file
        kwargs['pval_mat_file'] = pval_mat_file

    # finally invoke CorrelationViewer and visualise
    viewer = CorrelationViewer(age_array, title="R-ages", **kwargs)
    plt.show()
