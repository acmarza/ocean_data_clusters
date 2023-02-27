import numpy as np
import matplotlib.pyplot as plt

from tslearn.barycenters import euclidean_barycenter
from sktime.clustering.metrics import medoids


def make_subclusters_map(labels, sublabels):
    subclust_map = np.full_like(labels, np.nan)
    subclust_sizes = make_subclust_sizes(labels, sublabels)
    for (yi, xi) in np.argwhere(~np.isnan(labels)):
        label = labels[yi, xi]
        sublabel = sublabels[yi, xi]
        size = subclust_sizes[int(label)]
        subclust_map[yi, xi] = get_sublabel_colorval(label, sublabel, size)
    return subclust_map


def make_subclust_sizes(labels, sublabels):
    n_clusters = int(np.nanmax(labels)) + 1
    subclust_sizes = []
    for label in range(n_clusters):
        subclust_size = np.nanmax(sublabels[labels == label])
        subclust_sizes.append(int(subclust_size)+1)
    return subclust_sizes


def get_sublabel_colorval(label, sublabel, subclust_size):
    interval = 0.5
    offset = sublabel * (interval / (subclust_size - 1))
    colorval = label - interval/2 + offset
    return colorval


def show_map(map, cmap='viridis'):
    plt.figure()
    plt.imshow(map, origin='lower', cmap=cmap)
    plt.show()


def construct_barycenters(labels, sublabels, ts):
    labels, sublabels = map(lambda arr: arr[~np.isnan(arr)].flatten(),
                            [labels, sublabels])
    subclust_sizes = make_subclust_sizes(labels, sublabels)
    centers_dict = {}
    centers_dict['clusters'] = []
    centers_dict['subclusters'] = []

    for label in range(int(np.nanmax(labels)) + 1):
        label_match_cond = labels == label
        cluster_tss = ts[label_match_cond]
        cluster_center = euclidean_barycenter(cluster_tss)
        centers_dict['clusters'].append(cluster_center)
        subcluster_centers = []
        for sublabel in range(subclust_sizes[int(label)]):
            subclust_match_cond = sublabels == sublabel
            subclust_tss = ts[label_match_cond & subclust_match_cond]
            subclust_center = euclidean_barycenter(subclust_tss)
            subcluster_centers.append(subclust_center)
        centers_dict['subclusters'].append(subcluster_centers)

    return centers_dict


def construct_medoids(labels, sublabels, ts_array):

    subclust_sizes = make_subclust_sizes(labels, sublabels)
    centers_dict = {}
    centers_dict['clusters'] = []
    centers_dict['subclusters'] = []

    for label in range(int(np.nanmax(labels)) + 1):
        label_match_cond = labels == label
        cluster_tss = ts_array[label_match_cond]

        # remove nan timeseries
        mask = np.all(np.isnan(cluster_tss), axis=1)
        cluster_tss = cluster_tss[~mask]

        cluster_center = medoids(cluster_tss,
                                 distance_metric='euclidean')
        if np.isnan(cluster_center).all():
            print('all nan cluster center')
            for ts in cluster_tss:
                print(np.isnan(ts).all())
        centers_dict['clusters'].append(cluster_center)
        subcluster_centers = []
        for sublabel in range(subclust_sizes[int(label)]):
            subclust_match_cond = sublabels == sublabel
            subclust_tss = ts_array[label_match_cond & subclust_match_cond]

            # remove nan timeseries
            mask = np.all(np.isnan(subclust_tss), axis=1)
            subclust_tss = subclust_tss[~mask]

            subclust_center = medoids(subclust_tss,
                                      distance_metric='euclidean')
            if np.isnan(subclust_center).all():
                print('all nan subcluster center')
            subcluster_centers.append(subclust_center)
        centers_dict['subclusters'].append(subcluster_centers)

    return centers_dict


def locate_medoids(labels, sublabels, data_array):

    t, y, x = data_array.shape
    ts_array = np.moveaxis(data_array, 0, -1)
    centers_dict = construct_medoids(labels, sublabels, ts_array)

    n_ts = y*x
    ts_array_flat = np.reshape(ts_array, (n_ts, t))
    locations_dict = {'clusters': [], 'subclusters': []}
    for i, cluster_ts in enumerate(centers_dict['clusters']):
        for idx in range(n_ts):
            arr = ts_array_flat[idx]
            if np.array_equal(cluster_ts, arr):
                map_idx = np.unravel_index(idx, (y, x))
                locations_dict['clusters'].append(map_idx)
    for i, subclust_tss in enumerate(centers_dict['subclusters']):
        cluster_locs = []
        for j, subclust_ts in enumerate(subclust_tss):
            for idx in range(n_ts):
                arr = ts_array_flat[idx]
                if np.array_equal(subclust_ts, arr):
                    map_idx = np.unravel_index(idx, (y, x))
                    cluster_locs.append(map_idx)
        locations_dict['subclusters'].append(cluster_locs)
    return locations_dict


def ts_from_locs(locations_dict, data_array):
    centers_dict = {'clusters': [], 'subclusters': []}
    ts_array = np.moveaxis(data_array, 0, -1)
    for cluster_loc in locations_dict['clusters']:
        centers_dict['clusters'].append(ts_array[cluster_loc])
    for subcluster_locs in locations_dict['subclusters']:
        subclust_centers = []
        for subclust_loc in subcluster_locs:
            subclust_centers.append(ts_array[subclust_loc])
        centers_dict['subclusters'].append(subclust_centers)
    return centers_dict


def make_xy_coords(ds):
    # copy the coords of the original dataset, but keep only x and y
    all_coords = ds.to_xarray().coords
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


def reorder_labels(labels, ts_array):

    # prepare an empty array to hold the average variance of each cluster
    n_clusters = int(np.nanmax(labels) + 1)
    order_scores = np.zeros(n_clusters)
    if len(ts_array) > len(labels.flatten()):
        # drop nan-only rows
        mask = np.all(np.isnan(ts_array), axis=1)
        ts_array = ts_array[~mask]
    for label in range(0, n_clusters):
        # assume at this point self.mask is still set for this cluster
        # so can grab the timeseries array right away
        # and zip it with the labels

        # the zip above lets us this neat list comprehension
        # to retrieve just the time series with the current label
        all_tss_labeled = zip(ts_array, labels.flatten())
        cluster_tss = [ts for (ts, ll) in all_tss_labeled if ll == label]

        order_scores[label] = np.mean(np.array(cluster_tss))

    idx = np.argsort(order_scores)
    orig = np.arange(n_clusters)
    mapping = dict(zip(idx, orig))

    ordered_labels = np.copy(labels)
    for key in mapping:
        ordered_labels[labels == key] = mapping[key]

    return ordered_labels
