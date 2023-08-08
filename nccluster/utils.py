import numpy as np
import matplotlib.pyplot as plt

from tslearn.barycenters import euclidean_barycenter
from sktime.clustering.metrics import medoids


def make_subclusters_map(labels, sublabels):
    # taking a 2D array of labels and a 2D array of sublabels,
    # assign to each pixel an appropriate color

    # initialize an empty 2D array the same shape as the labels
    subclust_map = np.full_like(labels, np.nan)

    # figure out how many subclusters in each cluster
    subclust_sizes = make_subclust_sizes(labels, sublabels)

    # for every not-nan pixel on the map,
    for (yi, xi) in np.argwhere(~np.isnan(labels)):

        # extract its label and sublabel
        label = labels[yi, xi]
        sublabel = sublabels[yi, xi]

        # note the number of subclusters in the current cluster
        size = subclust_sizes[int(label)]

        # pass these data to a wrapper function that gives us the color
        subclust_map[yi, xi] = get_sublabel_colorval(label, sublabel, size)

    return subclust_map


def make_subclust_sizes(labels, sublabels):

    # figure out how many clusters there are
    n_clusters = int(np.nanmax(labels)) + 1

    # initialize empty array that will tell us the size of each cluster
    subclust_sizes = []

    # for each cluster,
    for label in range(n_clusters):

        # figure out how many subclusters it contains
        subclust_size = np.nanmax(sublabels[labels == label])

        # and add that to the list
        subclust_sizes.append(int(subclust_size) + 1)

    return subclust_sizes


def get_sublabel_colorval(label, sublabel, subclust_size):

    # remember the labels are integers,
    # e.g. red = 0, orange = 1, yellow = 2 depending on the colormap

    # spread out the color values over a decently-sized interval;
    # smaller interval translates to subcluster colors that are
    # harder to tell apart from each other;
    # bigger interval helps distinguish subclusters, but
    # smears the boundaries between clusters
    interval = 0.5

    # calculate where in this interval the sublabel falls,
    # where 0 is the first sublabel and the interval max is the last sublabel
    offset = sublabel * (interval / (subclust_size - 1))

    # set the color value in an interval centered on the label's integer value
    colorval = label - interval / 2 + offset
    return colorval


def show_map(map, cmap='viridis'):
    # simple  wrapper to show a map in the correct orientation with given cmap
    plt.figure()
    plt.imshow(map, origin='lower', cmap=cmap)
    plt.show()


def construct_barycenters(labels, sublabels, ts_array):
    return construct_centers(labels, sublabels, ts_array, euclidean_barycenter)


def construct_medoids(labels, sublabels, ts_array):

    return construct_centers(labels, sublabels, ts_array, medoids,
                             distance_metric="euclidean"
                             )


def construct_centers(labels, sublabels, ts_array, func, **kwargs):

    # array of cluster sizes
    subclust_sizes = make_subclust_sizes(labels, sublabels)

    # dictionary containing two arrays
    centers_dict = {}
    centers_dict['clusters'] = []
    centers_dict['subclusters'] = []

    # loops over labels
    for label in range(int(np.nanmax(labels)) + 1):

        # a binary mask that is true over the current cluster
        label_match_cond = labels == label

        # extract the time series for the current cluster
        cluster_tss = ts_array[label_match_cond]

        # remove nan timeseries
        # (remember there can be mismatch in coastlines between maps)
        mask = np.all(np.isnan(cluster_tss), axis=1)
        cluster_tss = cluster_tss[~mask]

        # locate the medoid among the valid time series in this cluster
        cluster_center = func(cluster_tss, **kwargs)

        # if np.isnan(cluster_center).all():
        #    print('[debug] all nan cluster center')

        # add the cluster center to the dictionary
        centers_dict['clusters'].append(cluster_center)

        # initialize empty array
        subcluster_centers = []

        # in the current cluster, loop over subclusters
        for sublabel in range(subclust_sizes[int(label)]):

            # a binary mask that is true over the current subcluster
            subclust_match_cond = sublabels == sublabel

            # extract the time series for the current subcluster
            subclust_tss = ts_array[label_match_cond & subclust_match_cond]

            # remove nan timeseries
            mask = np.all(np.isnan(subclust_tss), axis=1)
            subclust_tss = subclust_tss[~mask]

            # locate the cener  among the valid time series in this subcluster
            subclust_center = func(subclust_tss, **kwargs)

            # if np.isnan(subclust_center).all():
            #    print('[debug] all nan subcluster center')

            # add the subcluster center to the list for this cluster
            subcluster_centers.append(subclust_center)

        # add the list for this cluster to the nested list of subclust centers
        centers_dict['subclusters'].append(subcluster_centers)

    return centers_dict


def locate_medoids(labels, sublabels, data_array):

    t, y, x = data_array.shape
    ts_array = np.moveaxis(data_array, 0, -1)
    centers_dict = construct_medoids(labels, sublabels, ts_array)

    n_ts = y * x
    ts_array_flat = np.reshape(ts_array, (n_ts, t))
    locations_dict = {'clusters': [], 'subclusters': []}
    for i, cluster_ts in enumerate(centers_dict['clusters']):
        for idx in range(n_ts):
            arr = ts_array_flat[idx]
            if np.array_equal(cluster_ts, arr):
                map_idx = np.unravel_index(idx, (y, x))
                a, b = map_idx
                map_idx = (int(a), int(b))
                locations_dict['clusters'].append(map_idx)
    for i, subclust_tss in enumerate(centers_dict['subclusters']):
        cluster_locs = []
        for j, subclust_ts in enumerate(subclust_tss):
            for idx in range(n_ts):
                arr = ts_array_flat[idx]
                if np.array_equal(subclust_ts, arr):
                    map_idx = np.unravel_index(idx, (y, x))
                    a, b = map_idx
                    map_idx = (int(a), int(b))
                    cluster_locs.append(map_idx)
        locations_dict['subclusters'].append(cluster_locs)
    return locations_dict


def ts_from_locs(locations_dict, data_array):
    centers_dict = {'clusters': [], 'subclusters': []}
    ts_array = np.moveaxis(data_array, 0, -1)
    for cluster_loc in locations_dict['clusters']:
        cluster_loc = tuple(cluster_loc)
        centers_dict['clusters'].append(ts_array[cluster_loc])
    for subcluster_locs in locations_dict['subclusters']:
        subclust_centers = []
        for subclust_loc in subcluster_locs:
            subclust_loc = tuple(subclust_loc)
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
