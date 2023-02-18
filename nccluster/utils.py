import numpy as np
import matplotlib.pyplot as plt

from tslearn.barycenters import euclidean_barycenter


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
