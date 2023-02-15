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


def subset_coords(ds, axes=['X', 'Y']):
    # copy the coords of the original dataset, but keep only specified axes
    # default x and y only
    all_coords = ds.to_xarray().coords
    coords = {}
    for key in all_coords:
        try:
            if all_coords[key].axis in axes:
                coords[key] = all_coords[key]
        except AttributeError:
            pass
    # arcane magic to put the coordinates in reverse order
    # because otherwise DataArray expects the transpose of what we have
    coords = dict(reversed(list(coords.items())))
    return coords
