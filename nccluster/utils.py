import numpy as np
import matplotlib.pyplot as plt


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
