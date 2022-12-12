import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe


class ClusterMatcher:

    def labels_from_data_arrays(self, labels_left, labels_right):
        self.labels_left = labels_left
        self.labels_right = labels_right

    def labels_from_file(self, save_path_1, save_path_2):
        self.labels_left = xr.open_dataarray(save_path_1)
        self.labels_right = xr.open_dataarray(save_path_2)

    def save_labels(self, save_path_1, save_path_2):
        self.labels_left.to_netcdf(save_path_1)
        self.labels_right.to_netcdf(save_path_2)

    def compare_maps(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.labels_left.values, origin='lower')
        ax2.imshow(self.labels_right.values, origin='lower')
        plt.show()

    def regrid_left_to_right(self):
        regridder = xe.Regridder(self.labels_left,
                                 self.labels_right,
                                 'nearest_s2d'
                                 )
        self.labels_left = regridder(self.labels_left)

    def match_labels(self):
        # flatten labels into 1D array
        labels_left_flat = self.labels_left.values.flatten()
        labels_right_flat = self.labels_right.values.flatten()

        # the labels are 0 to n-1
        n_clusters = int(np.nanmax(self.labels_left.values)) + 1
        if n_clusters != int(np.nanmax(self.labels_right.values)) + 1:
            print("[!] The maps have a different number of clusters")
            exit()

        # initialise empty arrays
        array_of_sets_left = []
        array_of_sets_right = []

        # for every cluster (label), get all the indices that have
        # the current label and put them in a set
        for cluster in range(0, n_clusters):
            (indices_1, ) = np.where(labels_left_flat == cluster)
            array_of_sets_left.append(set(indices_1))
            (indices_2, ) = np.where(labels_right_flat == cluster)
            array_of_sets_right.append(set(indices_2))

        # reorder lists to have biggest sets first
        array_of_sets_left.sort(key=lambda x: -len(x))
        array_of_sets_right.sort(key=lambda x: -len(x))

        # quantify overlap between pairs of sets between the two label versions
        pairings_scores = []
        for i, left_set in enumerate(array_of_sets_left):
            for j, right_set in enumerate(array_of_sets_right):
                union_size = len(left_set.union(right_set))
                intersection_size = len(left_set.intersection(right_set))
                overlap = intersection_size/union_size
                # print(f'{i} vs. {j}: {overlap:.2%}')
                pairings_scores.append((i, j, overlap))

        # start with the highest overlap scores
        pairings_scores.sort(reverse=True, key=lambda x: x[2])

        # prepare an empty array to hold the 1:1 matching between
        # labels on the left and right
        translation = np.full(n_clusters, np.nan)

        # put the pairings with the highest scores in the translation array
        # skipping when an entry already exists
        for pair in pairings_scores:
            left, right, score = pair
            if right not in translation and np.isnan(translation[left]):
                translation[left] = right

        # relabel the sets to match based on our translation function
        for i, left_set in enumerate(array_of_sets_left):
            for idx in left_set:
                labels_left_flat[idx] = translation[i]

        for i, right_set in enumerate(array_of_sets_right):
            for idx in right_set:
                labels_right_flat[idx] = i

        # reshape translated 1D arrays and update attributes
        shape = self.labels_left.values.shape
        self.labels_left.values = np.reshape(labels_left_flat, shape)
        self.labels_right.values = np.reshape(labels_right_flat, shape)
