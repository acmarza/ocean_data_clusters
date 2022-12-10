import numpy as np
import matplotlib.pyplot as plt


class ClusterMatcher:

    def labels_from_arrays(self, labels_left, labels_right):
        self.labels_left = labels_left
        self.labels_right = labels_right

    def match_labels(self):
        # flatten labels into 1D array
        labels_left_flat = self.labels_left.flatten()
        labels_right_flat = self.labels_right.flatten()

        # the labels are 0 to n-1
        n_clusters = int(np.nanmax(self.labels_left)) + 1
        shape = self.labels_left.shape

        # initialise empty arrays
        array_of_sets_1 = []
        array_of_sets_2 = []

        # for every cluster (label), get all the indices that have
        # the current label and put them in a set
        for cluster in range(0, n_clusters):
            (indices_1, ) = np.where(labels_left_flat == cluster)
            array_of_sets_1.append(set(indices_1))
            (indices_2, ) = np.where(labels_right_flat == cluster)
            array_of_sets_2.append(set(indices_2))

        # reorder lists to have biggest sets first
        array_of_sets_1.sort(key=lambda x: -len(x))
        array_of_sets_2.sort(key=lambda x: -len(x))

        # quantify overlap between pairs of sets between the two label versions
        overlap_matrix = np.zeros((n_clusters, n_clusters))
        pairings_scores = []
        for i, left_set in enumerate(array_of_sets_1):
            for j, right_set in enumerate(array_of_sets_2):
                sym_diff_size = len(left_set.symmetric_difference(right_set))
                intersection_size = len(left_set.intersection(right_set))
                overlap = intersection_size/sym_diff_size
                # print(f'{i} vs. {j}: {overlap:.2%}')
                pairings_scores.append((i, j, overlap))

        pairings_scores.sort(reverse=True, key=lambda x: x[2])
        print('sorted')

        print(overlap_matrix)

        translation = np.full(n_clusters, np.nan)

        for pair in pairings_scores:

            print(pair)
            left, right, score = pair
            if right not in translation and np.isnan(translation[left]):
                translation[left] = right
            print(translation)

        for i, left_set in enumerate(array_of_sets_1):
            for idx in left_set:
                labels_left_flat[idx] = translation[i]

        for i, left_set in enumerate(array_of_sets_2):
            for idx in left_set:
                labels_right_flat[idx] = i

        self.labels_left = np.reshape(labels_left_flat, shape)
        self.labels_right = np.reshape(labels_right_flat, shape)

    def load_labels(self, save_path_1, save_path_2):
        with np.load(save_path_1) as npz:
            self.labels_left = np.ma.MaskedArray(**npz)
        with np.load(save_path_2) as npz:
            self.labels_right = np.ma.MaskedArray(**npz)

    def save_labels(self, save_path_1, save_path_2):
        np.savez_compressed(save_path_1,
                            data=self.labels_left.data,
                            mask=self.labels_left.mask)
        np.savez_compressed(save_path_2,
                            data=self.labels_right.data,
                            mask=self.labels_right.mask)

    def compare_maps(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.labels_left, origin='lower')
        ax2.imshow(self.labels_right, origin='lower')
        plt.show()
