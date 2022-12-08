import numpy as np
import matplotlib.pyplot as plt


class ClusterMatcher:

    def labels_from_arrays(self, labels_1, labels_2):
        self.labels_1 = labels_1
        self.labels_2 = labels_2

    def match_labels(self):
        # flatten labels into 1D array
        labels_1_flat = self.labels_1.flatten()
        labels_2_flat = self.labels_2.flatten()

        # the labels are 0 to n-1
        n_clusters = int(np.nanmax(self.labels_1)) + 1
        shape = self.labels_1.shape

        # initialise empty arrays
        array_of_sets_1 = []
        array_of_sets_2 = []

        # for every cluster (label), get all the indices that have
        # the current label and put them in a set
        for cluster in range(0, n_clusters):
            (indices_1, ) = np.where(labels_1_flat == cluster)
            array_of_sets_1.append(set(indices_1))
            (indices_2, ) = np.where(labels_2_flat == cluster)
            array_of_sets_2.append(set(indices_2))

        # reorder lists to have biggest sets first
        array_of_sets_1.sort(key=lambda x: -len(x))
        # array_of_sets_2.sort(key=lambda x: -len(x))
        overlap_matrix = np.zeros((n_clusters, n_clusters))
        for i, idx_set in enumerate(array_of_sets_1):
            for j, cf_set in enumerate(array_of_sets_2):
                union_size = len(idx_set.union(cf_set))
                sym_diff_size = len(idx_set.symmetric_difference(cf_set))
                mismatch = sym_diff_size / union_size
                overlap = 1 - mismatch
                print(f'{i} vs. {j}: {overlap:.2%}')
                overlap_matrix[i, j] = overlap

        translation = []
        for cluster in range(0, n_clusters):
            match_idx = overlap_matrix[cluster].argmax()
            translation.append(match_idx)
        print(translation)

        for i, idx_set in enumerate(array_of_sets_1):
            for idx in idx_set:
                labels_1_flat[idx] = translation[i]

        for i, idx_set in enumerate(array_of_sets_2):
            for idx in idx_set:
                labels_2_flat[idx] = i

        self.labels_1 = np.reshape(labels_1_flat, shape)
        self.labels_2 = np.reshape(labels_2_flat, shape)

    def load_labels(self, save_path_1, save_path_2):
        with np.load(save_path_1) as npz:
            self.labels_1 = np.ma.MaskedArray(**npz)
        with np.load(save_path_2) as npz:
            self.labels_2 = np.ma.MaskedArray(**npz)

    def save_labels(self, save_path_1, save_path_2):
        np.savez_compressed(save_path_1,
                            data=self.labels_1.data, mask=self.labels_1.mask)
        np.savez_compressed(save_path_2,
                            data=self.labels_2.data, mask=self.labels_2.mask)

    def compare_maps(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.labels_1, origin='lower')
        ax2.imshow(self.labels_2, origin='lower')
        plt.show()
