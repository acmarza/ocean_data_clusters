# multi slice viewer adapted from
# https://www.datacamp.com/tutorial/matplotlib-3d-volumetric-data

# kmeans pipeline based on
# https://realpython.com/k-means-clustering-python/
import argparse
from nccluster.workflows import KMeansWorkflow

# parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from,\
                     if parameters not supplied interactively")
args = parser.parse_args()

km_workflow = KMeansWorkflow(args.config)
km_workflow.run()

# labels = km_workflow.labels
# df = km_workflow.df.dropna()

# plt.figure()
# plt.scatter(df['LOCAL_AGE'], df['o2'], c=labels, marker=',')
# plt.show()


# labels_colors = cmap(np.linspace(0, 1, num=n_clusters))

# rescale the centroids back to original data ranges
# centroids = kmeans.cluster_centers_
# centroids = pipe['preprocessor'].inverse_transform(centroids)

# scatter plot of the centroids for each selected variable
# n_params = len(selected_vars)
# centroids_fig, centroids_axes = plt.subplots(1, n_params)
# for i, ax in enumerate(centroids_axes):
#    ax.set_xticks([1], [selected_vars[i]])
#    for j in range(0, centroids.shape[0]):
#        ax.scatter(1, centroids[j, i], color=labels_colors[j])
# centroids_fig.show()
