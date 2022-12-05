import argparse
from nccluster.workflows import KMeansWorkflow

# parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from,\
                     if parameters not supplied interactively")
args = parser.parse_args()

# initialise and run the workflow
km_workflow = KMeansWorkflow(args.config)
km_workflow.run()

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
