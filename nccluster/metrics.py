from kneed import KneeLocator
import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics_savefiles, suptitle="Clustering metrics summary",
                 styles=['solid', 'dotted'], run_names=["run1", "run2"]):

    # read the dataframes, put them into an array
    dfs = []
    for savefile in metrics_savefiles:
        with open(savefile) as f:
            df = pd.read_csv(f)
            dfs.append(df)

    # prepare 4 subplots for the 4 scores (one per row)
    fig, axes = plt.subplots(4, 1, figsize=(5, 10))

    # handy list of the plot titles
    y_labels = ['Sum of squared errors',
                'Silhouette Score',
                'Calinski-Harabasz Index',
                'Davies-Bouldin Index\n(flipped)']
    for (ax, label) in zip(axes, y_labels):
        ax.set_ylabel(label)

    for df, style, name in zip(dfs, styles, run_names):

        # handy list of the scores to plot
        scores = [df['inertia'], df['sil_score'],
                  df['ch_index'], -df['db_index']]

        # loop over the Axes, plot the scores
        for (ax, score) in zip(axes, scores):
            ax.plot(df['K'], score, linestyle=style, label=name)

        # on the plot of inertias, put the knee  point as a vertical line
        kn = KneeLocator(x=df['K'], y=df['inertia'], curve='convex',
                         direction='decreasing')
        axes[0].axvline(kn.knee, linestyle=style, label="_exclude")

    # bottom text
    axes[-1].set_xlabel("number of clusters, K")

    axes[0].legend()

    # top text
    fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()
