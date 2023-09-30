from kneed import KneeLocator
import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(config_csv, suptitle, col_titles):

    with open(config_csv) as f:
        conf_df = pd.read_csv(f)

    n_cols = conf_df['column'].max()
    fig, axes = plt.subplots(4, n_cols, sharex='col')

    # avoid treating 1D Axes array as a separate case by reshaping
    # to act like 2D array
    if len(axes.shape) == 1:
        axes = axes.reshape((4, 1))

    for index, row in conf_df.iterrows():
        plot_metrics_single(row['metrics_csv'],
                            axes[:, row['column'] - 1],
                            label=row['label'],
                            style=row['style']
                            )

    # handy list of the plot titles
    y_labels = ['Sum of\nsquared errors',
                'Silhouette\nscore',
                'Calinski-Harabsz\nindex',
                'Davies-Bouldin\nindex (flipped)']

    # score labels
    for (row, label) in enumerate(y_labels):
        axes[row][0].set_ylabel(label)

    # bottom text
    for ax in axes[-1][:]:
        ax.set_xlabel("number of clusters, K")

    # legends and titles in top row
    for ax, title in zip(axes[0][:], col_titles):
        ax.set_title(title)
        ax.legend()

    # every ax in scientific notation
    for ax in axes.flatten():
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        t = ax.yaxis.get_offset_text()
        t.set_visible(False)
        ax.grid(axis='x', color='lightgrey')

    plt.subplots_adjust(hspace=0.04)

    fig.canvas.draw()

    # after drawing, correct scientific notation position
    for ax in axes.flatten():
        t = ax.yaxis.get_offset_text()
        twin = ax.twinx()
        t.set_visible(False)
        twin.set_yticklabels([])
        twin.set_ylabel(t._text)

    fig.suptitle(suptitle)

    plt.show()


def plot_metrics_single(metrics_csv, axes, label, style='solid'):

    # read the dataframes, put them into an array
    with open(metrics_csv) as f:
        df = pd.read_csv(f)

    # handy list of the scores to plot
    scores = [df['inertia'], df['sil_score'],
              df['ch_index'], -df['db_index']]

    # loop over the Axes, plot the scores
    for (ax, score) in zip(axes, scores):
        ax.plot(df['K'], score, linestyle=style, label=label)

    # on the plot of inertias, put the knee  point as a vertical line
    kn = KneeLocator(x=df['K'], y=df['inertia'], curve='convex',
                     direction='decreasing')
    axes[0].axvline(kn.knee, linestyle=style, color='black',
                    label="_exclude")
