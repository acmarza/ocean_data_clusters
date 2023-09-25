import sys
import argparse

sys.path.append('.')

from nccluster.metrics import plot_metrics


def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--savefiles", "-f", required=True,
                        help="comma-separated list of metrics savefiles")
    parser.add_argument("--labels", "-l", required=True,
                        help="comma-separated list of labels for legend")
    parser.add_argument("--styles", "-s", required=True,
                        help="comma-separated list of linestyles (see matplot\
    lib docs)")
    parser.add_argument("--title", "-t", required=True,
                        help="title to display on top of figure")
    args = parser.parse_args()
    plot_metrics(metrics_savefiles=args.savefiles.split(","),
                 suptitle=args.title,
                 run_names=args.labels.split(","),
                 styles=args.styles.split(",")
                 )


if __name__ == "__main__":
    main()
