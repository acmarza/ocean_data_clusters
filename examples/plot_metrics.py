import sys
import argparse

sys.path.append('.')

from nccluster.metrics import plot_metrics


def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True,
                        help="config file to plot metrics")
    parser.add_argument("--subtitles", "-s", required=True,
                        help="comma-separated subtitles, one per column")
    parser.add_argument("--title", "-t", required=True,
                        help="title to display on top of figure")
    args = parser.parse_args()
    plot_metrics(args.config,
                 suptitle=args.title,
                 col_titles=args.subtitles.split(",")
                 )


if __name__ == "__main__":
    main()
