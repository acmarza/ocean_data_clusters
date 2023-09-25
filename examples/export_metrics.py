import argparse
import sys

sys.path.append('.')

from nccluster.ts import TimeSeriesClusteringWorkflow


def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True,
                        help="file to read configuration from,\
                        if parameters not supplied interactively")
    args = parser.parse_args()

    wf = TimeSeriesClusteringWorkflow(args.config)
    wf.export_clustering_metrics()

if __name__ == "__main__":
    main()
