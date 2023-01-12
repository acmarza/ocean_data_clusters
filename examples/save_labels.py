import argparse
import sys

sys.path.append('.')

from nccluster.workflows import TimeSeriesClusteringWorkflow
from nccluster.workflows import TwoStepTimeSeriesClusterer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help="config file for time series clustering")
    parser.add_argument('-m', '--method',
                        help='which method to use (ts = time series clustering\
                        ;two = two-step time series clustering)')
    parser.add_argument('-f', '--file',
                        help="save file (.nc) for resulting labels")
    args = parser.parse_args()
    if args.method == 'ts':
        wf = TimeSeriesClusteringWorkflow(args.config)
    elif args.method == 'two':
        wf = TwoStepTimeSeriesClusterer(args.config)
    else:
        print(f"[!] Unrecognised option {args.method}")

    wf.cluster()
    wf.save_labels(args.file)


if __name__ == "__main__":
    main()
