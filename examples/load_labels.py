import argparse
import sys

sys.path.append('.')

from nccluster.ts import TimeSeriesClusteringWorkflow
# from nccluster.ts import TwoStepTimeSeriesClusterer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help="config file for time series clustering")
    parser.add_argument('-m', '--method', required=True,
                        help='which method to use (ts = time series clustering\
                        ;two = two-step time series clustering)')
    parser.add_argument('-f', '--file', required=True,
                        help="save file (.nc) for resulting labels")
    args = parser.parse_args()
    if args.method == 'ts':
        wf = TimeSeriesClusteringWorkflow(args.config)
    # elif args.method == 'two':
    #    wf = TwoStepTimeSeriesClusterer(args.config)
    else:
        print(f"[!] Unrecognised option {args.method}")

    wf.load_model_labels_from_file(args.file)
    wf.view_results()


if __name__ == "__main__":
    main()
