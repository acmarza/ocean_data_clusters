import argparse
import sys

sys.path.append('.')

from nccluster.workflows import TSClusteringWorkflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help="config file for time series clustering")
    parser.add_argument('-l', '--long_name',
                        help="long name for nc variable")
    parser.add_argument('-f', '--file',
                        help="save file (.nc) for resulting labels")
    args = parser.parse_args()
    wf = TSClusteringWorkflow(args.config)
    wf.fit_model()
    wf.save_labels_data_array(args.file, args.long_name)


if __name__ == "__main__":
    main()
