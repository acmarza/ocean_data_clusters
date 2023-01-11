import sys
sys.path.append('.')

import argparse
from nccluster.compare import DdR_Histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help='the config file for the dataset')
    parser.add_argument('-l', '--labels',
                        help='.nc file containing subcluster labels')
    parser.add_argument('-m', '--medoids',
                        help='the pickle file containing cluster centroids')
    args = parser.parse_args()

    DdR_Histogram(args.config, args.labels, args.medoids)


if __name__ == "__main__":
    main()
