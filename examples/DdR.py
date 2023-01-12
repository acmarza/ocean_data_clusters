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
    args = parser.parse_args()

    DdR_Histogram(args.config, args.labels)


if __name__ == "__main__":
    main()
