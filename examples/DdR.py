import sys
sys.path.append('.')

import argparse
from nccluster.compare import DdR_Histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--left',
                        help='the config file for the first dataset')
    parser.add_argument('-r', '--right',
                        help='the config file for the second dataset')
    parser.add_argument('-c', '--clusters',
                        help='.nc file containing subcluster labels')
    args = parser.parse_args()

    DdR_Histogram(args.left, args.right, args.clusters)


if __name__ == "__main__":
    main()
