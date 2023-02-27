import sys
sys.path.append('.')

import argparse
from nccluster.compare import DdR_Histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='the config file for the dataset containing\
                        the R-ages to analyse')
    parser.add_argument('-o', '--original', required=True,
                        help='the config file for the dataset from which\
                        the subcluster labels were extracted')
    parser.add_argument('-l', '--labels', required=True,
                        help='.nc file containing subcluster labels')
    args = parser.parse_args()

    DdR_Histogram(args.config, args.original, args.labels)


if __name__ == "__main__":
    main()
