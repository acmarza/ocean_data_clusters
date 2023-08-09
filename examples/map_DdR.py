import sys
sys.path.append('.')

import argparse
from nccluster.compare import DdR_Maps


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

    mapper = DdR_Maps(args.config, args.original, args.labels)

    mapper.table_cluster_mean_and_std()

    print("[i] Map options:\n\
          1. cosines\n\
          2. average ΔR per cluster\n\
          3. stddev ΔR per cluster\n\
          4. average ΔR per grid point\n\
          5. stddev ΔR per grid point")

    funcs = [mapper.map_all_cosines,
             mapper.map_mean_diff,
             mapper.map_cluster_stddev,
             mapper.map_diffs,
             mapper.map_local_stddev
             ]

    while True:
        mode = int(input("[>] maps to show (Ctrl+C to quit): "))
        mode -= 1
        funcs[mode]()


if __name__ == "__main__":
    main()
