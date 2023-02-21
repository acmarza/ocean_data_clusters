import sys
sys.path.append('.')

import argparse
from nccluster.compare import ClusterMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--left',
                        help='the first .nc file containing cluster labels')
    parser.add_argument('-r', '--right',
                        help='the second .nc file containing cluster labels')
    parser.add_argument('-m', '--match', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='whether to attempt cluster matching')
    parser.add_argument('-g', '--regrid', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='whether to regrid the left set of labels')
    args = parser.parse_args()

    # initialise the matcher and load in the labels
    matcher = ClusterMatcher()
    matcher.labels_from_file(args.left, args.right)

    # first look at labels
    matcher.compare_maps()

    if args.regrid:
        # regrid labels to same coords and view results
        matcher.regrid_left_to_right()
        matcher.compare_maps()

    if args.match:
        # harmonise colors and view results
        matcher.match_labels()
        matcher.compare_maps()

    matcher.overlap()

if __name__ == "__main__":
    main()
