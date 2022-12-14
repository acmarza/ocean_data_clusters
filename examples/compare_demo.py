import argparse
from nccluster.compare import ClusterMatcher

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--left',
                    help='the first .nc file containing cluster labels')
parser.add_argument('-r', '--right',
                    help='the second .nc file containing cluster labels')
args = parser.parse_args()

# initialise the matcher and load in the labels
matcher = ClusterMatcher()
matcher.labels_from_file(args.left, args.right)

# first look at labels
matcher.compare_maps()

# regrid labels to same coords and view results
matcher.regrid_left_to_right()
matcher.compare_maps()

# harmonise colors and view results
matcher.match_labels()
matcher.compare_maps()
