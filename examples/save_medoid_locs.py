import argparse
import json
import sys
import xarray as xr

sys.path.append('.')

from nccluster.utils import locate_medoids
from nccluster.radiocarbon import RadioCarbonWorkflow


def main():
    # parse args: radiocarbon config, sublabels file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help="configuration file for RadioCarbonWorkflow")
    parser.add_argument('-l', '--labels',
                        help='file containing labels and sublabels')
    parser.add_argument('-f', '--file',
                        help='file to save locations dict to')
    args = parser.parse_args()

    wf = RadioCarbonWorkflow(args.config)
    R_ages = wf.ds_var_to_array('R_age')[:, 0]

    ds = xr.load_dataset(args.labels)
    labels = ds['labels'].values
    sublabels = ds['sublabels'].values

    locations_dict = locate_medoids(labels, sublabels, R_ages)
    with open(args.file, 'w') as file:
        file.write(json.dumps(locations_dict))


if __name__ == "__main__":
    main()
