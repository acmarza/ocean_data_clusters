import argparse
import sys

sys.path.append('.')

from nccluster.radiocarbon import RadioCarbonWorkflow


def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True,
                        help="file to read configuration from,\
                        if parameters not supplied interactively")
    args = parser.parse_args()

    wf = RadioCarbonWorkflow(args.config)
    wf.list_plottable_vars()
    wf.interactive_var_plot()


if __name__ == "__main__":
    main()
