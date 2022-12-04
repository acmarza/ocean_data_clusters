import argparse
import matplotlib.pyplot as plt
from nccluster.workflows import TSClusteringWorkflow

# define and parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from, \
                    if parameters not supplied interactively")
args = parser.parse_args()

# initialise time series workflow based on config file read on commandline
ts_workflow = TSClusteringWorkflow(args.config)

# run the workflow and show plots
ts_workflow.run()
plt.show()
