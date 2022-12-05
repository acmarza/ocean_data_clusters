import argparse
from nccluster.workflows import TSClusteringWorkflow

# define and parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from, \
                    if parameters not supplied interactively")
args = parser.parse_args()

# initialise and run the workflow
ts_workflow = TSClusteringWorkflow(args.config)
ts_workflow.run()
