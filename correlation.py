import argparse
from nccluster.workflows import CorrelationWorkflow

# define and parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from, \
                    if parameters not supplied interactively")
args = parser.parse_args()

# initialise and run workflow
corr_workflow = CorrelationWorkflow(args.config)
corr_workflow.run()
