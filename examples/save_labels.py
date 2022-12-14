import argparse
from nccluster.workflows import TSClusteringWorkflow

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    help="config file for time series clustering")
parser.add_argument('-s', '--savefile',
                    help="save file (.nc) for resulting labels")
args = parser.parse_args()
wf = TSClusteringWorkflow(args.config)
wf.fit_model()
wf.save_labels_data_array(args.savefile)
