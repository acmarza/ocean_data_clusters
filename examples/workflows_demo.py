import argparse
import sys

sys.path.append('.')

from nccluster.workflows import KMeansWorkflow
from nccluster.workflows import CorrelationWorkflow
from nccluster.workflows import TSClusteringWorkflow


def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True,
                        help="file to read configuration from,\
                        if parameters not supplied interactively")
    parser.add_argument("--method", "-m", required=True,
                        help=" which method to use to analyse\
                        the data: km = k-means; ts = timeseries clustering;\
                        corr = correlation viewer + hierarchical clustering"
                        )
    args = parser.parse_args()

    # depending on chosen method, run appropriate workflow
    if args.method == 'km':
        km_workflow = KMeansWorkflow(args.config)
        km_workflow.run()
    elif args.method == 'ts':
        ts_workflow = TSClusteringWorkflow(args.config)
        ts_workflow.run()
    elif args.method == 'corr':
        corr_workflow = CorrelationWorkflow(args.config)
        corr_workflow.run()
    else:
        print("[!] Unrecognised method passed via commandline; see help (-h)")


if __name__ == "__main__":
    main()
