import argparse
import sys

sys.path.append('.')

from nccluster.workflows import KMeansWorkflow
from nccluster.workflows import CorrelationWorkflow
from nccluster.ts import TimeSeriesClusteringWorkflow
from nccluster.ts import TwoStepTimeSeriesClusterer
from nccluster.workflows import DendrogramWorkflow


def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True,
                        help="file to read configuration from,\
                        if parameters not supplied interactively")
    parser.add_argument("--method", "-m", required=True,
                        help=" which method to use to analyse\
                        the data: km = k-means; ts = timeseries clustering;\
                        corr = correlation viewer + hierarchical clustering;\
                        two = two-step (clusters + subclusters) time series;\
                        dendro = dendrogram viewer + hierarchical clustering;\
                        metrics = time series clustering metrics"
                        )
    args = parser.parse_args()

    # depending on chosen method, run appropriate workflow
    if args.method == 'km':
        wf = KMeansWorkflow(args.config)
    elif args.method == 'ts':
        wf = TimeSeriesClusteringWorkflow(args.config)
    elif args.method == 'metrics':
        wf = TimeSeriesClusteringWorkflow(args.config)
        wf.clustering_metrics()
        exit()
    elif args.method == 'corr':
        wf = CorrelationWorkflow(args.config)
    elif args.method == 'two':
        wf = TwoStepTimeSeriesClusterer(args.config)
    elif args.method == 'dendro':
        wf = DendrogramWorkflow(args.config)
    else:
        print("[!] Unrecognised method passed via commandline; see help (-h)")

    wf.run()


if __name__ == "__main__":
    main()
