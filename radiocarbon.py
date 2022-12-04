import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nccluster.corrviewer import CorrelationViewer
from nccluster.workflows import RadioCarbonWorkflow

# define and parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from, \
                    if parameters not supplied interactively")
args = parser.parse_args()

radio_c_workflow = RadioCarbonWorkflow(args.config)


if run_corr:
    # find out whether to mask using p-values, from config or interactively
    try:
        pvalues = config['correlation'].getboolean('pvalues')
    except (NameError, KeyError):
        yn = input("[>] Mask out grid points with insignificant \
                ( p > 0.05 ) correlation? (y/n): ")
        pvalues = (yn == 'y')

    # define keyword arguments for CorrelationViewer
    kwargs = {'pvalues': pvalues}

    if pvalues:
        # if taking into account p-values, will use the slower scipy pearsonr
        # so it's better to work with save files
        # their names are read from config or user input
        try:
            corr_mat_file = config['correlation']['corr_mat_file']
        except (KeyError, NameError):
            print("[!] Correlation matrix save file not provided")
            corr_mat_file = input(
                "[>] Enter file path to save/read correlation matrix now: "
            )
        try:
            pval_mat_file = config['correlation']['pval_mat_file']
        except (KeyError, NameError):
            print("[!] P-value matrix save file not provided")
            pval_mat_file = input(
                "[>] Enter file path to save/read p-value matrix now: "
            )
        # add the save file paths to keyword arguments for CorrelationViewer
        kwargs['corr_mat_file'] = corr_mat_file
        kwargs['pval_mat_file'] = pval_mat_file

    # finally invoke CorrelationViewer and visualise
    viewer = CorrelationViewer(age_array, title="R-ages", **kwargs)
    plt.show()
