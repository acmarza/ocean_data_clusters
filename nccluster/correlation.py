import matplotlib.pyplot as plt
import numpy as np

from nccluster.corrviewer import CorrelationViewer, DendrogramViewer
from nccluster.radiocarbon import RadioCarbonWorkflow
from nccluster.ts import TimeSeriesWorkflowBase


class CorrelationWorkflow(TimeSeriesWorkflowBase):
    '''A workflow for visualising correlations between time series at each
    grid point in the surface ocean.'''
    def _checkers(self):
        super()._checkers()

        # only worry about save files if required to use p-values
        self.__check_pvalues_bool()
        if self.config['correlation'].getboolean('pvalues'):
            self.__check_corr_mat_file()
            self.__check_pval_mat_file()

    def run(self):
        # run parent workflow (time series plot)
        TimeSeriesWorkflowBase.run(self)

        # keyword arguments to be passed to CorrelationViewer
        kwargs = self.config['correlation']

        # initialise and show CorrelationViewer
        self.corrviewer = CorrelationViewer(self._age_array,
                                            'R-ages', **kwargs)
        plt.show()

    def __check_pvalues_bool(self):
        self._check_config_option(
            'correlation', 'pvalues',
            missing_msg="[!] You have not specified the p-values boolean.",
            input_msg="[>] Mask out grid points with insignificant \
                    ( p > 0.05 ) correlation? (y/n): ",
            confirm_msg="[i] P-values boolean: "
        )

    def __check_corr_mat_file(self):
        self._check_config_option(
            'correlation', 'corr_mat_file',
            missing_msg="[!] Correlation matrix save file not provided",
            input_msg="[>] Enter file path to save/read correlation matrix: ",
            confirm_msg='[i] Correlation matrix savefile: '
        )

    def __check_pval_mat_file(self):
        self._check_config_option(
            'correlation', 'pval_mat_file',
            missing_msg="[!] P-value matrix save file not provided",
            input_msg="[>] Enter file path to save/read p-value matrix : ",
            confirm_msg='[i] P-value matrix savefile: '
        )


class DendrogramWorkflow(RadioCarbonWorkflow):

    def run(self):
        t, _, y, x = self._age_array.shape
        evolutions = np.reshape(self._age_array[:, 0], [t, x*y]).T
        corr_mat = np.corrcoef(evolutions)
        DendrogramViewer(corr_mat, (y, x))
