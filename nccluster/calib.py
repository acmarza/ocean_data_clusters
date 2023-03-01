from nccluster.radiocarbon import RadioCarbonWorkflow
from nccluster.utils import make_subclusters_map, ts_from_locs
import json
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


class CalibrationWorkflowBase(RadioCarbonWorkflow):

    def _checkers(self):
        super()._checkers()
        self.__check_time_step_size()

    def __check_time_step_size(self):
        self._check_config_option(
            'calibration', 'time_step_size',
            required=True,
            missing_msg='[!] You have not specified the time step size.',
            input_msg='[>] Input the time step size in years: ',
            confirm_msg='[i] Continuing with time step size in years = '
        )

    def _setters(self):
        super()._setters()
        self.__set_R_ages()
        self.__set_timesteps()

    def __set_R_ages(self):
        self.R_ages = self.ds_var_to_array('R_age')[:, 0]

    def __set_timesteps(self):
        n_times = self.R_ages.shape[0]
        step_size = int(self.config['calibration']['time_step_size'])
        max_time = (n_times - 1) * step_size
        self.timesteps = np.linspace(start=0, num=n_times, stop=max_time)


class CalibrationPlotter:

    def __init__(self, config_path, sublabels_file, sublabels_locs_file):
        wf = CalibrationWorkflowBase(config_path)
        self.R_ages = wf.R_ages
        self.timesteps = wf.timesteps
        self._init_fig()
        ds = xr.load_dataset(sublabels_file)
        self.labels = ds['labels'].values
        self.sublabels = ds['sublabels'].values
        with open(sublabels_locs_file, 'rb') as file:
            self.locations_dict = json.load(file)
            self.centers_dict = ts_from_locs(self.locations_dict,
                                             self.R_ages)

    def interactive_calib_plot(self):
        base_map = self._make_base_map()
        self.map_ax.imshow(base_map, origin='lower')
        self.fig.canvas.mpl_connect('button_press_event',
                                    self._process_click)
        plt.show()

    def _plot_marker(self, x, y):
        # clear previous marker if present
        try:
            for handle in self.loc_marker:
                handle.remove()
        except AttributeError:
            pass

        self.loc_marker = self.map_ax.plot(x, y, color='red', marker='*')

    def _plot_calib_at_location(self, y_loc, x_loc):
        R_age_history = self.R_ages[:, y_loc, x_loc]
        self._plot_calibration(R_age_history, self.calib_ax)

    def _plot_calibration(self, R_age_history, ax):
        ax.cla()
        ax.plot(self.timesteps, self.timesteps + R_age_history)
        ax.plot(self.timesteps, self.timesteps)
        ax.set_ylim(0, self.timesteps[-1] + np.nanmax(self.R_ages))
        ax.set_xlabel('atmosphere age')
        ax.set_ylabel('ocean age')

    def _init_fig(self):
        self.fig = plt.figure()
        self.map_ax = self.fig.add_subplot(131)
        self.calib_ax = self.fig.add_subplot(232)
        self.medoid_ax = self.fig.add_subplot(235)
        self.compare_ax = self.fig.add_subplot(133)

    def _plot_medoid_calib(self, y, x):
        medoid = self._get_medoid_at_loc(y, x)
        self._plot_calibration(medoid, self.medoid_ax)

    def _get_medoid_at_loc(self, y, x):
        label = self.labels[y, x]
        sublabel = self.sublabels[y, x]
        medoid = self.centers_dict['subclusters'][int(label)][int(sublabel)]
        return medoid

    def _get_medoid_loc(self, y, x):
        label = self.labels[y, x]
        sublabel = self.sublabels[y, x]
        med_loc = self.locations_dict['subclusters'][int(label)][int(sublabel)]
        return med_loc

    def _plot_site_vs_medoid(self, y, x):
        medoid = self._get_medoid_at_loc(y, x)
        site = self.R_ages[:, y, x]
        self.compare_ax.cla()
        self.compare_ax.plot(self.timesteps + medoid,
                             self.timesteps + site)
        self.compare_ax.plot(self.timesteps + medoid,
                             self.timesteps + medoid)
        self.compare_ax.set_xlim(0,
                                 self.timesteps[-1] + np.nanmax(self.R_ages))
        self.compare_ax.set_ylim(0,
                                 self.timesteps[-1] + np.nanmax(self.R_ages))
        self.compare_ax.set_xlabel('medoid R-age')
        self.compare_ax.set_ylabel('site R-age')

    def _make_base_map(self):
        return make_subclusters_map(self.labels, self.sublabels)

    def _plot_medoid_marker(self, y, x):

        # clear previous marker if present
        try:
            for handle in self.medoid_marker:
                handle.remove()
        except AttributeError:
            pass
        y, x = self._get_medoid_loc(y, x)
        self.medoid_marker = self.map_ax.plot(x, y, color='magenta',
                                              marker='o')

    def _process_click(self, event):
        if not event.inaxes == self.map_ax:
            return
        y_pos = int(event.ydata)
        x_pos = int(event.xdata)
        self._plot_calib_at_location(y_pos, x_pos)
        self._plot_medoid_calib(y_pos, x_pos)
        self._plot_site_vs_medoid(y_pos, x_pos)
        self._plot_marker(x_pos, y_pos)
        self._plot_medoid_marker(y_pos, x_pos)
        self.fig.canvas.draw()
