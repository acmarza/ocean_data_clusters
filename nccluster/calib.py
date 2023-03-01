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

    def __init__(self, config_path):
        wf = CalibrationWorkflowBase(config_path)
        self.R_ages = wf.R_ages
        self.timesteps = wf.timesteps
        self._init_fig()

    def interactive_calib_plot(self):
        base_map = self._make_base_map()
        self.map_ax.imshow(base_map, origin='lower')
        self.fig.canvas.mpl_connect('button_press_event',
                                    self._process_click)
        plt.show()

    def _init_fig(self):
        self.fig, (self.map_ax, self.calib_ax) = plt.subplots(1, 2)

    def _make_base_map(self):
        return np.isnan(self.R_ages[0])

    def _process_click(self, event):
        if not event.inaxes == self.map_ax:
            return
        y_pos = int(event.ydata)
        x_pos = int(event.xdata)
        self._plot_calib_at_location(y_pos, x_pos)
        self._plot_marker(x_pos, y_pos)
        self.fig.canvas.draw()

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
        ax.set_xlabel('atmosphere age')
        ax.set_ylabel('ocean age')


class CalibrationSampler(CalibrationPlotter):

    def __init__(self, config_path, sublabels_file, sublabels_locs_file):
        CalibrationPlotter.__init__(self, config_path)
        ds = xr.load_dataset(sublabels_file)
        self.labels = ds['labels'].values
        self.sublabels = ds['sublabels'].values
        with open(sublabels_locs_file, 'rb') as file:
            locations_dict = json.load(file)
            self.centers_dict = ts_from_locs(locations_dict,
                                             self.R_ages)
            print(self.centers_dict['clusters'][0].shape)

    def _init_fig(self):
        self.fig, (self.map_ax,
                   self.calib_ax, self.medoid_ax) = plt.subplots(1, 3)

    def _plot_medoid_calib(self, y, x):
        label = self.labels[y, x]
        sublabel = self.sublabels[y, x]
        medoid = self.centers_dict['subclusters'][int(label)][int(sublabel)]
        self._plot_calibration(medoid, self.medoid_ax)

    def _make_base_map(self):
        return make_subclusters_map(self.labels, self.sublabels)

    def _process_click(self, event):
        if not event.inaxes == self.map_ax:
            return
        y_pos = int(event.ydata)
        x_pos = int(event.xdata)
        self._plot_calib_at_location(y_pos, x_pos)
        self._plot_medoid_calib(y_pos, x_pos)
        self._plot_marker(x_pos, y_pos)
        self.fig.canvas.draw()
