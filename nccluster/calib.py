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
        plt.tight_layout()
        plt.show()

    def _plot_marker(self):
        # clear previous marker if present
        try:
            for handle in self.loc_marker:
                handle.remove()
        except AttributeError:
            pass
        y, x = self.clicked_pos
        self.loc_marker = self.map_ax.plot(x, y, color='red', marker='*')

    def _plot_calib_at_location(self):
        self._plot_calibration(self.site, self.calib_ax,
                               title='selected site')

    def _plot_calibration(self, R_age_history, ax, refresh=True, c='red',
                          title='calibration'):
        if refresh:
            ax.cla()
            ax.plot(self.timesteps, self.timesteps)
            ax.set_ylim(0, self.timesteps[-1] + np.nanmax(self.R_ages))
            ax.set_xlabel('atmosphere age')
            ax.set_ylabel('ocean age')
        ax.plot(self.timesteps, self.timesteps + R_age_history, c=c)
        ax.set_title(title)

    def _init_fig(self):
        self.fig = plt.figure()
        self.map_ax = self.fig.add_subplot(131)
        self.calib_ax = self.fig.add_subplot(232)
        self.medoid_ax = self.fig.add_subplot(235)
        self.compare_ax = self.fig.add_subplot(233)
        self.age_model_ax = self.fig.add_subplot(236)

    def _plot_medoid_calib(self):
        self._plot_calibration(self.medoid, self.medoid_ax, c='magenta',
                               title='medoid')

    def _set_ts_at_loc(self):
        y, x = self.clicked_pos
        self.site = self.R_ages[:, y, x]

    def _set_medoid_at_loc(self):
        y, x = self.clicked_pos
        label = self.labels[y, x]
        sublabel = self.sublabels[y, x]
        medoid = self.centers_dict['subclusters'][int(label)][int(sublabel)]
        self.medoid = medoid

    def _get_medoid_loc(self, y, x):
        label = self.labels[y, x]
        sublabel = self.sublabels[y, x]
        med_loc = self.locations_dict['subclusters'][int(label)][int(sublabel)]
        return med_loc

    def _plot_site_vs_medoid(self):
        self.compare_ax.cla()
        self.compare_ax.plot(self.timesteps + self.medoid,
                             self.timesteps + self.site)
        self.compare_ax.plot(self.timesteps + self.medoid,
                             self.timesteps + self.medoid)
        self.compare_ax.set_xlim(0,
                                 self.timesteps[-1] + np.nanmax(self.R_ages))
        self.compare_ax.set_ylim(0,
                                 self.timesteps[-1] + np.nanmax(self.R_ages))
        self.compare_ax.set_xlabel('medoid ocean age')
        self.compare_ax.set_ylabel('site ocean age')

    def _make_base_map(self):
        return make_subclusters_map(self.labels, self.sublabels)

    def _plot_medoid_marker(self):
        y, x = self.clicked_pos

        # clear previous marker if present
        try:
            for handle in self.medoid_marker:
                handle.remove()
        except AttributeError:
            pass
        y, x = self._get_medoid_loc(y, x)
        self.medoid_marker = self.map_ax.plot(x, y, color='magenta',
                                              marker='o')

    def _select_points_from_ts(self, ts, n_points=20):
        idx = range(len(ts))
        choice_idx = np.random.choice(idx, size=n_points, replace=False)
        choice_idx = np.sort(choice_idx)
        return choice_idx

    def _plot_medoid_samples(self, choice_idx):
        x = self.timesteps[choice_idx]
        y = self.medoid[choice_idx]

        self.medoid_ax.plot(x, x + y,
                            marker='o', markersize=2, c='green',
                            linestyle='None')

    def _plot_site_samples(self, choice_idx):
        x = self.timesteps[choice_idx]
        y = self.site[choice_idx]

        self.calib_ax.plot(x, x + y,
                           marker='o', markersize=2, c='blue',
                           linestyle='None')

    def _plot_age_model(self, choice_idx):

        self._plot_calibration(self.interp, ax=self.age_model_ax, c='green',
                               title='age model from medoid')

    def _set_age_model(self, choice_idx):
        x = self.timesteps
        xp = self.timesteps[choice_idx]
        fp = self.medoid[choice_idx]
        self.interp = np.interp(x, xp, fp)

    def _get_sample_calibrated_ts(self, sample_idx):

        sample_r_ages = self.site[sample_idx]
        calibrated_samples = np.interp(self.timesteps[sample_idx] + sample_r_ages,
                                       self.timesteps + self.interp,
                                       self.timesteps)
        return calibrated_samples

    def _process_click(self, event):
        if not event.inaxes == self.map_ax:
            return
        y_pos = int(event.ydata)
        x_pos = int(event.xdata)

        self.clicked_pos = (y_pos, x_pos)
        self._set_ts_at_loc()
        self._plot_marker()

        self._plot_calib_at_location()

        self._set_medoid_at_loc()
        self._plot_medoid_calib()
        self._plot_medoid_marker()

        choice_idx = self._select_points_from_ts(self.medoid)
        self._plot_medoid_samples(choice_idx)
        self._set_age_model(choice_idx)
        self._plot_age_model(choice_idx)

        sample_idx = self._select_points_from_ts(self.site)
        self._plot_site_samples(sample_idx)
        calibrated_sample = self._get_sample_calibrated_ts(sample_idx)
        y = self.timesteps[sample_idx] + self.site[sample_idx]
        self.age_model_ax.plot(
            calibrated_sample,
            y,
            c='blue', marker='o', markersize=2, linestyle='None')
        self.compare_ax.plot(self.timesteps, self.timesteps)
        self.compare_ax.plot(self.timesteps[sample_idx],
                             calibrated_sample)
        self.compare_ax.set_xlabel('true age')
        self.compare_ax.set_ylabel('reconstructed age')
        self.fig.canvas.draw()
