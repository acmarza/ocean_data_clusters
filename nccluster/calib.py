from nccluster.radiocarbon import RadioCarbonWorkflow
from nccluster.utils import make_subclusters_map
import numpy as np
import matplotlib.pyplot as plt


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
        self.fig, (self.map_ax, self.calib_ax) = plt.subplots(1, 2)
        self.interactive_calib_plot()

    def interactive_calib_plot(self):
        base_map = self._make_base_map()
        self.map_ax.imshow(base_map, origin='lower')
        self.fig.canvas.mpl_connect('button_press_event',
                                    self._process_click)
        plt.show()

    def _make_base_map(self):
        return np.isnan(self.R_ages[0])

    def _process_click(self, event):
        if not event.inaxes == self.map_ax:
            return
        y_pos = int(event.ydata)
        x_pos = int(event.xdata)
        self.plot_calibration(y_pos, x_pos)

        # clear previous marker if present
        try:
            for handle in self.loc_marker:
                handle.remove()
        except AttributeError:
            pass
        self.loc_marker = self.map_ax.plot(x_pos, y_pos,
                                           color='black',
                                           marker='*'
                                           )
        self.fig.canvas.draw()

    def plot_calibration(self, y_loc, x_loc):

        R_age_history = self.R_ages[:, y_loc, x_loc]

        self.calib_ax.cla()
        self.calib_ax.plot(self.timesteps, self.timesteps + R_age_history)
        self.calib_ax.plot(self.timesteps, self.timesteps)
        self.calib_ax.set_xlabel('atmosphere age')
        self.calib_ax.set_ylabel('ocean age')
