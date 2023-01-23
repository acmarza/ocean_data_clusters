import atexit
import configparser
import nctoolkit as nc

from nccluster.multisliceviewer import MultiSliceViewer


class Workflow:
    '''Base class for defining a workflow that operates on a netCDF dataset'''
    def __init__(self, config_path):
        # save the config_path as an attribute, read config and
        # be ready to save config changes  when script exits
        self.config_path = config_path
        self.__read_config_file()
        atexit.register(self.__offer_save_config)

        # make sure all required fields are defined in the config
        # before loading the data from file and applying any preprocessing
        self._checkers()
        self.__load_ds()
        self._preprocess_ds()

        # initialise any other required attributes
        self._setters()

    def __read_config_file(self):
        # parse config file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        print(f"[i] Using config: {self.config_path}")

    def __offer_save_config(self):
        # will only ask to save if a new options is added to config that's
        # not already in the config file
        # first read in the original config again
        original_config = configparser.ConfigParser()
        original_config.read(self.config_path)

        # then for each config section and option, check for additions
        for section in dict(self.config.items()).keys():
            for option in dict(self.config[section]).keys():
                if not original_config.has_option(section, option):
                    print(f"[i] The script is about to exit, but you have \
unsaved changes to {self.config_path}.")
                    yn = input("[>] Save modified config to file? (y/n): ")
                    if yn == 'y':
                        with open(self.config_path, 'w') as file:
                            self.config.write(file)
                    return

    def _check_config_option(self, section, option,
                             missing_msg="[!] Missing option in config",
                             input_msg="[>] Please enter value: ",
                             confirm_msg="[i] Got value: ",
                             default=None,
                             required=True,
                             isbool=False,
                             force=False):
        # convenience function to check if an option is defined in the config;
        # if not, interactively get its value and add to config
        try:
            # don't attempt to read from config if option is forced
            # move straight to except block where default should be applied
            if force:
                raise KeyError
            # otherwise try to read value from config
            value = self.config[section][option]
        except KeyError:
            # do nothing if option missing from config and not required
            if not required:
                return

            # apply default or ask user to input a value
            if default is not None:
                value = default

            else:
                # ask the user for the value
                print(missing_msg)
                value = input(input_msg)

                # for bools expect a y/n answer
                if isbool:
                    value = (value.lower() == 'y')

            # may need to create a new section
            if not self.config.has_section(section):
                self.config.add_section(section)

            value = str(value)
            # set the option we've just read in
            self.config[section][option] = value

        # either way echo the value to the user for sanity checking
        print(confirm_msg + value)

    def __check_nc_files(self):
        self._check_config_option(
            'default', 'nc_files',
            missing_msg='[i] Data file path not provided',
            input_msg='[>] Please type the path of the netCDF file to use: ',
            confirm_msg='[i] NetCDF file(s): '
        )

    def __check_vars_subset(self):
        self._check_config_option(
            'default', 'subset',
            required=False,
            confirm_msg="[i] Data will be subset keeping only variables: "
        )

    def __check_surface_only(self):
        self._check_config_option(
            'default', 'surface_only',
            default=False, isbool=True,
            confirm_msg='[i] Restrict analysis to ocean surface: '
        )

    def __check_timesteps_subset(self):
        self._check_config_option(
            'default', 'timesteps_subset',
            required=False,
            confirm_msg='[i] Data will be subset to the following timesteps: '
        )

    def __load_ds(self):
        print("[i] Loading data")

        # get a list of file paths and create the DataSet object
        nc_files = self.config['default']['nc_files'].split(",")
        self._ds = nc.DataSet(nc_files)

    def _preprocess_ds(self):
        # optionally limit analysis to a subset of variables
        if self.config.has_option('default', 'vars_subset'):
            subset = self.config['default']['vars_subset'].split(",")
            print("[i] Subsetting data")
            self._ds.subset(variable=subset)

        # optionally limit analysis to the ocean surface (level 0):
        if self.config['default'].getboolean('surface_only'):
            self._ds.top()

        # optionally limit analysis to an interval of time steps
        if self.config.has_option('default', 'timesteps_subset'):
            str_opt = self.config['default']['timesteps_subset']
            start, end = str_opt.strip('[]').split(",")
            self._ds.subset(timesteps=range(int(start), int(end)))

        # merge datasets if multiple files specified in config
        try:
            print("[i] Merging datasets")
            self._ds.merge()
        except Exception:
            print("[fatal] You have specified multiple netCDF files to use",
                  "but these could not be merged for further analysis.",
                  "Consider subsetting the data as shown in the example",
                  "config, keeping only variables of interest. Or merge your",
                  "datasets externally.")
            exit()

    def _checkers(self):
        self.__check_nc_files()
        self.__check_vars_subset()
        self.__check_surface_only()
        self.__check_timesteps_subset()

    def _setters(self):
        print("[i] All attributes have been initialized")

    def regrid_to_ds(self, target_ds):
        # interpolate/extrapolate the data to fit the grid of a target dataset
        self._ds.regrid(target_ds)
        # features and time series need to be re-created for new grid
        self._setters()

    def run(self):
        print("[!] Nothing to run. Did you forget to override self.run()?")

    def list_plottable_vars(self):
        # get an alphabetical list of plottable variables
        plottable_vars = self._ds.variables

        # easy reference to a dataframe containing useful info on variables
        df = self._ds.contents

        # list plottable variables for the user to inspect
        for i, var in enumerate(plottable_vars):
            try:
                # print extra info if possible
                long_name = df.loc[df['variable'] == var].long_name.values[0]
                print(f"{i}. {var}\n\t{long_name}")
            except Exception:
                # just print the variable name
                print(f"{i}. {var}")

    def plot_var(self, var_to_plot):
        try:
            # get the data as an array
            ds_tmp = self._ds.copy()
            ds_tmp.subset(variables=var_to_plot)
            var_xr = ds_tmp.to_xarray()
            data_to_plot = var_xr[var_to_plot].values

            try:
                # add more info in plot title if possible
                plot_title = var_to_plot + " ("\
                    + var_xr[var_to_plot].long_name + ") "\
                    + var_xr[var_to_plot].units
            except AttributeError:
                # stick to basic info if the above goes wrong
                plot_title = var_to_plot

            # pass on the data array to interactive viewer
            MultiSliceViewer(data_to_plot, plot_title).show()

        except IndexError:
            print(f"[!] {var_to_plot} not found; check spelling")

    def ds_var_to_array(self, var_name):
        # some  manipulation to isolate age data and cast it into a useful form
        # make a copy of the original dataset to subset to one variable
        ds_tmp = self._ds.copy()
        ds_tmp.subset(variables=var_name)

        # convert to xarray and extract the numpy array of values
        as_xr = ds_tmp.to_xarray()
        as_array = as_xr[var_name].values

        return as_array
