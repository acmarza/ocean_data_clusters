from math import log
from nccluster.base import Workflow


class RadioCarbonWorkflow(Workflow):
    '''Base for workflows involving dc14 and radiocarbon age calculations.'''
    def _preprocess_ds(self):
        # first call the parent's preprocessing function
        super()._preprocess_ds()

        # for computing radiocarbon ages, will use nctoolkit's DataSet.assign,
        # however this will require knowing the variable names in advance;
        # to avoid confusion, rename the variables used in this computation
        # to something simple and consistent
        print("[i] Preprocessing data")
        self.__rename_vars(['dc14', 'dic', 'di14c'])

        # compute dc14 if not present in dataset
        # and provided we have the necessary vars
        if 'dic' in self._ds.variables and\
                'di14c' in self._ds.variables and\
                'dc14' not in self._ds.variables:
            self.__compute_dc14()

        # verify we have the other required data and compute R-age
        self.__check_mean_radiocarbon_lifetime()
        self.__check_atm_dc14()
        self.__compute_R_age()

    def __rename_vars(self, vars):
        # initialize empty dictionary
        rename_dict = {}

        # for each variable that needs renaming, note its original name
        for var in vars:
            try:
                # try to read original name from config
                rename_dict[var] = self.config['radiocarbon'][var]
            except KeyError:
                # if option not defined, assume user does not need it
                # and continue without asking
                pass

        # rename variables to be used in calculations for easy reference
        for key, value in rename_dict.items():
            self._ds.rename({value: key})
            print(f"[i] Renamed variable {value} to {key}")

    def __check_mean_radiocarbon_lifetime(self):
        self._check_config_option(
            'radiocarbon', 'mean_radiocarbon_lifetime',
            missing_msg="[!] Mean lifetime of radiocarbon was not provided",
            input_msg="[>] Enter mean radiocarbon lifetime \
                                        (Cambridge=8267, Libby=8033): ",
            confirm_msg="[i] Mean lifetime of radiocarbon: ")

    def __check_atm_dc14(self):
        self._check_config_option(
            'radiocarbon', 'atm_dc14',
            missing_msg="[!] Atmospheric dc14 was not provided",
            input_msg="[>] Type the atmospheric dc14 as integer in per mil: ",
            confirm_msg="[i] Proceeding with atmospheric dc14 = "
        )

    def __compute_R_age(self):
        # using mean radiocarbon lifetime, dc14, atm_dc14
        mean_radio_life =\
            self.config['radiocarbon'].getint('mean_radiocarbon_lifetime')
        atm_dc14 = self.config['radiocarbon'].getint('atm_dc14')
        self._ds.assign(R_age=lambda x: -mean_radio_life*log(
            (x.dc14/1000 + 1)/(atm_dc14/1000 + 1))
                       )
        print("[i] Converted dc14 to age")

    def __compute_dc14(self):
        # from dic and di14c
        self._ds.assign(dc14=lambda x:
                        (x.di14c/x.dic-1)*1000)
        print("[i] Computed dc14 from di14c and dic")

    def _setters(self):
        # will make use of this array often enough that it's worth saving
        self._age_array = self.ds_var_to_array('R_age')
