import argparse
import configparser
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np

from math import log
from stats_viewer import CorrelationViewer

print("[i] Started correlation analysis workflow")

# define and parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="file to read configuration from, \
                    if parameters not supplied interactively")
args = parser.parse_args()

# parse config file if present
try:
    config = configparser.ConfigParser()
    config.read(args.config)
    print(f"[i] Read config: {args.config}")
except Exception:
    print('[!] Config file not passed via commandline')

# load data from file, asking user to specify a path if not provided in
try:
    nc_file = config['default']['nc_file']
except (KeyError, NameError):
    print('[i] Data file path not provided')
    nc_file = input("[>] Please type the path of the netCDF file to use: ")
print(f"[i] NetCDF file: {nc_file}")

# load in the data
ds = nc.open_data(nc_file)

# get the name of the Delta14C variable in dataset from config or interactively
try:
    dc14_var_name = config['radiocarbon']['dc14']
except (NameError, KeyError):
    print("[!] Name of the Delta14Carbon variable was not provided")
    dc14_var_name = input("[>] Enter Delta14Carbon variable name \
                          as it appears in the dataset: ")

# rename Delta14C variable to dc14 for easy reference
ds.rename({dc14_var_name: 'dc14'})
print(f"[i] Renamed variable {dc14_var_name} to dc14")

# get the mean radiocarbon lifetime from config or interactively
try:
    mean_radio_life = int(config['radiocarbon']['mean_radiocarbon_lifetime'])
except (NameError, KeyError):
    print("[!] Mean lifetime of radiocarbon was not provided")
    mean_radio_life = int(input("[>] Enter mean radiocarbon lifetime \
                                (Cambrdige=8267, Libby=8033): "))

# compute local age from Delta14C
ds.assign(local_age=lambda x: -mean_radio_life*log((1000+x.dc14)/1000))
print(
    f"[i] Converted dc14 to age using mean radioC lifetime {mean_radio_life}"
)

# get the raw data array of local ages
xr_ds = ds.to_xarray()
age_array = xr_ds['local_age'].__array__()

# slice before anomaly specific to our data
# age_array = age_array[:78]

# offset ages to make more sense (else they'd be negative)
min_age = np.nanmin(age_array)
age_array -= min_age

# find out from config or interactively whether user wants to plot all time
# series on one plot
try:
    pvalues = config['correlation'].getboolean('pvalues')
except (NameError, KeyError):
    yn = input("[>] Mask out grid point with insignificant \
               ( p > 0.05 ) correlation? (y/n): ")
    pvalues = (yn == 'y')

# visualize
viewer = CorrelationViewer(age_array, 'Ages', pvalues=pvalues)
plt.show()
