import argparse
import nctoolkit as nc
import numpy as np

from math import log
from slice_viewer import MultiSliceViewer

# only argument is netCDF file path, for now
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", help="netCDF data file")
args = parser.parse_args()

# load in the data
ds = nc.open_data(args.file)

# compute local age from Delta14C
ds.assign(local_age=lambda x: -8267*log((1000+x.O_DC14)/1000))

# get the data array
xr_ds = ds.to_xarray()
age_array = xr_ds['local_age'].__array__()

# correct ages so they're not negative
min_age = np.nanmin(age_array)
age_array -= min_age

# visualize
viewer = MultiSliceViewer(age_array, 'Ages')
