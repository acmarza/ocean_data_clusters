import argparse
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np

from math import log
from stats_viewer import CorrelationViewer

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

# focus on surface
surface_age_array = age_array[:, 0, :, :]

# get x*y number of 1D arrays showing evolution of each gridpoint in time
t, y, x = surface_age_array.shape
evolutions = np.reshape(surface_age_array, [t, x*y]).T
print(evolutions.shape)


# code below plots evolution of every grid point over time (fun to look at)
# for point in range(0, x*y):
#     plt.plot(range(0, t), evolutions[point, :])
# plt.xlabel('time step')
# plt.ylabel('age')
# plt.title('age over time')
# plt.show()

# compute correlation matrix for all data points
corr_mat = np.corrcoef(evolutions)
# plt.imshow(corr_mat)
# plt.show()

# visualize
viewer = CorrelationViewer(surface_age_array, corr_mat, 'Ages')
plt.show()
