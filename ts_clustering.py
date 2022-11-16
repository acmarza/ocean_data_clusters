import argparse
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np

from math import log
from tqdm import tqdm

# only argument is netCDF file path, for now
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", help="netCDF data file")
parser.add_argument("--dc14", help="the name of the Delta14C variable as it\
                    appears in the dataset")
args = parser.parse_args()

# load in the data
ds = nc.open_data(args.file)

# rename Delta14C variable to dc14 regardless of original name
ds.rename({args.dc14: 'dc14'})

# compute local age from Delta14C
# ds.assign(local_age=lambda x: -8267*log((1000+x.O_DC14)/1000))
ds.assign(local_age=lambda x: -8267*log((1000+x.dc14)/1000))

# get the data array
xr_ds = ds.to_xarray()
age_array = xr_ds['local_age'].__array__()


# slice before anomaly specific to our data
# age_array = age_array[:78]
# correct ages so they're not negative
min_age = np.nanmin(age_array)
age_array -= min_age

# time series of age at each grid point
t, z, y, x = age_array.shape
evolutions = np.reshape(age_array[:, 0], [t, x*y]).T

# code below plots evolution of every grid point over time (fun to look at)
for point in tqdm(range(0, x*y)):
    plt.plot(range(0, t), evolutions[point, :])
plt.xlabel('time step')
plt.ylabel('age')
plt.title('age over time')
plt.show()
