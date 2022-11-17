import argparse
import matplotlib.pyplot as plt
import nctoolkit as nc
import numpy as np
import pandas as pd
import pickle

from math import log
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", help="netCDF data file")
parser.add_argument("--pickle", "-p", help="pickle  file to save/read\
                    clustering results")
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
# slice before anomaly
evolutions = evolutions[:, :78]
t = 78
# code below plots evolution of every grid point over time (fun to look at)
for point in tqdm(range(0, x*y)):
    plt.plot(range(0, t), evolutions[point, :])
plt.xlabel('time step')
plt.ylabel('age')
plt.title('age over time')
plt.show()
# convert to pandas dataframe to drop NaN entries, and back to array
df = pd.DataFrame(evolutions)
evolutions = np.array(df.dropna())
ts = to_time_series_dataset(evolutions)

n_clusters = 4
try:
    with open(args.pickle, 'rb') as file:
        km = pickle.load(file)
        print(f"[i] Read in {args.pickle}")
except FileNotFoundError:
    km = TimeSeriesKMeans(n_clusters=n_clusters,
                          metric="euclidean",
                          max_iter=10,
                          verbose=1,
                          n_jobs=-1
                          )
    km.fit(ts)
    with open(args.pickle, 'wb') as file:
        pickle.dump(km, file)

y_pred = km.predict(ts)
plt.figure()
for yi in range(n_clusters):
    plt.subplot(n_clusters, 1, yi + 1)
    for xx in ts[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
plt.title("Soft-DTW $k$-means")
plt.show()

df.loc[
    df.index.isin(df.dropna().index),
    'labels'] = km.labels_
labels_flat = np.ma.masked_array(df['labels'])
labels_shaped = np.reshape(labels_flat, [x, y])

plt.imshow(labels_shaped, origin='lower')
plt.show()
