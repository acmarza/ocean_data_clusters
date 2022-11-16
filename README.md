
# Ocean Data Clusters

A python script that reads in a NetCDF data file defined in three or four dimensions (x, y, optionally z, plus time) and, interactively or based on a config, walks the user through visualizing the data on a map, selecting variables for further analysis, and finding clusters in the data using the k-means method.

## Installation

Clone the repository::
```bash
git clone --depth=1 https://gitlab.com/earth15/ocean_data_clusters.git

```
Install the required python modules:
```bash
pip install configparser matplotlib nctoolkit netCDF4 numpy pandas scikit-learn scipy tqdm
```
or
```bash
conda install -c conda-forge configparser matplotlib nctoolkit netCDF4 numpy pandas scikit-learn scipy tqdm
```
## Usage
### K-means clustering
1. Edit the example configuration file in folder 'configs' as needed (see explanations therein). The only crucial value to specify for a first run is nc_file (path to netCDF data file); the script will walk you through setting the other values.

2. Run the script:
```bash
python k_means.py -c configs/example.conf
```

3. Interacting with the plots:
- left-hand plot is the main view
- right-hand plot is a surface slice across the z axis; a black line will appear here to help you locate slices taken perpendicular to x or y
- **left/right** arrow to move backward/forward in time
- **up/down** arrow to navigate the current spatial dimension that the viewer is slicing across
- press **x, y, z** to change the main view; a line on the plot below will appear to help you locate the current slice on a map
- **q** to close viewer and continue workflow

### Radiocarbon ages & correlation
1. Run the script:
```
python radiocarbon_anomaly.py -f datafile.nc --dc14 O_DC14
```
Hint:
```
usage: radiocarbon_anomaly.py [-h] [--file FILE]
                              [--dc14 DC14] [--pvals]

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  netCDF data file
  --dc14 DC14           the name of the Delta14C variable
                        as it appears in the dataset
  --pvals, -p           pass this flag to ignore grid
                        points with statistically
                        insignificant correlation when
                        plotting the correlation map
```
Note: the -p flag will change the correlation analysis from a quick numpy.corrcoef to the much slower scipy.stats.stats.pearsonr. A progress bar appears to estimate remaining time. The results are saved by default to 2 files callef corr_mat.npy and pval_mat.npy, to be read on the next script run instead of computing these all over again.

2. Interaction:
- upper-left plot is the main view
- upper-right plot is a surface slice across the z axis; a black line will appear here to help you locate slices taken perpendicular to x or y; click anywhere on this plot to set a location for further analysis (a marker will appear where you clicked)
- lower-left plot is the evolution plot, this will activate once you click a point on the surface slice, showing the evolution of radiocarbon ages at that location in time; a vertical line appears when you change time-step in the main view, to help you locate yourself in time
- lower-right plot is the correlation map, this will activate once you click a point on the surface slice, showing the correlation coefficient of every other map point relative to the one you clicked
- **left/right** arrow to move backward/forward in time
- **up/down** arrow to navigate the current spatial dimension that the viewer is slicing across
- press **x, y, z** to change the main view; a line on the plot below will appear to help you locate the current slice on a map
- **click** on the upper-right surface slice or lower-right correlation map to set a location for analysis
- **q** to close viewer and continue workflow
