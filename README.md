
# Ocean Data Clusters

A collection of python scripts that use unsupervised machine learning techniques to define clusters in ocean data (but could be used with other netCDF data with 2 or 3 spatial dimensions and a time dimension).

## Installation

Clone the repository::
```bash
git clone https://gitlab.com/earth15/ocean_data_clusters.git

```
Create a new conda environment and install the required python packages:
```bash
conda create -n nccluster -c conda-forge configparser nctoolkit tqdm tslearn
```
Activate the environment:
```bash
conda activate nccluster
```
## Usage
Edit the example configuration file in folder 'configs' as needed (see explanations therein). The only crucial value to specify for a first run is nc_files (path to netCDF data file(s)); the script will walk you through setting the other values.
### K-means clustering

1. Run the script:
```bash
python main.py --method km --config configs/example.conf
```

2. Interacting with the plots:
- left-hand plot is the main view
- right-hand plot is a surface slice across the z axis; a black line will appear here to help you locate slices taken perpendicular to x or y
- **left/right** arrow to move backward/forward in time
- **up/down** arrow to navigate the current spatial dimension that the viewer is slicing across
- press **x, y, z** to change the main view; a line on the plot below will appear to help you locate the current slice on a map
- **q** to close viewer and continue workflow

### Correlation clustering
1. Run the script:
```
python main.py --method corr --config configs/example.conf
```
2. Interaction:
- upper-left plot is the main view
- upper-right plot is a surface slice across the z axis; a black line will appear here to help you locate slices taken perpendicular to x or y; 
- lower-left plot is the correlation map, click anywhere on this plot to set a location for further analysis (a marker will appear where you clicked), showing the correlation coefficient of every other map point relative to the one you clicked
- lower-left plot is the cluster map, this shows the results of agglomerative clustering (the flattening of the hierarchy into clusters can change depending on the exact methods and threshold used; see the input fields and button selections on the right for options)
- bottom plot is the evolution plot, this will activate once you click a point on the surface slice, showing the evolution of radiocarbon ages at that location in time; a vertical line appears when you change time-step in the main view, to help you locate yourself in time
- **left/right** arrow to move backward/forward in time
- **up/down** arrow to navigate the current spatial dimension that the viewer is slicing across
- press **x, y, z** to change the main view; a line on the plot below will appear to help you locate the current slice on a map
- **click** on the lower-right correlation map to set a location for analysis
- **q** to close viewer and continue workflow

### Timeseries clustering
1. Run the script:
```
python main.py --method corr --config configs/example.conf
```
2. Explanations of plots:
- the first plot that appears shows the timeseries assigned to each cluster (thin black lines) and the cluster barycenter (red line)
- the second plot is a map of the clustering results

If you can only see one plot, try dragging the window to the side as the plots may overlap.
