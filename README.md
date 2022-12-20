
# Ocean Data Clusters

A collection of python scripts that use unsupervised machine learning techniques to define clusters in ocean data (but could be used with other netCDF data with 2 or 3 spatial dimensions and a time dimension).

## Installation

Clone the repository:
```
git clone https://gitlab.com/earth15/ocean_data_clusters.git
cd ocean_data_clusters

```
Create a conda environment from the file provided, with all the required python modules:
```
conda env create -f environment.yml
```
Alternatively create a new conda environment and install the required python packages (and good luck solving conflicts):
```
conda create -n nccluster -c conda-forge matplotlib-venn nctoolkit sktime tqdm tslearn xesmf
```
Activate the environment:
```
conda activate nccluster
```
## Usage
Edit the example configuration file in folder 'configs' as needed (see explanations therein). 
### K-means clustering

1. Run the script:
```
python examples/workflows_demo.py --method km --config configs/example.conf
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
python examples/workflows_demo.py --method corr --config configs/example.conf
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
python examples/workflows_demo.py --method ts --config configs/example.conf
```
For more advanced analysis, the two-step method first detects shape-based clusters in the normalised time series, then splits these further into amplitude-based subclusters:
```
python examples/workflows_demo.py --method two --config configs/example.conf
```

2. Explanations of results:
- the left-hand plot shows the timeseries assigned to each cluster (thin black lines) and the (sub)cluster barycenters (red line(s))
- the right-hand plot is a map of the clustering results
