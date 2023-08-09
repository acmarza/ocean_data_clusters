
# Ocean Data Clusters

Find clusters in ocean data using unsupervised machine learning. Could be used with similar netCDF data that have 2 or 3 spatial dimensions and a time dimension.

## Setup
### 1. Python + Conda

1. Clone the repository:
```
git clone https://gitlab.com/earth15/ocean_data_clusters.git
cd ocean_data_clusters
```
2. Create a conda environment from the file provided:
```
conda env create -f environment.yml
```
3. Activate the environment:
```
conda activate nccluster
```
### 2. Docker
1. Download the docker image:
```
docker pull anamarza/nccluster
```
2. Download this git repo:
```
git clone https://gitlab.com/earth15/ocean_data_clusters.git
```

3. Note the paths of the downloaded git repo and your data folder.
4. Start up the container:
```
docker run -id -p 5901:5901 --name nccluster --mount type=bind,source=/path/to/ocean_data_clusters,target=/app  --mount type=bind,source=/path/to/data,target=/data anamarza/nccluster
```
5. Use your preferred VNC viewer to connect to the container at address localhost:5901, for example:
```
vncviewer localhost:5901
```
6. Right-click and open a terminal emulator.
7. Navigate to the app folder: ```cd /app```

We are now ready to use the programs.

## Basic usage
Edit the example configuration file in folder 'configs' as needed (see explanations therein). 

### Data exploration
```
python examples/data_explorer.py ---config configs/example.conf
```
Interact with the two views using arrow keys:
- **left-right** to navigate in time (both plots);
- **up-down** to navigate in space (left-hand plot).

Change the cross-section direction by pressing **x, y, z**. For the x and y views, a line appears on the right-hand plot to show you the location of the cross-section.

Press **q** to close the figure.

### View combined time series
```
python examples/workflows_demo.py --method all_ts --config configs/example.conf
```

### Time series clustering
```
python examples/workflows_demo.py --method ts --config configs/example.conf
```
For more advanced analysis, the two-step method first detects shape-based clusters in the normalised time series, then splits these further into amplitude-based subclusters:
```
python examples/workflows_demo.py --method two --config configs/example.conf
```
### Dendrogram
```
python examples/workflows_demo.py --method dendro --config configs/example.conf
```
On the left you can set options for the agglomerative clustering.
Top-right plot shows clustering results on map.
Bottom-right plot shows the dendrogram. **Click** to set a new threshold for flattening the dendrogram (only applies when f-cluster criterion is set to distance).

### Correlation clustering
```
python examples/workflows_demo.py --method corr --config configs/example.conf
```
Depending on your config file, you
Explanation of the interface:

- 2x2 table of maps:
	+ top row is the [data explorer](#data-exploration)
	+ bottom-left  is the correlation map, **click** anywhere on this plot to set a location for further analysis (a marker will appear where you clicked), showing the correlation coefficient of every other map point relative to the one you clicked
	+ bottom-right plot is the cluster map, this shows the results of agglomerative clustering and the procedure can be modified based on the controls to the right, see [Dendrogram](#dendrogram)
- bottom plot is the evolution plot, this will activate once you click a point on the surface slice, showing the evolution of radiocarbon ages at that location in time; a vertical line appears when you change time-step in the main view, to help locate the timeslice

### K-means clustering
```
python examples/workflows_demo.py --method km --config configs/example.conf
```
This proceeds interactively, follow the instructions. The clustering results are shown using the [data explorer](#data-exploration). Unlike time series clustering, the clusters are re-computer at each timestep based on several variables, so you can navigate through time.

### Clustering metrics
```
python examples/workflows_demo.py --method metrics --config configs/example.conf
```
Each iteration runs the [ts](#time-series-clustering) method with an increasing number of clusters. Can take several minutes to complete depending on the method, maximum number of iterations, and number of initializations as defined in the config file.

In the sum of squared errors (top-most plot), the vertical line denotes the elbow/knee point.

## Saving and re-using labels
Ensure you have familiarized yourself with at least the [data explorer](#data-exploration) and [time series clustering](#time-series-clustering). For now only saves/loads labels created by time series clustering.

### Save labels to file after time series clustering
```
python examples/save_labels.py --config configs/example.conf --method [ts|two] --file /some/path/to/savefile
```
### Load labels from file and plot
```
python examples/load_labels.py --config configs/example.conf --method [ts|two] --file /some/path/to/savefile
```
### Compare saved labels between two data sets
```
python examples/compare_demo.py --left /path/to/first/file.nc --right /path/to/second/file.nc
```
Press **q** to dismiss each consecutive plot. The script will automatically regrid and reorder the labels to improve overlap between the two maps, showing each step until the overlap is computed and illustrated with Venn diagrams, at which point pressing q again will exit the program.

To skip the intermediate steps, pass the flags --no-regrid (only if you know for sure that the maps are the same size!) and/or --no-match, for example:
```
python examples/compare_demo.py --left /path/to/first/file.nc --right /path/to/second/file.nc --no-regrid --no-match
```
### Delta-R histograms
Apply labels from one data set (described by original.conf) to another data set (described by example.conf):
```
python examples/DdR.py --config configs/example.conf --labels /path/to/savefile.nc --original configs/original.conf

```
The labels must correspond to the configuration file passed as --original!
Click on the map to inspect a subcluster. The cosine similarity is plotted for the selected subcluster. The subcluster medoid is marked by a green star. On the right, the bottom plot shows the R-ages for grid points in this subcluster in grey, the subcluster centroid in blue, the cluster centroid in orange, the global surface mean R-age in green. The differences between each of the three benchmarks the and the subcluster time series are summarized as density plots in the top-right plot.

### Delta-R stats maps
Apply labels from one data set (described by original.conf) to another data set (described by example.conf):
```
python examples/map_DdR.py --config configs/example.conf --labels /path/to/savefile.nc --original configs/original.conf

```
The script proceeds interactively. Type a number and hit enter to plot the desired maps.