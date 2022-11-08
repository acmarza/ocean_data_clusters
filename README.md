
# Ocean Data Clusters

A python script that reads in a NetCDF data file defined in three or four dimensions (x, y, optionally z, plus time) and, interactively or based on a config, walks the user through visualizing the data on a map, selecting variables for further analysis, and finding clusters in the data using the k-means method.

## Installation

1. Clone the repository:
```bash
git clone --depth=1 https://gitlab.com/earth15/ocean_data_clusters.git
cd ocean_data_clusters

```
2. Install the required python modules:
```bash
pip install configparser matplotlib netCDF4 numpy pandas scikit-learn tqdm
```
3. Edit the example configuration file in folder 'configs' as required.
4. Run the script:
```bash
python k_means.py -c configs/example.conf
```
