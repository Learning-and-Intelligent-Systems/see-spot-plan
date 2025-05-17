# see-spot-plan
A collection of tools to connect a Spot robot with vision models and planning utilities. The goal is to be simple, minimal, and easy-to-use.

## Installation
* This repository uses Python versions 3.10-3.11. We recommend 3.10.14.
* Create a new conda environment with `conda create --name see_spot_plan python=3.10.14`
* Run `pip install -e .` to install dependencies.


## Running/examples
Currently, the main script in the repo is `spot_localization.py`. Here is an example command (for the LIS spot Jasper):
```
python spot_utils/spot_localization.py --hostname 192.168.80.3 --map_name b45-621
```
You can 'hijack' the robot via the tablet when prompted, move it around, and then have it print out its pose! We can then command it to navigate to any saved pose via other utilities (which are forthcoming).

## Mapping
> Last Updated: 05/17/2025

Our code is currently designed to operate given a saved map of a particular
environment. The map is required to define a coordinate system that persists
between robot runs. Moreover, each map contains associated metadata information
that we use for visualization, collision checking, and a variety of other
functions.

To create a new map of a new environment:
1. Print out and tape april tags around the environment. The tags are [here](https://support.bostondynamics.com/s/article/About-Fiducials)
2. Run the interactive script from the spot SDK to create a map, while walking
   the spot around the environment. The script is [here](https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/graph_nav_command_line/recording_command_line.py)   
3. Save the map files to spot_utils / graph_nav_maps / <your new env name>
4. Create a file named `metadata.yaml` if one doesn't already exist within the folder
associated with a map. See below for more details.
5. Set --spot_graph_nav_map to your new env name.


**Converting a map to a pointcloud**
1. Run [this script](https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/graph_nav_extract_point_cloud) on the pre-made map to yield an output `.ply` pointcloud file.
[Optional]
2. Install the [Open3D package](http://www.open3d.org/docs/release/getting_started.html) with `pip install open3d`.
3. Open up a python interpreter in your terminal, and run the following commands:
```
import open3d as o3d

pcd = o3d.io.read_point_cloud("<path to your pointcloud file>")
o3d.visualization.draw_geometries_with_editing([pcd])
```
4. Within the window, do SHIFT + left click on a point to print out its 3D coords to the terminal. Copy the first two (x, y) of all the
boundary points into the yaml. Note that you can do SHIFT + right click to unselect a point.