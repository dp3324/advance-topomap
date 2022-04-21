# Execution Steps:

The `original.ipynb` file is the main tool. Run this file in jupyter notebook to see the results.

### Environment:
`MiniForge` conda environment is used with `Python 3.9.x`. you can find `MiniForge` installtion [here](https://github.com/conda-forge/miniforge).

### Library Requirements:
1) `math`: for finding angle cosine and sin.
2) `random`: creating random colors.
3) `numpy`: create numpy data structure for variables.
4) `matplotlib`: plotting the results.
5) `sklearn`: for the datasets.
6) `scipy`: for convex hull spatial function.

### Tweaks:
The following configuration can be changed as per the requirement.
1) `Dataset`: include 3 sklearn datasets. `dtset_name = ['iris', 'wine', 'breast_cancer'][2]`
2) `MST cost function`: Change the MST cost function for creating minimum spanning tree. `mst_fn_name = ['euclidian_distance', 'angle'][1]`
3) `Output`: directory name to save the output. `output_dir = 'output'`

