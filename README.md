## GNN-IN-CAD

### Environment

Clone the CS5284 course env, then install this project's extra packages into the clone. `gnn_course` stays unchanged.

```sh
# Once, from the CS5284_2026 course repository, if gnn_course does not exist yet
conda env create -f environment_osx_arm64.yml

# Clone it
conda create --name gnn_project --clone gnn_course
conda activate gnn_project
pip install -r requirements.txt
```
