# Large-scale transformation prediction
Large-scale transformation prediction (LSTP) consists of three steps: time-constraing clustering, LSTP training and LSTP inference. We provide a separate script for each step.

## Data retrieval
The data from the corresponding paper can be retrieved from Dataverse at https://doi.org/10.7910/DVN/07RT92. Download the data via your favorite browser.

You can also use a custom snapshot data set. Ensure that the snapshots are ordered in time and saved as numpy array in a numpy file (.npy) with the shape (snapshot, 1, height, width). Create a checkpoints and a data folder in the root directory of the project:

```bash
mkdir /path/to/project/checkpoints
mkdir /path/to/project/data
```

Move the data set into the data folder (example D1.npy):
```bash
mv /path/to/D1.npy /path/to/project/data
```

Change the working directory to the project directory:

```bash
cd /path/to/project
```

Modify the config file (config.yaml) as needed. It contains an example config, which should be straight forward to adapt. At least the file name base (data_fname) of your dataset and the checkpoint file name (cp_fname) should be set accordingly.

```yaml
data_params:
  ...
  data_fname: D1
  ...
lstp_params:
  ...
  cp_fname: runXY
  ...
```

Exectue the LSTP procedure by running the main.py script.

```bash
python3 main.py
```

On the first run, you will want to train the LSTP model from scratch. In later runs you can set the load_cp: true in the config and it will try to load the checkpoint that is created during the first run.
The first step is the clustering, and the script will create a visualization of the clusters and save it as peaks.png.
The training will be the most time intensive step. Therefore, the script saves the checkpoint of the best validation loss epoch in the checkpoints directory. The checkpoint will be named according to the cp_fname that was set in the config.
During inference the model predicts the transformations between structures from the clustering. In the end it will output the support value for each augmentation of the augmentation pool (it is currently fixed to the augmentations used in the paper).
