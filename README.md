# Large-scale transformation prediction
Large-scale transformation prediction (LSTP) consists of three steps: time-constraing clustering, LSTP training and LSTP inference. We provide a separate script for each step.

## Data retrieval
The data from the corresponding paper can be retrieved from Dataverse via wget

```bash
wget "https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/07RT92/18ee0ea2f7d-7e024b0676ef?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27D1_rbc_vertical.npy&response-content-type=application%2Foctet-stream&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240917T112715Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240917%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a21875b2c9f00ee7cddbfe7a4b225c10cf7626f0861d4174d4c03639798c4fd7" -O D1.npy
wget "https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/07RT92/18ee0e9cf59-6c494593a632?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27D2_rbc_horizontal.npy&response-content-type=application%2Foctet-stream&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240917T113509Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240917%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=67246e13f3c96edd1d0279452f11febfe9e21163717834941fc4680aca1c8653" -O D2.npy
wget "https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/07RT92/18ee0ea4d7a-4581600b0dfc?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27D3_vkf.npy&response-content-type=application%2Foctet-stream&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240917T113530Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240917%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e1c8b97eec46829b2b993397e1e3f055f2399e4956eb3fcb40d1906e37ca3c6d" -O D3.npy
```

Alternatively, you can follow this link: https://doi.org/10.7910/DVN/07RT92 and download the data in your favorite browser.

You can also use a custom snapshot data set. Ensure that the snapshots are ordered in time and saved as numpy array in a numpy file (.npy) with the shape (snapshot, 1, height, width). Create a checkpoints and a data folder in the root directory of the project:

```bash
mkdir /path/to/project/checkpoints
mkdir /path/to/project/data
```

Move the data set into the data folder (example D1.npy):
```bash
mv /path/to/D1.npy /path/to/project/data
```

Modify the config file (config.yaml) as needed. It contains an example config, which should be straight forward to adapt. At least the file name base (data_fname) of your dataset should be set accordingly.

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
cd /path/to/project
python3 main.py
```

On the first run, you will want to train the LSTP model from scratch. In later runs you can set the load_cp: true in the config and it will try to load the checkpoint that is created during the first run.
The first step is the clustering, and the script will create a visualization of the clusters and save it as peaks.png.
The training will be the most time intensive step. Therefore, the script saves the checkpoint of the best validation loss epoch in the checkpoints directory. The checkpoint will be named according to the cp_fname that was set in the config.
During inference the model predicts the transformations between structures from the clustering. In the end it will output the support value for each augmentation of the augmentation pool (it is currently fixed to the augmentations used in the paper).
