# Large-scale transformation prediction
Large-scale transformation prediction (LSTP) constists of three steps: time-constraing clustering, LSTP training and LSTP inference. We provide a separate script for each step.

To perform LSTP for a given data set, it has to be added to the data dir as a numpy file (.npy). The data set should have the shape (snapshots, 1, height, width). If there is not data directory, create one. Modify the config file (config.yaml) as needed and run the main.py script.

```bash
python3 main.py
```
