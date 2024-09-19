import yaml
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
import os

from lstp_data import LSTPData
from lstp_controller import LSTPController
from time_constrained_clustering import TimeConstrainedClustering

# check workspace
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# load config
print('Loading config...')
try:
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print(f'Could not load config: {e}')

# set seeds
seed = config['seed'] if 'seed' in config else 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# load raw data
print('Loading data...')
data_fname = config['data_params']['data_fname']
raw_data = np.load(os.path.join('data', f'{data_fname}.npy'))

# time-constraing clustering
print('Time-constrained clustering...')
clustering = TimeConstrainedClustering(raw_data, config)
clustering.fit()

# lstp training
## data preparation
print('Data preparation...')
c_size = config['data_params']['c_size']
data = np.array([raw_data[i: i + c_size].mean(0) for i in range(raw_data.shape[0] - c_size + 1)])
idxs = np.arange(len(data))
if 'data_params' in config and 'pre_shuffle' in config['data_params'] and config['data_params']['pre_shuffle']:
    np.random.shuffle(idxs)

if 'data_params' in config and 'samples' in config['data_params']:
    samples = config['data_params']['samples']
else:
    samples = len(data)

train_samples = int(samples * 0.8)
valid_samples = int(samples * 0.1)
test_samples = samples - train_samples - valid_samples

train = data[idxs[:int(len(idxs) * 0.8)]]
valid = data[idxs[int(len(idxs) * 0.8): int(len(idxs) * 0.9)]]
test = data[idxs[int(len(idxs) * 0.9):]]

train_ds = LSTPData(train, config, subset='train', samples=train_samples)
valid_ds = LSTPData(valid, config, subset='valid', data_min=train_ds.data_min, data_max=train_ds.data_max, samples=valid_samples)
test_ds = LSTPData(test, config, subset='test', data_min=train_ds.data_min, data_max=train_ds.data_max, samples=test_samples)

batch_size = train_ds.batch_size
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

config['lstp_params']['batch_size'] = batch_size
config['lstp_params']['num_augs'] = len(train_ds.aug_pool)
lstp = LSTPController(config)
if config['lstp_params']['load_cp']:
    print('Loading checkpoint of LSTP...')
    try:
        lstp.load_cp()
    except Exception as e:
        print(f'Could not load checkpoint: {e}') 
        lstp.train(train_dl, valid_dl)
else:
    print('Training LSTP...')
    lstp.train(train_dl, valid_dl)

# lstp inference
lstp.infer(clustering, test_dl)
