import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import transform

class Augmentations:
    @staticmethod
    def x_flip(x, p=1.0):
        x = torch.flip(x, dims=[2])
        return x

    @staticmethod
    def y_flip(x, p=1.0):
        x = torch.flip(x, dims=[1])
        return x

    @staticmethod
    def x_cyclic_shift(x, p=0.5):
        offset = 4
        shift = offset + int((x.shape[2] - (offset + 1)) * p)
        x = torch.roll(x, shifts=shift, dims=2)
        return x

    @staticmethod
    def y_cyclic_shift(x, p=0.5):
        offset = 4
        shift = offset + int((x.shape[1] - (offset + 1)) * p)
        x = torch.roll(x, shifts=shift, dims=1)
        return x

    @staticmethod
    def value_inversion(x, p=1.0):
        x = x.max() - x + x.min()
        return x

    @staticmethod
    def x_shear(x, p=0.5):
        w = x.shape[2]
        x = torch.cat([x[:, :, :], x, x[:, :, :]], dim=2)
        x = x[0]
        
        if p <= 0.5:
            p *= 2.0
            ps = -1.0
        else:
            p = 2.0 * p - 1.0
            ps = 1.0
        
        shear = ps * (0.05 + (p * 0.1))
        
        tf = transform.AffineTransform(shear=shear)
        x = transform.warp(x, tf)
        x = torch.tensor(x[:, w: w + w]).unsqueeze(0)
        return x

    @staticmethod
    def y_shear(x, p=0.5):
        x = torch.swapaxes(x, 1, 2)
        x = Augmentations.x_shear(x, p)
        x = torch.swapaxes(x, 1, 2)
        return x

    @staticmethod
    def rotate(x, p=1.0):
        if p > 0.5:
            x = torch.rot90(x, dims=[1, 2])
        return x

class LSTPData(Dataset):
    @property
    def batch_size(self):
        if 'data_params' in self.config:
            if 'batch_size' in self.config['data_params']:
                return self.config['data_params']['batch_size']
        return 64

    @property
    def width(self):
        return self.data.shape[3]

    @property
    def height(self):
        return self.data.shape[2]

    @property
    def data_fname(self):
        if 'data_params' in self.config:
            if 'data_fname' in self.config['data_params']:
                return self.config['data_params']['data_fname']
        return 'D1'

    @property
    def c_size(self):
        if 'data_params' in self.config:
            if 'c_size' in self.config['data_params']:
                return self.config['data_params']['c_size']
        return 500

    def __len__(self):
        return self.data.shape[0]

    def __init__(self, data, config, subset, data_min=None, data_max=None):
        self.config = config
        self.subset = subset
        self.data = data

        self.data_min = data_min if data_min is not None else self.data.min()
        self.data_max = data_max if data_max is not None else self.data.max()
        
        self.aug = Augmentations()
        if self.width == self.height:
            self.aug_pool = [self.aug.x_flip, self.aug.y_flip, self.aug.value_inversion, self.aug.x_cyclic_shift, self.aug.y_cyclic_shift, self.aug.x_shear, self.aug.y_shear, self.aug.rotate]
        else:
            self.aug_pool = [self.aug.x_flip, self.aug.y_flip, self.aug.value_inversion, self.aug.x_cyclic_shift, self.aug.y_cyclic_shift, self.aug.x_shear, self.aug.y_shear]

    def augment(self, x, transf):
        for i, aug in enumerate(self.aug_pool):
            if transf[i] > 0.5:
                x = aug(x, np.random.rand())
        return x

    def fix_augment(self, x, transf):
        for i, aug in enumerate(self.aug_pool):
            if transf[i] > 0.5:
                x = aug(x, (transf[i] - 0.5) * 2.0)
        return x

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx, :, :, :])
        x = (x - self.data_min) / (self.data_max - self.data_min)

        pre_transf = torch.randint(0, 2, (len(self.aug_pool),))
        x = self.augment(x, pre_transf)

        x_aug = x.clone()
        transf = torch.randint(0, 2, (len(self.aug_pool),))
        x_aug = self.augment(x_aug, transf)
        return torch.cat([x, x_aug], axis=0).float(), transf.float()

