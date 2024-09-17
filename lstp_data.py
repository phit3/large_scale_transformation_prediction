import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import transform

class Augmentations:
    @staticmethod
    def x_flip(x, p=1.0):
        if p > 0.5:
            x = torch.flip(x, dims=[2])
        return x

    @staticmethod
    def y_flip(x, p=1.0):
        if p > 0.5:
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
        if p > 0.5:
            x = x.max() - x + x.min()
        return x

    @staticmethod
    def x_shear(x, p=0.5):
        w = x.shape[2]
        x = torch.cat([x[:, :, :], x, x[:, :, :]], dim=2)
        x = x[0]
        
        p = (p - 0.5) * 2.0
        ps = np.sign(p) or 1.0

        shear = ps * 0.05 + p * 0.1
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
    def data_min(self):
        if 'data_params' in self.config:
            if 'data_min' in self.config['data_params']:
                return self.config['data_params']['data_min']
        return 0.0

    @property
    def data_max(self):
        if 'data_params' in self.config:
            if 'data_max' in self.config['data_params']:
                return self.config['data_params']['data_max']
        return 1.0

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

    #def __init__(self, config, subset):
    def __init__(self, data, config, subset):
        self.config = config
        self.subset = subset
        #raw_data = np.load(os.path.join('data', f'{self.data_fname}_{self.subset}.npy'))
        #self.data = np.array([raw_data[i: i + self.c_size].mean(0) for i in range(raw_data.shape[0] - self.c_size + 1)])
        self.data = data
        
        self.aug = Augmentations()
        if self.width == self.height:
            self.aug_pool = [self.aug.x_flip, self.aug.y_flip, self.aug.x_cyclic_shift, self.aug.y_cyclic_shift, self.aug.value_inversion, self.aug.x_shear, self.aug.y_shear, self.aug.rotate]
        else:
            self.aug_pool = [self.aug.x_flip, self.aug.y_flip, self.aug.x_cyclic_shift, self.aug.y_cyclic_shift, self.aug.value_inversion, self.aug.x_shear, self.aug.y_shear]

    def augment(self, x, transf):
        for i, aug in enumerate(self.aug_pool):
            #print(i, aug.__name__, transf[i].item())
            x = aug(x, transf[i])
        return x

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx, :, :, :])
        x = (x - self.data_min) / (self.data_max - self.data_min)
        # pre augment
        #if self.subset == 'train':
        pre_transf = torch.randint(0, 2, (len(self.aug_pool),))
        x = self.augment(x, pre_transf)

        x_aug = x.clone()
        transf = torch.randint(0, 2, (len(self.aug_pool),))
        x_aug = self.augment(x_aug, transf)
        return torch.cat([x, x_aug], axis=0).float(), transf.float()

