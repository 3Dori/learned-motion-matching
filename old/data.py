import numpy as np
import torch
from torch.utils.data import Dataset

import mocap.datasets.cmu as CMU


class CMUDataset(Dataset):
    def __init__(self, subjects, transform=None):
        ds = CMU.CMU(subjects=subjects)
        self.data = torch.cat([torch.tensor(seq[0]) for seq in ds])
        self.input_size = self.data.shape[1] * self.data.shape[2]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class UE4DataSet(Dataset):
    def __init__(self, x_filename, y_filename, transform=None):
        self.x_data = torch.tensor(np.load(x_filename).astype(np.float32))
        self.y_data = torch.tensor(np.load(y_filename).astype(np.float32))
        if self.x_data.shape[0] != self.y_data.shape[0]:
            raise ValueError('The length of x data and y data must be equal')
        self.x_size = self.x_data.shape[1]
        self.y_size = self.y_data.shape[1]
        self.transform = transform

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        x_sample = self.x_data[idx]
        y_sample = self.y_data[idx]
        if self.transform:
            x_sample = self.transform(x_sample)
            y_sample = self.transform(y_sample)

        return (x_sample, y_sample), idx
