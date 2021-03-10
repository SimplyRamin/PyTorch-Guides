import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        # in wine dataset first row is wine category.
        xy = np.loadtxt('./PyTorch/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]     # the shape of y is -> n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform

    
    def __getitem__(self, index):
        # to implement calling dataset with index eg. -> dataset[0]
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        # to implement calling len method for data set eg. -> len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, labels = sample
        inputs *= self.factor
        return inputs, labels

dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))