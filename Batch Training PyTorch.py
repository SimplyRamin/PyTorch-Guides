import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        # in wine dataset first row is wine category.
        xy = np.loadtxt('./PyTorch Guides/data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])     # the shape of y is -> n_samples, 1
        self.n_samples = xy.shape[0]

    
    def __getitem__(self, index):
        # to implement calling dataset with index eg. -> dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        # to implement calling len method for data set eg. -> len(dataset)
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# training loop (dummy training loop)
num_epoch = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)    # total samples / batch size, our batch size is 4
print(f'total samples : {total_samples}, number of iterations: {n_iterations}')

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass and backward pass and update parameters
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1} / {num_epoch}, step: {i+1} / {n_iterations}, inputs: {inputs.shape}')
            
