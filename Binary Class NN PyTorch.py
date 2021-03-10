import torch
import torch.nn as nn

class BinaryClassNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(BinaryClassNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        # We must have Sigmoid at the last layer
        # Because we are using BCELoss for Binary Class NN.
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.sigmoid1(out)
        return out
    
model = BinaryClassNN(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()

