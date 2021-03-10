import torch
import torch.nn as nn

class MultiClassNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_classes):
        super(MultiClassNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        # We should not use Softmax for last layer!
        # Because of PyTorch Cross Entropy method.
        return out

model = MultiClassNN(input_size=28*28, hidden_size=5, n_classes=3)
criterion = nn.CrossEntropyLoss()   #This will apply Softmax