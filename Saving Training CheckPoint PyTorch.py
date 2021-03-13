import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        y_hat = self.linear1(x)
        y_hat = self.sigmoid1(y_hat)
        return y_hat

model = Model(n_input_features=6)
# Here we train the model 
learning_rate = .01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f'Model State Dictionary: {model.state_dict()}')
print(f'Optimizer State Dictionary: {optimizer.state_dict()}')

checkpoint = {
    'epoch': 90,
    'model_state': model.state_dict(),
    'optim_state': optimizer.state_dict()
}

# Saving
torch.save(checkpoint, 'PyTorch Guides/checkpoint_state.pth')
