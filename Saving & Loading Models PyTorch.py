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
# ...

# Method 1 -> Lazy Method
# Saving
FILE = 'PyTorch Guides/model-method1-lazy.pth'
torch.save(model, FILE)
# Loading
model = torch.load(FILE)
model.eval()


model = Model(n_input_features=6)
# Method 2 -> Preferred Method
# Saving
FILE = 'PyTorch Guides/model-method2-recommended.pth'
torch.save(model.state_dict(), FILE)
# Loading
model = Model(n_input_features=6)
model.load_state_dict(torch.load(FILE))
model.eval()

for param in model.parameters():
    print(param)

