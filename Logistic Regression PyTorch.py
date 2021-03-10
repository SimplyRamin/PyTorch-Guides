import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) preparing data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 69)

# scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshaping the tensors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# f = wx + b , activation function -> sigmoid
class LogisticRegression(nn.Module):
    
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 1)
        self.sigmoid1 = nn.Sigmoid()
    
    def forward(self, x):
        y_hat = self.linear1(x)
        y_hat = self.sigmoid1(y_hat)
        return y_hat

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = .01
critirion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop
n_epochs = 100
for epoch in range(n_epochs):
    # forward pass and loss
    y_hat = model(X_train)
    loss = critirion(y_hat, y_train)

    # backward pass
    loss.backward()

    # update parameters
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# to detach this part from computational graph we have to implement them this way or they
# will be a part of computation graph and we will get errors
# we can also implement this fact with detach() like this:
    # y_hat = model(X_test).detach()
    # y_hat_classes = y_hat.round()
    # accuracy = y_hat_classes.eq(y_test).sum() / float(y_test.shape[0])
    # print(f'accuracy = {accuracy:.4f}')

with torch.no_grad():
    y_hat = model(X_test)
    y_hat_classes = y_hat.round()
    accuracy = y_hat_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')

