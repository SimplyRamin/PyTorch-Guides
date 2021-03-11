import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784    # 28 x 28
hidden_size = 100
n_classes = 10      # from 0 to 9 digit
n_epochs = 2
batch_size = 100
learning_rate = .01

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./PyTorch Guides/data', train=True,
                                           transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./PyTorch Guides/data', train=False,
                                           transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# now we can look to one batch of data
examples = iter(train_loader)
samples, labels = examples.next()
print(f'Samples shape: {samples.shape}, labels shape: {labels.shape}')
print(f'Each Batch have: {samples.shape[0]}, number of channels: {samples.shape[1]}')
print(f'Samples image size is : {samples.shape[2]} x {samples.shape[3]}')

# monitoring samples:
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_classes)
        # we dont want to use softmax at last layer
        # because this is multi-class classification and we will use Cross entropy loss.
    
    def forward(self, x):
        y_hat = self.linear1(x)
        y_hat = self.relu1(y_hat)
        y_hat = self.linear2(y_hat)
        return y_hat

# have in mind that if you are using gpu, you should move your model to gpu
model = NeuralNet(input_size, hidden_size, n_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # we must reshape images first
        # as we saw the shape of samples in each batch: 100 x 1 x 28 x 28
        # but input shape of our NN is: 100 x 784
        images = images.reshape(-1, 28*28).to(device)   # Pushing to gpu
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1} / {n_epochs}, step: {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

# test
n_correct = 0
n_samples = 0
for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images).detach()

    _, predictions = torch.max(outputs, 1)
    # torch.max returns: value, index
    # we are not interested in value, we want just index
    n_samples += labels.shape[0]    # number of samples in current batch
    n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'Accuracy = {acc}')


# i have used detach() for disabling the phase to be part of computation graph
# you can also use torch.no_grad() like this:
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')