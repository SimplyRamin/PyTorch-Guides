import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
n_epochs = 4
batch_size = 4
learning_rate = .01

# CIFAR dataset has PILImage of range [0, 1].
# we want to transform them to tensors and normalize them to range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./PyTorch Guides/data/CIFAR-10', train=True,
                                            download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./PyTorch Guides/data/CIFAR-10', train=False,
                                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# now we can look to one batch of data
examples = iter(train_loader)
images, labels = examples.next()
print(f'Samples shape: {images.shape}, labels shape: {labels.shape}')
print(f'Each Batch have: {images.shape[0]}, number of channels: {images.shape[1]}')
print(f'Samples image size is : {images.shape[2]} x {images.shape[3]}')

# monitoring samples:
def imshow(img):
    img = img / 2 + .5      # unnormalizing
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# show images
# imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        # again since we use cross entropy loss, we must not use softmax layer for last layer

    def forward(self, x):
        y_hat = self.conv1(x)
        y_hat = self.relu1(y_hat)
        y_hat = self.pool1(y_hat)
        y_hat = self.conv2(y_hat)
        y_hat = self.relu2(y_hat)
        y_hat = self.pool2(y_hat)
        y_hat = y_hat.view(-1, 16*5*5)  # flattening the output of last conv pool layer
        y_hat = self.fc1(y_hat)
        y_hat = self.relu3(y_hat)
        y_hat = self.fc2(y_hat)
        y_hat = self.relu4(y_hat)
        y_hat = self.fc3(y_hat)
        return y_hat

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(n_epochs):
    for i, (image, labels) in enumerate(train_loader):
        # sample shape: [4, 3, 32, 32]
        # input: 3 channels, output: 6 channels, filter size = 5x5
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch: {epoch+1} / {n_epochs}, step: {i+1} / {n_total_steps}, loss: {loss.item():.4f}')


n_correct = 0
n_samples = 0
n_class_correct = [0 for i in range(10)]
n_class_samples = [0 for i in range(10)]
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images).detach()

    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

    for i in range(batch_size):
        label = labels[i]
        pred = predicted[i]
        if (label == pred):
            n_class_correct[label] += 1
        n_class_samples[label] += 1

acc = 100 * n_correct / n_samples
print(f'accuracy: {acc}')

for i in range(10):
    acc = 100 * n_class_correct[i] / n_class_samples[i]
    print(f'accuracy of {classes[i]}: {acc} %')