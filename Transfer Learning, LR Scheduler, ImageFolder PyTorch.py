import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([.485, .456, .406])
std = np.array([.229, .224, .225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# importing data
data_dir = './PyTorch Guides/data/hymenoptera_data/'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in sets}

dataset_size = {x: len(image_datasets[x]) for x in sets}
class_names = image_datasets['train'].classes
print(class_names)

def train_model(model, criterion, optimizer, scheduler, n_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1} / {n_epochs+1}')
        print('-' * 10)

        # remember each epoch have training and test phase
        for phase in sets:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            # Iterating over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forwarding
                # we track history just when we are at training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only when we are at training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in : {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# # you can recall this as Fine Tuning
# # loading pre trained model
# model = models.resnet18(pretrained=True)

# # now we want to change last FC layer
# n_features = model.fc.in_features

# # creating new layer and assign it to the last layer
# model.fc = nn.Linear(n_features, 2)     # 2 since we want to classify 2 classes.
# model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=.001)

# # scheduler
# # we want every 7 epochs, learning rate multiply by 0.1
# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)

# model = train_model(model, criterion, optimizer, step_lr_scheduler, n_epochs=20)



# other thing we can do is to freeze all the layers and only train the last layer
# for that we can implement this way:
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_gra = False

# now we want to change last FC layer
n_features = model.fc.in_features

# creating new layer and assign it to the last layer
model.fc = nn.Linear(n_features, 2)     # 2 since we want to classify 2 classes.
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.001)

# scheduler
# we want every 7 epochs, learning rate multiply by 0.1
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, n_epochs=20)