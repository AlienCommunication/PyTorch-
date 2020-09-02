#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy


# In[3]:


"""

This is only for checking the device configuration
IF CUDA Is present or not

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""

Hyper Parameters

"""
num_epochs = 5
batch_size = 4
learning_rate = 0.001



"""

Transformation of the Images into Tensor that too Normalised Tensor

Normalize  Args:

    mean (sequence): Sequence of means for each channel.
    std (sequence): Sequence of standard deviations for each channel.


"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# In[9]:


"""

Import the Dataset

"""
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    ]),
}


# In[10]:


data_dir = 'main_dir'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)


# In[11]:


"""

Let's iterate the dataloader It is used to shuffle the dataset"""
inputs, classes = next(iter(dataloaders['train']))


# In[12]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        """
        
        Every epoch will have training and validation  result
        
        """
        for phase in ['train', 'val']:
            if phase == 'train':
                
                """
                
                This will set our model to training mode
                
                """
                model.train()
            else:
                
                """
                
                And This will set our model to evaluation mode
                
                """
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            """
            
            Let's iterate over the data
            
            """
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                """
                
                Forward 
                
                """
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    """
                    
                    Backward with optimisation
                    
                    """
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()


# ## Transfer Learning 

# In[21]:


"""
Load weights of the Model

"""

model = models.resnet18(pretrained= True)

"""

Here I will modify the last fully connected layer

But before that Let's get the number of feature from the Last layer

"""

num_features = model.fc.in_features


"""

Now We will create a new layer and assign it to the Last layer 

nn.linear accepts args :

1. Number of output features from previous layer and desire output we want from our last layer


"""

model.fc = nn.Linear(num_features, 2)


model = model.to(device)

"""
Defining the loss function and optimiser

"""


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

"""

This lr.scheduler.StepLR : Decays the learning rate of each parameter group by gamma every
step_size epochs. Notice that such decay can happen simultaneously with
other changes to the learning rate from outside this scheduler. When
last_epoch=-1, sets initial lr as lr.

Args:
    optimizer (Optimizer): Wrapped optimizer.
    step_size (int): Period of learning rate decay.
    gamma (float): Multiplicative factor of learning rate decay.
        Default: 0.1.
    last_epoch (int): The index of last epoch. Default: -1.


"""


step_learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_learning_rate_scheduler, num_epochs=25)




model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False


# ## Fine Tuning 

# In[ ]:




