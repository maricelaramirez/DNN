#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:42:15 2020

@author: ryanjeong
"""

# Training a CNN on MNIST, using PyTorch.
# Heavily drawn from PyTorch's tutorial code; uses the same framework, with
# some adjusted hyperparameters.

# IMPORTS
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# LOAD DATA
def load():
    # Apply transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    # MNIST
    train = torchvision.datasets.MNIST(root = './data', train = True, 
                                            download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=4,
                                            shuffle=True, num_workers=2)

    test = torchvision.datasets.MNIST(root = './data', train = False, 
                                            download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=4,
                                            shuffle=False, num_workers=2)
    return [train, trainloader, test, testloader]

# CONSTRUCT NEURAL NET
# Change hyperparameters to test different architectures.
def construct_nn():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return Net()

# DEFINE OPTIMIZER
def optimizer(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return [criterion, optimizer]

# Computes error on the test set.
def computeError(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

# TRAINS NEURAL NET
def train_net(train, net, criterion, num_epochs):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

# FULL DL PIPELINE
# Trains a specified neural network for a canonical benchmark problem.
def main():
    [train, trainloader, test, testloader] = load()
    
    print('Accuracy of the network on the 10000 test images: %f' % computeError())
