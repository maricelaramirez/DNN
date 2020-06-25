#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:42:15 2020

@author: ryanjeong
"""

# Training a CNN on MNIST, using PyTorch.

# IMPORTS
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# LOAD DATA
def load():
    # Apply transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # MNIST
    train = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=4,
                                              shuffle=True, num_workers=2)

    test = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=4,
                                             shuffle=False, num_workers=2)
    return [train, trainloader, test, testloader]


# CONSTRUCTING NEURAL NETS
# Change hyperparameters to test different architectures.
# Constructing standard feedforward neural network - assumed hidden layers of equal width, for now
# num_neurons neurons over i hidden layers (does not include the final output layer)
def construct_nn(num_neurons, i):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            width = int(num_neurons/i)
            setattr(self, 'fc1', nn.Linear(784, width))
            setattr(self, 'fc%s' % (i+1), nn.Linear(width, 10))
            for idx in range(2, i+1):
                setattr(self, 'fc%s' % idx, nn.Linear(width, width))
            self.softmax = nn.Softmax()

        def forward(self, x):
            x = x.view(-1, 784)
            for idx in range(1, i+2):
                x = F.relu(getattr(self, 'fc%s' % idx)(x))
            x = self.softmax(x)
            return x

    return Net()

# Constructing CNN.
def construct_cnn():
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

# Computes error on test set
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

# adds to the running loss for a particular iteration of SGD
def single_iter(data, criterion, opt, net):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    opt.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    opt.step()

    return loss.item()

# TRAINS NEURAL NET
def train_net(trainloader, testloader, net, criterion, opt, num_epochs):
    iters = []
    losses = []
    test_err = []

    # give error at initialization
    iters.append(0)
    losses.append(single_iter(list(enumerate(trainloader, 0))[0][1], criterion, opt, net))
    test_err.append(round(computeError(testloader, net), 2))

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # print statistics
            running_loss += single_iter(data, criterion, opt, net)
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                losses.append(round(running_loss / 2000, 2))
                iters.append(epoch * 14999 + i + 1)
                test_err.append(round(computeError(testloader, net), 2))
                running_loss = 0.0

    return [iters, losses, test_err]

# Shows relevant plots.
# Training error vs. iterations
def train_plot(iters, losses):
    plt.title('training error over time')
    plt.xlabel('iterations')
    plt.ylabel('training error')
    plt.plot(iters, losses)

# Test error vs. iterations
def test_plot(iters, test_err):
    plt.title('test accuracy over time')
    plt.xlabel('iterations')
    plt.ylabel('test accuracy')
    plt.plot(iters, test_err)

# FULL DL PIPELINE
# Trains a specified neural network for a canonical benchmark problem.
if __name__ == "__main__":
    [train, trainloader, test, testloader] = load()
    net = construct_cnn()
    [criterion, opt] = optimizer(net)

    [iters, losses, test_err] = train_net(trainloader, testloader, net, criterion, opt, 3) # change number of epochs

    print('Accuracy of the network on the 10000 test images: %f' %
          computeError(testloader, net))

    train_plot(iters, losses)
    plt.show()
    test_plot(iters, test_err)
    plt.show()
