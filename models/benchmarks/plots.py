#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 04:52:25 2020

@author: ryanjeong
"""

# Graphing functions for visualizing model performance.

# import functions from cnn_mnist
from models.benchmarks.cnn_mnist import *


# Graphs learning trajectories over different numbers of layers,
# while maintaining the number of neurons as a constant.
# lst contains set of number of layers to be observed.
def nn_layer_traj(lst, num_neurons):
    [_, trainloader, _, testloader] = load()
    for i in range(len(lst)):
        net = construct_nn(num_neurons, lst[i])
        [criterion, opt] = optimizer(net)

        # change number of epochs as a hyperparameter
        [iters, _, test_err] = train_net(trainloader, testloader, net, criterion, opt, 10)

        print('Accuracy of the network on the 10000 test images: %f' % computeError(testloader, net))
        test_plot(iters, test_err)

    plt.legend(lst)
    plt.title("training over different num_layers, total of %s neurons" % num_neurons)
    plt.show()

# Graphs learning trajectories over different numbers of convolutional layers,
# while maintaining the number of neurons as a constant.
# num_conv_layers contains list of number of convolutional layers to be observed.
def cnn_layer_traj(num_conv_layers, kernel_widths, num_filters):
    [_, trainloader, _, testloader] = load()
    for i in range(len(num_conv_layers)):
        net = construct_cnn(num_conv_layers[i], kernel_widths, num_filters)
        [criterion, opt] = optimizer(net)

        # change number of epochs as a hyperparameter
        [iters, _, test_err] = train_net(trainloader, testloader, net, criterion, opt, 2)

        print('Accuracy of the network on the 10000 test images: %f' % computeError(testloader, net))
        test_plot(iters, test_err)

    plt.legend(num_conv_layers)
    plt.title("training over different num_conv_layers")
    plt.show()

if __name__ == "__main__":
    lst = [1, 2, 3]
    #nn_layer_traj(lst, 200)

    cnn_layer_traj(lst, [5, 3, 2], [6, 16, 16])