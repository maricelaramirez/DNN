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
def layer_traj(lst, num_neurons):
    [train, trainloader, test, testloader] = load()
    for i in range(len(lst)):
        net = construct_nn(num_neurons, lst[i])
        [criterion, opt] = optimizer(net)

        # change number of epochs as a hyperparameter
        [iters, losses, test_err] = train_net(trainloader, testloader, net, criterion, opt, 1)

        print('Accuracy of the network on the 10000 test images: %f' % computeError(testloader, net))
        test_plot(iters, test_err)

    plt.legend(lst)
    plt.show()

if __name__ == "__main__":
    lst = [1, 2, 3]
    layer_traj(lst, 500)