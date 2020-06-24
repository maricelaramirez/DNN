#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:55:17 2020

@author: ryanjeong
"""

# IMPORTS
import gzip
import numpy as np

# LOAD DATA
f = gzip.open('train-images-idx3-ubyte.gz','r')
f2 = gzip.open('train-labels-idx1-ubyte.gz','r')
f3 = gzip.open('t10k-images-idx3-ubyte.gz','r')
f4 = gzip.open('t10k-labels-idx1-ubyte.gz','r')

# PREPROCESSING
image_size = 28
num_train = 60000
num_test = 10000

# read in image data for train/test sets
f.read(16)
buf = f.read(image_size * image_size * num_train)
train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train_data = train_data.reshape(num_train, image_size * image_size)

f3.read(16)
buf = f3.read(image_size * image_size * num_test)
test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(num_test, image_size * image_size).T

# read in labels for train/test sets
f2.read(8)
buf = f2.read(num_train)
train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
train_labels = train_labels.reshape(num_train, 1)

f4.read(8)
buf = f4.read(num_test)
test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
test_labels = test_labels.reshape(num_test, 1)

# TRAINING THE MODEL (computing SVD, for each set of examples per label, 0-9)
tmp_dict = {}
for i in range(10):
    # gather all data points corresponding to label i
    boolArr = (train_labels == i)
    idx = np.where(boolArr)[0]
    idx = idx.reshape(len(idx), 1)
    temp = np.take(train_data, idx, 0).squeeze().T
    
    # train SVD model for this label by selecting num_pcs principal components
    u, s, vh = np.linalg.svd(temp, full_matrices = False)
    tmp_dict[i] = u

# Uses top num_pcs principal component subspaces for each class, and
# classifies new images based on which subspace the orthogonal projection of
# the unrolled vector is closest to in Euclidean distance.
# Returns model accuracy.
def svd_model(num_pcs):
    model_dict = {}

    # use num_pcs dimensional subspaces
    for i in range(10):
        model_dict[i] = tmp_dict[i][:, :num_pcs]

    # testing the model
    vec = np.zeros(test_labels.shape)
    for i in range(num_test):
        tmp_ix = -1
        min_proj = float("inf")
        for j in range(10): # consider all possible labels, 0-9
            pt = test_data[:,i].reshape((784,1))
            curr_proj = np.linalg.norm(pt - model_dict[j] @ (model_dict[j].T @ pt))
            if curr_proj < min_proj:
                min_proj = curr_proj
                tmp_ix = j
        if tmp_ix == test_labels[i,0]:
            vec[i,0] = 1
    
    # model accuracy
    return vec.mean()
    