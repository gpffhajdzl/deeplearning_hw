'''
HW2 problem
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg



# you can define/use whatever functions to implememt

########################################
# cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    # print(W.shape)
    # print(b.shape)

    x=np.reshape(x.T, (-1, n))
    # print(x.shape)

    # this will give you a score matrix s of size (num_class)-by-(n)
    # the i-th column vector of s will be
    # the score vector of size (num_class)-by-1, for the i-th input data point
    # performing s=Wx+b
    s=W@x+b
    # print(s.shape)



    for j in range(n):
      total =0
      for i in range(4):
        # print(s[i][0])
        s[i][j] = np.exp(s[i][j])
        # print(s[i][0])
        total += s[i][j]
        # print(total)
      for i in range(4):
        s[i][j] = (-1) * np.log((s[i][j] / total))
        # print(s[i][j])


    # print(s.shape)
    # print(y.shape)

    all_total=0
    for i in range(n):
     all_total += s[y[i],i]

    # print(arr)

    # cross_entropy_loss = np.mean(arr)
    cross_entropy_loss = all_total/n
    # print(cross_entropy_loss)
    return cross_entropy_loss



# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]

    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    # W has shape (num_class, feat_dim), b has shape (num_class,)

    # score
    s = x@W.T + b
    # score has shape (n_test, num_class)

    # get argmax over class dim
    res = np.argmax(s, axis = 1)

    # get accuracy
    accuracy = (res == y).astype('uint8').sum()/n_test

    return accuracy


# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# start training softmax classifier
print('training softmax classifier...')
w0 = np.random.normal(0, 1, (2 * num_class + num_class))
result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

print('testing softmax classifier...')

Wb = result.x
print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')