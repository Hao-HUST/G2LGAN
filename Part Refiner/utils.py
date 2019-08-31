#!/usr/bin/env python

__author__ = "Hao Wang"
__license__ = "MIT"


import numpy as np


#load batch data for training AE
def load_batch(data, data_64, pairs, BATCH_SIZE,label_num ):
    idx = np.random.randint(0, len(pairs), BATCH_SIZE)
    x = data[pairs[idx, 0]]
    y = data_64[pairs[idx, 1]]
    label = np.zeros((BATCH_SIZE, label_num))
    label[pairs[idx, 2]] = 1

    return x, label, y

def load_data(path):
    exm = np.load(path)
    return exm


def separate_parts(data, dim, part_num):

    separate = {}

    for ID in xrange(part_num):
    	separate[ID] = np.empty([0, dim, dim, dim, 2])


    for i in xrange(len(data)):
        tmp = np.zeros([part_num, dim, dim, dim, 2])
        print 'Separate', i
        for j in xrange(data.shape[1]):
            for k in xrange(data.shape[2]):
                for l in xrange(data.shape[3]):
                    index = np.argmax(data[i, j, k, l])
                    for n in xrange(len(separate)):
                        if (index - 1) == n:
                            tmp[n, j, k, l, 1] = 1
                        else:
                            tmp[n, j, k, l, 0] = 1
        for n in xrange(len(separate)):
            separate[n] = np.concatenate((separate[n], np.reshape(tmp[n], (np.append([1], tmp[n].shape)))), axis=0)

    return separate

