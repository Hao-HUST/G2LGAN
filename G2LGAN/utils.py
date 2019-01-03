#!/usr/bin/env python

__author__ = "Hao Wang"
__license__ = "MIT"

import tensorflow as tf
from tensorflow.python.ops import  math_ops

import scipy.io as sio
import numpy as np
import math
import os
from scipy import ndimage
from scipy.io import loadmat


def load_mat(matFile, cube_len):
    data = sio.loadmat(matFile)
    volume_size = (cube_len, cube_len, cube_len)
    array = np.ndarray(volume_size, np.int32)
    array = data['instance']
    return array


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)



def sig_smooth_purity_sqr(images, batch_size, cube_len, label_num, is_loss=False):
    if is_loss:
        smooth_limit = 2.89
        purity_limit = 1.69
        images = my_sigmoid(images)
    else:
        smooth_limit = 3
        purity_limit = 2
        images_max = tf.argmax(images, axis=4)
        images = tf.one_hot(images_max, depth=label_num + 1)

    pixel_dif1 = math_ops.abs(images[:, 1:, :, :, :] - images[:, :-1, :, :, :])
    pixel_dif2 = math_ops.abs(images[:, :, 1:, :, :] - images[:, :, :-1, :, :])
    pixel_dif3 = math_ops.abs(images[:, :, :, 1:, :] - images[:, :, :, :-1, :])

    sum_axis = [1, 2, 3]

    shape = [batch_size, 1, cube_len, cube_len, label_num+1]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_1 = tf.concat((pixel_dif1, padding), axis=1) + tf.concat((padding, pixel_dif1), axis=1)
    shape = [batch_size, cube_len, 1, cube_len, label_num+1]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_2 = tf.concat((pixel_dif2, padding), axis=2) + tf.concat((padding, pixel_dif2), axis=2)
    shape = [batch_size, cube_len, cube_len, 1, label_num+1]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_3 = tf.concat((pixel_dif3, padding), axis=3) + tf.concat((padding, pixel_dif3), axis=3)

    total_sqr_tv = math_ops.reduce_sum(tv_ax_1 + tv_ax_2 + tv_ax_3, axis=4)

    chair_voxels = tf.reshape(tf.reduce_sum(images[:, :, :, :, 1:], axis=4),
                              (batch_size, cube_len, cube_len, cube_len, 1))
    backgroung_voxels = tf.reshape(images[:, :, :, :, 0], (batch_size, cube_len, cube_len, cube_len, 1))
    images = tf.concat((backgroung_voxels, chair_voxels), 4)

    pixel_dif1 = math_ops.abs(images[:, 1:, :, :, :] - images[:, :-1, :, :, :])
    pixel_dif2 = math_ops.abs(images[:, :, 1:, :, :] - images[:, :, :-1, :, :])
    pixel_dif3 = math_ops.abs(images[:, :, :, 1:, :] - images[:, :, :, :-1, :])

    shape = [batch_size, 1, cube_len, cube_len, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_1 = tf.concat((pixel_dif1, padding), axis=1) + tf.concat((padding, pixel_dif1), axis=1)
    shape = [batch_size, cube_len, 1, cube_len, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_2 = tf.concat((pixel_dif2, padding), axis=2) + tf.concat((padding, pixel_dif2), axis=2)
    shape = [batch_size, cube_len, cube_len, 1, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_3 = tf.concat((pixel_dif3, padding), axis=3) + tf.concat((padding, pixel_dif3), axis=3)
    smooth = math_ops.reduce_sum(tv_ax_1 + tv_ax_2 + tv_ax_3, axis=4)

    purity_matrix = tf.square(tf.nn.relu(math_ops.div(total_sqr_tv - smooth, 2) - tf.ones(total_sqr_tv.shape)*purity_limit))
    purity_sqr = math_ops.reduce_sum(purity_matrix, axis=sum_axis)
    smooth_matrix = tf.square(tf.nn.relu(math_ops.div(smooth, 2) - tf.ones(smooth.shape)*smooth_limit))
    smooth_sqr = math_ops.reduce_sum(smooth_matrix, axis=sum_axis)

    return smooth_sqr, purity_sqr



def sig_smooth_sqr(images, batch_size, cube_len, label_num, is_loss=False):
    sum_axis = [1, 2, 3]

    if is_loss:
        smooth_limit = 2.89
        images = my_sigmoid(images)
    else:
        smooth_limit = 3
        images_max = tf.argmax(images, axis=4)
        images = tf.one_hot(images_max, depth=label_num)

    chair_voxels = tf.reshape(tf.reduce_sum(images[:, :, :, :, 1:], axis=4),
                              (batch_size, cube_len, cube_len, cube_len, 1))
    backgroung_voxels = tf.reshape(images[:, :, :, :, 0], (batch_size, cube_len, cube_len, cube_len, 1))
    images = tf.concat((backgroung_voxels, chair_voxels), 4)

    pixel_dif1 = math_ops.abs(images[:, 1:, :, :, :] - images[:, :-1, :, :, :])
    pixel_dif2 = math_ops.abs(images[:, :, 1:, :, :] - images[:, :, :-1, :, :])
    pixel_dif3 = math_ops.abs(images[:, :, :, 1:, :] - images[:, :, :, :-1, :])

    shape = [batch_size, 1, cube_len, cube_len, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_1 = tf.concat((pixel_dif1, padding), axis=1) + tf.concat((padding, pixel_dif1), axis=1)
    shape = [batch_size, cube_len, 1, cube_len, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_2 = tf.concat((pixel_dif2, padding), axis=2) + tf.concat((padding, pixel_dif2), axis=2)
    shape = [batch_size, cube_len, cube_len, 1, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_3 = tf.concat((pixel_dif3, padding), axis=3) + tf.concat((padding, pixel_dif3), axis=3)
    smooth = math_ops.reduce_sum(tv_ax_1 + tv_ax_2 + tv_ax_3, axis=4)

    smooth_matrix = tf.square(tf.nn.relu(math_ops.div(smooth, 2) - tf.ones(smooth.shape)*smooth_limit))
    smooth_sqr = math_ops.reduce_sum(smooth_matrix, axis=sum_axis)

    return smooth_sqr


def my_sigmoid(x):
    return tf.div(1.0, 1.0 + tf.exp(-100 * (x - 0.5)))


def smooth_purity_sqr(images, batch_size, cube_len, is_loss=False):
    if is_loss:
        smooth_limit = 2.89
        purity_limit = 1.69
        images = my_sigmoid(images)
    else:
        smooth_limit = 3
        purity_limit = 2
        images_max = tf.argmax(images, axis=4)
        images = tf.one_hot(images_max, depth=5)

    label_num = 4
    pixel_dif1 = math_ops.abs(images[:, 1:, :, :, :] - images[:, :-1, :, :, :])
    pixel_dif2 = math_ops.abs(images[:, :, 1:, :, :] - images[:, :, :-1, :, :])
    pixel_dif3 = math_ops.abs(images[:, :, :, 1:, :] - images[:, :, :, :-1, :])

    sum_axis = [1, 2, 3]

    shape = [batch_size, 1, cube_len, cube_len, label_num+1]
    padding = tf.zeros(shape, tf.float32)

    tv_ax_1 = tf.concat((pixel_dif1, padding), axis=1) + tf.concat((padding, pixel_dif1), axis=1)
    shape = [batch_size, cube_len, 1, cube_len, label_num+1]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_2 = tf.concat((pixel_dif2, padding), axis=2) + tf.concat((padding, pixel_dif2), axis=2)
    shape = [batch_size, cube_len, cube_len, 1, label_num+1]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_3 = tf.concat((pixel_dif3, padding), axis=3) + tf.concat((padding, pixel_dif3), axis=3)

    total_sqr_tv = math_ops.reduce_sum(tv_ax_1 + tv_ax_2 + tv_ax_3, axis=4)

    chair_voxels = tf.reshape(tf.reduce_sum(images[:, :, :, :, 1:], axis=4),
                              (batch_size, cube_len, cube_len, cube_len, 1))
    backgroung_voxels = tf.reshape(images[:, :, :, :, 0], (batch_size, cube_len, cube_len, cube_len, 1))
    images = tf.concat((backgroung_voxels, chair_voxels), 4)

    pixel_dif1 = math_ops.abs(images[:, 1:, :, :, :] - images[:, :-1, :, :, :])
    pixel_dif2 = math_ops.abs(images[:, :, 1:, :, :] - images[:, :, :-1, :, :])
    pixel_dif3 = math_ops.abs(images[:, :, :, 1:, :] - images[:, :, :, :-1, :])

    shape = [batch_size, 1, cube_len, cube_len, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_1 = tf.concat((pixel_dif1, padding), axis=1) + tf.concat((padding, pixel_dif1), axis=1)
    shape = [batch_size, cube_len, 1, cube_len, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_2 = tf.concat((pixel_dif2, padding), axis=2) + tf.concat((padding, pixel_dif2), axis=2)
    shape = [batch_size, cube_len, cube_len, 1, 2]
    padding = tf.zeros(shape, tf.float32)
    tv_ax_3 = tf.concat((pixel_dif3, padding), axis=3) + tf.concat((padding, pixel_dif3), axis=3)
    smooth = math_ops.reduce_sum(tv_ax_1 + tv_ax_2 + tv_ax_3, axis=4)

    purity_matrix = tf.square(tf.nn.relu(math_ops.div(total_sqr_tv - smooth, 2) - tf.ones(total_sqr_tv.shape)*purity_limit))
    purity = math_ops.reduce_sum(purity_matrix, axis=sum_axis)
    smooth_matrix = tf.square(tf.nn.relu(math_ops.div(smooth, 2) - tf.ones(smooth.shape)*smooth_limit))
    smooth_sqr = math_ops.reduce_sum(smooth_matrix, axis=sum_axis)

    return smooth_sqr, purity




def do_masking(input_image, input_mask):
    
    mask_operation = tf.multiply(input_image,input_mask)

    return mask_operation


#define the attention mask function
def attention_mask(batch_image, batch_image_mask):

    masked_image_list = []
    
    input_image_list = tf.unstack(batch_image, axis = 0)

    input_mask_list = tf.unstack(batch_image_mask, axis = 0)

    for (input_image, input_mask) in zip(input_image_list, input_mask_list):
    	masked_image = tf.cond(tf.count_nonzero(input_mask) > 0,  lambda: do_masking(input_image,input_mask), lambda: input_image)
        masked_image_list.append(masked_image)

    masked_image_batch = tf.stack(masked_image_list)

    return masked_image_batch

def loading_training_data(training_data_path, cube_len, label_num):

    dataset_files = []

    models = os.listdir(training_data_path)
    count_loaded_data = 0

    for model in models:
        dataset_files.append(training_data_path + '/' +  model)
        count_loaded_data=count_loaded_data+1

    print 'loading training data:', count_loaded_data

    training_data_volume = np.zeros((count_loaded_data,cube_len,cube_len,cube_len,label_num+1), dtype=np.float32)

    for index in range(count_loaded_data):
        data_file = dataset_files[index]
        mp = load_mat(data_file,cube_len)
        for x in range(cube_len):
            for y in range(cube_len):
                for z in range(cube_len):
                    training_data_volume[index-1, x, y, z, mp[x,y,z]] = 1
    dataset_files = np.array(dataset_files)   


    return training_data_volume, dataset_files.shape[0]




def load_volumSDF(size):
    dataset_files = []
    # dataPath = 	'./training_data/'+obj
    dataPath = './training_data/pure_' + str(size)
    models = os.listdir(dataPath)
    volumSDF = np.zeros((len(models), size, size, size, label_num + 1), dtype=np.float32)
    for model in models:
        dataset_files.append(dataPath + '/' + model)
        mp = load_mat(dataset_files[-1], size)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    volumSDF[len(dataset_files) - 1, x, y, z, mp[x, y, z]] = 1
                    # print 'loading ', len(dataset_files), 'in ', len(models)
    dataset_files = np.array(dataset_files)

    return volumSDF, dataset_files.shape[0]









