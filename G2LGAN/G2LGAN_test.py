# !/usr/bin/env python
import os
import sys

import numpy as np
import tensorflow as tf

from utils import *
import scipy.io as sio
import argparse

'''
Global Parameters
'''


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))


parser = argparse.ArgumentParser()
parser.add_argument('--input_obj', default='chair', help='The input data path for seperating [default: None]')
parser.add_argument('--num_semantic_parts', type=int, default=4, help='The amount of semantic parts on shape')
parser.add_argument('--GPU_ID', type=int, default=0, help='Choose which GPU device you want to use')
parser.add_argument('--BTACH_SIZE', type=int, default=64, help='The btach sie fedd into the model')
parser.add_argument('--voxel_resolution', type=int, default=32, help='The voxel resolution represent the 3D shape')
parser.add_argument('--NUM_BATCH', type=int, default=16, help='How many batches you want to sample from pretrained models')
parser.add_argument('--model_path', default='/model/20010.cptk', help='model checkpoint file path [default: /model/20010.cptk]')

FLAGS = parser.parse_args()

#choose your GPU device
device_ID = FLAGS.GPU_ID
obj = FLAGS.input_obj
label_num = FLAGS.num_semantic_parts
batch_size = FLAGS.BTACH_SIZE
cube_len = FLAGS.voxel_resolution
obj = FLAGS.input_obj
NUM_EPOCH = FLAGS.NUM_BATCH


#The saving path for G2LGAN model
PRETRAINED_MODEL_PATH = os.path.join(BASE_DIR, 'result/' +  obj + FLAGS.model_path)

#Path for saving the genreated shapes from pre-trained model
SAMPLES_PATH = os.path.join(BASE_DIR, 'result/' +  obj + '/sample/testing')

if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)

#The hyper-parameter for pre-trained model
z_size = 200

weights = {}



def generator(z, batch_size=batch_size, phase_train=True, reuse=False):
	
    strides    = [1,2,2,2,1]
    
    with tf.variable_scope("gen", reuse = reuse):

        z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
        g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,256), strides=[1,1,1,1,1], padding="VALID")
        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
        g_1 = tf.nn.relu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,128), strides=strides, padding="SAME")
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = tf.nn.relu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,64), strides=strides, padding="SAME")
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = tf.nn.relu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,label_num+1), strides=strides, padding="SAME")
        g_4 = tf.nn.softmax(g_4)

    return g_4


def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable("gen/wg1", shape=[4, 4, 4, 256, z_size], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("gen/wg2", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("gen/wg3", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("gen/wg4", shape=[4, 4, 4, label_num+1, 64], initializer=xavier_init)


    return weights



def testGAN(num_test_epoch, checkpoint):

    weights = initialiseWeights()

    z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32) 
    net_g_test= generator(z_vector, phase_train=False, reuse=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    sess = tf.Session()
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        #---------------restore the pretained model----------------
        print "Loading previous model..."
        saver.restore(sess, checkpoint)
        print "Done."

        for epoch in xrange(int(num_test_epoch)):

            z_sample = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)
            g_models = sess.run(net_g_test, feed_dict={z_vector: z_sample})

            g_models.dump(SAMPLES_PATH + '/' + 'batch_'+str(epoch)+ '.npy')

if __name__ == '__main__':

    testGAN(NUM_EPOCH, PRETRAINED_MODEL_PATH)

