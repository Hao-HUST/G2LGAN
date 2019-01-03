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
parser.add_argument('--model_path', default=None, help='model checkpoint file path [default: None]')


FLAGS = parser.parse_args()

#choose your GPU device
device_ID = FLAGS.GPU_ID
obj = FLAGS.input_obj
label_num = FLAGS.num_semantic_parts
batch_size = FLAGS.BTACH_SIZE
cube_len = FLAGS.voxel_resolution
obj = FLAGS.input_obj
check_point = FLAGS.model_path


#The saving path for G2LGAN model
MODEL_SAVE = os.path.join(BASE_DIR, 'result/' +  obj + '/model')

#Path for saving the genreated shapes during training
SAMPLES_DURING_TRAINING = os.path.join(BASE_DIR, 'result/' +  obj + '/sample/training')

#The saving path for log files
LOG_PATH = os.path.join(BASE_DIR, 'result/' +  obj +'/log')

if not os.path.exists(MODEL_SAVE):
    os.makedirs(MODEL_SAVE)
if not os.path.exists(SAMPLES_DURING_TRAINING):
    os.makedirs(SAMPLES_DURING_TRAINING)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)



#The hyper-parameter for trainig model
n_epochs = 20100
BASE_LEARNING_RATE_GENERATOR = 0.0001
BASE_LEARNING_RATE_DISCRIMINATOR = 0.0001
DECAY_RATE = 0.9
beta = 0.5
z_size = 200
leak_value = 0.2

CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10  # Gradient penalty lambda hyperparameter
EPSOLON    = 0.001

GPU_OCCUPY_RATE = 0.9


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))



#Saving Path

#Path for saving the genreated shapes during training
train_sample_directory = './result/' + obj + '/sample'
#Path for saving the trained model
model_directory = './result/' + obj + '/stored_model'
#Path for saveing the saving the loss cureves during training
summary_directory = './summary/' + obj


#Parameters for networks
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


     	#one hot operation 
    	input_max = tf.reduce_max(g_4, axis=4)
    	input_max = tf.expand_dims(input_max,-1)

    	g_4_one_hot = tf.div(tf.nn.relu(tf.subtract(g_4, input_max * 0.99)), input_max*0.01)

    	g_4_backgruond_one_hot, g_4_part0_one_hot, g_4_part1_one_hot, g_4_part2_one_hot, g_4_part3_one_hot = tf.split(g_4_one_hot, [1,1,1,1,1], 4) 
 
    	maksed_g_4_part0 = attention_mask(g_4, g_4_part0_one_hot)
    	maksed_g_4_part1 = attention_mask(g_4, g_4_part1_one_hot)
    	maksed_g_4_part2 = attention_mask(g_4, g_4_part2_one_hot)
    	maksed_g_4_part3 = attention_mask(g_4, g_4_part3_one_hot)


    return g_4, maksed_g_4_part0, maksed_g_4_part1, maksed_g_4_part2, maksed_g_4_part3


def part0_discriminator(inputs, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("part0_DIS", reuse=reuse):

        d_1 = tf.nn.conv3d(inputs, weights['Part0_wd1'], strides=strides, padding="SAME")
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['Part0_wd2'], strides=strides, padding="SAME")
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['Part0_wd3'], strides=strides, padding="SAME")
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['Part0_wd4'], strides=strides, padding="SAME")
        d_4 = lrelu(d_4, leak_value)


        output = d_4

    return output

def part1_discriminator(inputs, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("part1_DIS", reuse=reuse):

        d_1 = tf.nn.conv3d(inputs, weights['Part1_wd1'], strides=strides, padding="SAME")
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['Part1_wd2'], strides=strides, padding="SAME")
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['Part1_wd3'], strides=strides, padding="SAME")
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['Part1_wd4'], strides=strides, padding="SAME")
        d_4 = lrelu(d_4, leak_value)


        output = d_4

    return output

def part2_discriminator(inputs, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("part2_DIS", reuse=reuse):

        d_1 = tf.nn.conv3d(inputs, weights['Part2_wd1'], strides=strides, padding="SAME")
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['Part2_wd2'], strides=strides, padding="SAME")
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['Part2_wd3'], strides=strides, padding="SAME")
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['Part2_wd4'], strides=strides, padding="SAME")
        d_4 = lrelu(d_4, leak_value)


        output = d_4

    return output

def part3_discriminator(inputs, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("part3_DIS", reuse=reuse):

        d_1 = tf.nn.conv3d(inputs, weights['Part3_wd1'], strides=strides, padding="SAME")
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['Part3_wd2'], strides=strides, padding="SAME")
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['Part3_wd3'], strides=strides, padding="SAME")
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['Part3_wd4'], strides=strides, padding="SAME")
        d_4 = lrelu(d_4, leak_value)

        output = d_4

    return output

def global_discriminator(inputs, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("global_DIS", reuse=reuse):


        d_1 = tf.nn.conv3d(inputs, weights['Global_wd1'], strides=strides, padding="SAME")
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['Global_wd2'], strides=strides, padding="SAME")
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['Global_wd3'], strides=strides, padding="SAME")
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['Global_wd4'], strides=strides, padding="SAME")
        d_4 = lrelu(d_4, leak_value)


        output = d_4

    return output


def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()


    weights['wg1'] = tf.get_variable("gen/wg1", shape=[4, 4, 4, 256, z_size], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("gen/wg2", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("gen/wg3", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("gen/wg4", shape=[4, 4, 4, label_num+1, 64], initializer=xavier_init)


    weights['Global_wd1'] = tf.get_variable("global_DIS/Global_wd1", shape=[4, 4, 4, label_num+1, 64], initializer=xavier_init)
    weights['Global_wd2'] = tf.get_variable("global_DIS/Global_wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['Global_wd3'] = tf.get_variable("global_DIS/Global_wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['Global_wd4'] = tf.get_variable("global_DIS/Global_wd4", shape=[4, 4, 4, 256, 1], initializer=xavier_init)

    weights['Part0_wd1'] = tf.get_variable("part0_DIS/Part0_wd1", shape=[4, 4, 4, label_num+1, 64], initializer=xavier_init)
    weights['Part0_wd2'] = tf.get_variable("part0_DIS/Part0_wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['Part0_wd3'] = tf.get_variable("part0_DIS/Part0_wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['Part0_wd4'] = tf.get_variable("part0_DIS/Part0_wd4", shape=[4, 4, 4, 256, 1], initializer=xavier_init)

    weights['Part1_wd1'] = tf.get_variable("part1_DIS/Part1_wd1", shape=[4, 4, 4, label_num+1, 64], initializer=xavier_init)
    weights['Part1_wd2'] = tf.get_variable("part1_DIS/Part1_wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['Part1_wd3'] = tf.get_variable("part1_DIS/Part1_wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['Part1_wd4'] = tf.get_variable("part1_DIS/Part1_wd4", shape=[4, 4, 4, 256, 1], initializer=xavier_init)

    weights['Part2_wd1'] = tf.get_variable("part2_DIS/Part2_wd1", shape=[4, 4, 4, label_num+1, 64], initializer=xavier_init)
    weights['Part2_wd2'] = tf.get_variable("part2_DIS/Part2_wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['Part2_wd3'] = tf.get_variable("part2_DIS/Part2_wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['Part2_wd4'] = tf.get_variable("part2_DIS/Part2_wd4", shape=[4, 4, 4, 256, 1], initializer=xavier_init)

    weights['Part3_wd1'] = tf.get_variable("part3_DIS/Part3_wd1", shape=[4, 4, 4, label_num+1, 64], initializer=xavier_init)
    weights['Part3_wd2'] = tf.get_variable("part3_DIS/Part3_wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['Part3_wd3'] = tf.get_variable("part3_DIS/Part3_wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['Part3_wd4'] = tf.get_variable("part3_DIS/Part3_wd4", shape=[4, 4, 4, 256, 1], initializer=xavier_init)


    return weights



def trainGAN(checkpoint=None):


    #The training data path
    DATA_PATH = os.path.join(BASE_DIR, 'dataset/' + obj)

    data_volume, data_size = loading_training_data(DATA_PATH, cube_len, label_num)

    with tf.device('/gpu:' + str(device_ID)):

        global_step = tf.Variable(0, trainable=False)

        weights = initialiseWeights()

        z_vector = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)
        x_vector = tf.placeholder(shape=[batch_size, cube_len, cube_len, cube_len, label_num+1], dtype=tf.float32)


    	x_vector_backgruond, x_vector_part0, x_vector_part1, x_vector_part2, x_vector_part3 = tf.split(x_vector, [1,1,1,1,1], 4) 
    	real_input_part0 = attention_mask(x_vector, x_vector_part0)
	real_input_part1 = attention_mask(x_vector, x_vector_part1)
    	real_input_part2 = attention_mask(x_vector, x_vector_part2)
	real_input_part3 = attention_mask(x_vector, x_vector_part3)

        net_g_train, net_g_train_part0, net_g_train_part1, net_g_train_part2, net_g_train_part3 = generator(z_vector, phase_train=True, reuse=False)


        #------------------------------global loss----------------------------
        d_no_sigmoid_output_x_global = global_discriminator(x_vector, phase_train=True, reuse=False)
        d_no_sigmoid_output_z_global = global_discriminator(net_g_train, phase_train=True, reuse=True)

        g_loss_global = -tf.reduce_mean(d_no_sigmoid_output_z_global)
        d_loss_global = tf.reduce_mean(d_no_sigmoid_output_z_global) - tf.reduce_mean(d_no_sigmoid_output_x_global)

        epsilon_penalty_global = tf.square(tf.reduce_mean(d_no_sigmoid_output_x_global))

        #------------------------------part0 loss----------------------------
        d_no_sigmoid_output_x_part0 = part0_discriminator(real_input_part0, phase_train=True, reuse=False)
        d_no_sigmoid_output_z_part0 = part0_discriminator(net_g_train_part0, phase_train=True, reuse=True)

        g_loss_part0 = -tf.reduce_mean(d_no_sigmoid_output_z_part0)
        d_loss_part0 = tf.reduce_mean(d_no_sigmoid_output_z_part0) - tf.reduce_mean(d_no_sigmoid_output_x_part0)

        epsilon_penalty_part0 = tf.square(tf.reduce_mean(d_no_sigmoid_output_x_part0))

        #------------------------------part1 loss----------------------------
        d_no_sigmoid_output_x_part1 = part1_discriminator(real_input_part1, phase_train=True, reuse=False)
        d_no_sigmoid_output_z_part1 = part1_discriminator(net_g_train_part1, phase_train=True, reuse=True)

        g_loss_part1 = -tf.reduce_mean(d_no_sigmoid_output_z_part1)
        d_loss_part1 = tf.reduce_mean(d_no_sigmoid_output_z_part1) - tf.reduce_mean(d_no_sigmoid_output_x_part1)
  
        epsilon_penalty_part1 = tf.square(tf.reduce_mean(d_no_sigmoid_output_x_part1))

        #------------------------------part2 loss----------------------------
        d_no_sigmoid_output_x_part2 = part2_discriminator(real_input_part2, phase_train=True, reuse=False)
        d_no_sigmoid_output_z_part2 = part2_discriminator(net_g_train_part2, phase_train=True, reuse=True)

        g_loss_part2 = -tf.reduce_mean(d_no_sigmoid_output_z_part2)
        d_loss_part2 = tf.reduce_mean(d_no_sigmoid_output_z_part2) - tf.reduce_mean(d_no_sigmoid_output_x_part2)

        epsilon_penalty_part2 = tf.square(tf.reduce_mean(d_no_sigmoid_output_x_part2))

        #------------------------------part3 loss----------------------------
        d_no_sigmoid_output_x_part3 = part3_discriminator(real_input_part3, phase_train=True, reuse=False)
        d_no_sigmoid_output_z_part3 = part3_discriminator(net_g_train_part3, phase_train=True, reuse=True)

        g_loss_part3 = -tf.reduce_mean(d_no_sigmoid_output_z_part3)
        d_loss_part3 = tf.reduce_mean(d_no_sigmoid_output_z_part3) - tf.reduce_mean(d_no_sigmoid_output_x_part3)

        epsilon_penalty_part3 = tf.square(tf.reduce_mean(d_no_sigmoid_output_x_part3))

        #------------------------------purity and smooth term with one-hot operation-----------------
        g_smooth_purity_loss = sig_smooth_purity_sqr(net_g_train, batch_size, cube_len, label_num, True)
        smooth_factor = 20
        purity_factor = 15
        g_smooth_loss_32 = tf.div(tf.reduce_mean(g_smooth_purity_loss[0]), smooth_factor)
        g_purity_loss_32 = tf.div(tf.reduce_mean(g_smooth_purity_loss[1]), purity_factor)

        g_tv_32 = g_smooth_loss_32 + g_purity_loss_32


        #------------------------total g_loss-----------------------
        g_loss_total = g_loss_global + g_loss_part0 + g_loss_part1+ 3*g_loss_part2 + 3*g_loss_part3 + g_tv_32


        alpha = tf.random_uniform(
            shape=[batch_size, 1, 1, 1, 1],
            minval=0.,
            maxval=1.
        )

        differences = net_g_train- x_vector

        inter = []
        for i in range(batch_size):
            inter.append(differences[i] * alpha[i])
        inter = tf.stack(inter)

        interpolates = x_vector + inter
   
        #-------------------------global d loss----------------------
        temp_d5_no_sig = global_discriminator(interpolates, phase_train=True, reuse=True)

        gradients = tf.gradients(temp_d5_no_sig, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        d_loss_global += LAMBDA * gradient_penalty

        #-------------------------part0 d loss----------------------------------
        interpolates_backgruond, interpolates_part0, interpolates_part1, interpolates_part2, interpolates_part3 = tf.split(interpolates, [1,1,1,1,1], 4)  
    	interpolates_input_part0 = attention_mask(interpolates, interpolates_part0)
        
        temp_d5_no_sig_part0 = part0_discriminator(interpolates_input_part0, phase_train=True, reuse=True)

        gradients_part0 = tf.gradients(temp_d5_no_sig_part0, [interpolates_input_part0])[0]
        slopes_part0 = tf.sqrt(tf.reduce_sum(tf.square(gradients_part0), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty_part0 = tf.reduce_mean((slopes_part0 - 1.) ** 2)
        d_loss_part0 += LAMBDA * gradient_penalty_part0

        #-------------------------part1 d loss----------------------------------
    	interpolates_input_part1 = attention_mask(interpolates, interpolates_part1)
        
        temp_d5_no_sig_part1 = part1_discriminator(interpolates_input_part1, phase_train=True, reuse=True)

        gradients_part1 = tf.gradients(temp_d5_no_sig_part1, [interpolates_input_part1])[0]
        slopes_part1 = tf.sqrt(tf.reduce_sum(tf.square(gradients_part1), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty_part1 = tf.reduce_mean((slopes_part1 - 1.) ** 2)
        d_loss_part1 += LAMBDA * gradient_penalty_part1


        #-------------------------part2 d loss----------------------------------
    	interpolates_input_part2 = attention_mask(interpolates, interpolates_part2)
        
        temp_d5_no_sig_part2 = part2_discriminator(interpolates_input_part2, phase_train=True, reuse=True)

        gradients_part2 = tf.gradients(temp_d5_no_sig_part2, [interpolates_input_part2])[0]
        slopes_part2 = tf.sqrt(tf.reduce_sum(tf.square(gradients_part2), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty_part2 = tf.reduce_mean((slopes_part2 - 1.) ** 2)
        d_loss_part2 += LAMBDA * gradient_penalty_part2

        #-------------------------part3 d loss----------------------------------
    	interpolates_input_part3 = attention_mask(interpolates, interpolates_part3)
        
        temp_d5_no_sig_part3 = part3_discriminator(interpolates_input_part3, phase_train=True, reuse=True)

        gradients_part3 = tf.gradients(temp_d5_no_sig_part3, [interpolates_input_part3])[0]
        slopes_part3 = tf.sqrt(tf.reduce_sum(tf.square(gradients_part3), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty_part3 = tf.reduce_mean((slopes_part3 - 1.) ** 2)
        d_loss_part3 += LAMBDA * gradient_penalty_part3

        summary_d_loss_global = tf.summary.scalar("d_loss_global", d_loss_global)
        summary_g_loss_global = tf.summary.scalar("g_loss_global", g_loss_global)
        
        summary_d_loss_part0 = tf.summary.scalar("d_loss_part0", d_loss_part0)
        summary_g_loss_part0 = tf.summary.scalar("g_loss_part0", g_loss_part0)

        summary_d_loss_part1 = tf.summary.scalar("d_loss_part1", d_loss_part1)
        summary_g_loss_part1 = tf.summary.scalar("g_loss_part1", g_loss_part1)

	summary_d_loss_part2 = tf.summary.scalar("d_loss_part2", d_loss_part2)
        summary_g_loss_part2 = tf.summary.scalar("g_loss_part2", g_loss_part2)

        summary_d_loss_part3 = tf.summary.scalar("d_loss_part3", d_loss_part3)
        summary_g_loss_part3 = tf.summary.scalar("g_loss_part3", g_loss_part3)
 
	summary_g_loss_total = tf.summary.scalar("g_loss_total", g_loss_total)

        summary_purity_loss = tf.summary.scalar("purity_loss", g_purity_loss_32)
        summary_smoothness_loss = tf.summary.scalar("smooth_loss", g_smooth_loss_32)
        summary_g_tv_32 = tf.summary.scalar("g_tv_32", g_tv_32)



        net_g_test, net_g_test_part0, net_g_test_part1 ,net_g_test_part2, net_g_test_part3 = generator(z_vector, phase_train=False, reuse=True)

        para_g = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'bg', 'gen'])]
        para_d_global = [var for var in tf.trainable_variables() if any(x in var.name for x in ['global_DIS'])]
	para_d_part0 = [var for var in tf.trainable_variables() if any(x in var.name for x in ['part0_DIS'])]
	para_d_part1 = [var for var in tf.trainable_variables() if any(x in var.name for x in ['part1_DIS'])]
	para_d_part2 = [var for var in tf.trainable_variables() if any(x in var.name for x in ['part2_DIS'])]
	para_d_part3 = [var for var in tf.trainable_variables() if any(x in var.name for x in ['part3_DIS'])]

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            # only update the weights for the discriminator network
	    optimizer_op_d_global = tf.train.AdamOptimizer(learning_rate=BASE_LEARNING_RATE_DISCRIMINATOR,beta1=beta).minimize(d_loss_global,var_list=para_d_global, global_step=global_step)
	    optimizer_op_d_part0 = tf.train.AdamOptimizer(learning_rate=BASE_LEARNING_RATE_DISCRIMINATOR,beta1=beta).minimize(d_loss_part0,var_list=para_d_part0, global_step=global_step)
	    optimizer_op_d_part1 = tf.train.AdamOptimizer(learning_rate=BASE_LEARNING_RATE_DISCRIMINATOR,beta1=beta).minimize(d_loss_part1,var_list=para_d_part1, global_step=global_step)
	    optimizer_op_d_part2 = tf.train.AdamOptimizer(learning_rate=BASE_LEARNING_RATE_DISCRIMINATOR,beta1=beta).minimize(d_loss_part2,var_list=para_d_part2, global_step=global_step)
	    optimizer_op_d_part3 = tf.train.AdamOptimizer(learning_rate=BASE_LEARNING_RATE_DISCRIMINATOR,beta1=beta).minimize(d_loss_part3,var_list=para_d_part3, global_step=global_step)
            # only update the weights for the generator network
            optimizer_op_g = tf.train.AdamOptimizer(learning_rate=BASE_LEARNING_RATE_GENERATOR,beta1=beta).minimize(g_loss_total,var_list=para_g, global_step=global_step)


        saver = tf.train.Saver() 


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
        merged_summary_op = tf.summary.merge_all()

        ckpt_folder = MODEL_SAVE
        ckpt = tf.train.get_checkpoint_state(ckpt_folder)
        if checkpoint is not None:
            print "Loading previous model..."
            saver.restore(sess, checkpoint)
            print "Done."
            restore_epoch = os.path.splitext(os.path.basename(checkpoint))[0]
        else:
            restore_epoch = 0
            print "No saved model found."

        for epoch in range(int(restore_epoch), n_epochs):

            for k in range(CRITIC_ITERS):
                idx_CRITIC_ITERS = np.random.randint(data_size, size=batch_size)
                x_CRITIC_ITERS = data_volume[idx_CRITIC_ITERS]
                z__CRITIC_ITERS = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)
                summary_merge, _,  _,  _,  _, _,  = sess.run([merged_summary_op, optimizer_op_d_global, optimizer_op_d_part0,  optimizer_op_d_part1, optimizer_op_d_part2, optimizer_op_d_part3],feed_dict={z_vector: z__CRITIC_ITERS,x_vector: x_CRITIC_ITERS})

            _, generator_loss_total= sess.run([optimizer_op_g,  g_loss_total], feed_dict={z_vector: z__CRITIC_ITERS})
            print 'Generator Training ', "epoch: ", epoch, 'g_loss_total:', generator_loss_total

            summary_writer.add_summary(summary_merge, epoch)

            # output generated chairs
            if epoch % 500 == 10:
                z_sample = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)
                g_models = sess.run(net_g_test, feed_dict={z_vector: z_sample})

                g_models.dump(SAMPLES_DURING_TRAINING + '/' + 'epoch_'+str(epoch)+'.npy')

                saver.save(sess, save_path=MODEL_SAVE + '/' + str(epoch) + '.cptk')


if __name__ == '__main__':
    trainGAN(check_point)
