import tensorflow as tf
import numpy as np

#CHAIR_WEIGHT = ['0.02345-0.97655', '0.03-0.97', '0.01293-0.98707', '0.14-0.86'] # 32
#CHAIR_WEIGHT = ['0.017346-0.982654', '0.0242-0.9758', '0.008236-0.991764', '0.010608-0.989392'] # 64
#CHAIR_WEIGHT = ['0.20659-0.79341', '0.22578-0.77422', '0.1684-0.8316', '0.18067-0.81933'] # 64^(1/3)
#CHAIR_WEIGHT = ['0.11728-0.88272', '0.13606-0.86394', '0.1421-0.8579', '0.09383-0.90617'] # 64^(1/2)
#CHAIR_WEIGHT = ['0.0486-0.9514', '0.0597-0.9403', '0.0465-0.9535', '0.0418-0.9582'] # 64^guess
#CHAIR_WEIGHT = ['0.0486-0.9514', '0.0597-0.9403', '0.0465-0.9535', '0.0418-0.9582'] # 64 plane

#CHAIR_WEIGHT = ['0.1804-0.8196', '0.1134-0.8866', '0.1426-0.8574'] # 64 table
#CHAIR_WEIGHT = ['0.074-0.926', '0.143-0.857', '0.04-0.96', '0.074-0.926'] # 64 lamp

CHAIR_WEIGHT = ['0.134-0.866', '0.158-0.842', '0.092-0.908', '0.104-0.896'] # 64 chair
'''
CHAIR_WEIGHT 
'''

WEIGHT = CHAIR_WEIGHT

DIM_IN = 32
DIM_OUT = 64
LEAKY_VALUE = 0.2


class Network:
    def __init__(self, batch_size, weight, bn_set, skip, cate_dim, loss_type, output_layer, device):
        self.batch_size = batch_size
        self.weight = weight
        self.is_training = bn_set
        self.skip = int(skip)
        self.cate_dim = cate_dim
        self.loss_type = loss_type
        self.output_layer = output_layer
        with tf.device('/gpu:' + str(device)):
            with tf.name_scope('inputs'):
                self.x_obj = tf.placeholder(
                    tf.float32, [self.batch_size, DIM_IN, DIM_IN, DIM_IN, 2], name='x_obj')
                self.y_obj = tf.placeholder(
                    tf.float32, [self.batch_size, DIM_OUT, DIM_OUT, DIM_OUT, 2], name='y_obj')
                self.label = tf.placeholder(
                    tf.float32, [self.batch_size, self.cate_dim], name='label')

            with tf.variable_scope('encoder'):#, reuse=tf.AUTO_REUSE):
                out_e2, out_e3, z = self.encoder(self.x_obj, self.label, is_training=self.is_training)

            with tf.variable_scope('decoder'):
                self.iobj = self.decoder_iobj(z, out_e2, out_e3, is_training=self.is_training)

            self.x_rec = self.iobj

            with tf.name_scope('loss'):
                y_obj_list = tf.unstack(self.y_obj, axis=0)
                x_rec_list = tf.unstack(self.x_rec, axis=0)
                label_list = tf.unstack(self.label, axis=0)
                rec_loss_list = []
                for i in range(self.batch_size):
                    temp = WEIGHT[np.argmax(label_list[i])].split('-')
                    w = []
                    for b in temp:
                        w.append(float(b))
                    if self.loss_type == 'icrosse':
                        w[0] = 0
                    rec_w = tf.constant(w, dtype=tf.float32)
                    rec_loss = -tf.reduce_mean(rec_w * (y_obj_list[i] * tf.log(x_rec_list[i] + 1e-9)))
                    rec_loss_list.append(rec_loss)

                rec_loss = tf.stack(rec_loss_list, axis=0)
                self.rec_loss = tf.reduce_mean(rec_loss)
                tf.summary.scalar('rec_loss', self.rec_loss)

        print('batch_size:', batch_size, 'weight', weight, 'bn', bn_set, 'skip', skip, 'cate_dim', cate_dim,
              'loss_type', loss_type, 'output_layer', output_layer)

    def encoder(self, x, label, is_training=True):
        # encode x
        conv_e1 = self.conv3d_layer(x, 32, 'conv_e1')
        out_e1 = tf.nn.relu(conv_e1)
        print(out_e1)

        conv_e2 = self.conv3d_layer(out_e1, 64, 'conv_e2')
        out_e2 = tf.nn.relu(self.batch_norm(conv_e2, is_training, 'bn_e2'))
        print(out_e2)

        conv_e3 = self.conv3d_layer(out_e2, 128, 'conv_e3')
        out_e3 = tf.nn.relu(self.batch_norm(conv_e3, is_training, 'bn_e3'))
        print(out_e3)

        #conv_e4 = self.conv3d_layer(out_e3, 256, 'conv_e4')
        #out_e4 = tf.nn.relu(self.batch_norm(conv_e4, is_training, 'bn_e4'))
        #print(out_e4)

        #fc_e5 = tf.reshape(out_e4, [self.batch_size, -1])
        fc_e5 = tf.reshape(out_e3, [self.batch_size, -1])
        fc_e5 = self.fc_layer(fc_e5, 128, 'fc_e5')
        out_fc_e5 = tf.nn.relu(self.batch_norm(fc_e5, is_training, 'bn_e5'))
        print(out_fc_e5)
        fx = out_fc_e5

        # encode c
        fc_e6 = self.fc_layer(label, 128, 'fc_e6')
        out_fc_e6 = tf.nn.relu(self.batch_norm(fc_e6, is_training, 'bn_e6'))
        print(out_fc_e6)

        fc_e7 = self.fc_layer(out_fc_e6, 128, 'fc_e7')
        out_fc_e7 = tf.nn.relu(self.batch_norm(fc_e7, is_training, 'bn_e7'))
        print(out_fc_e7)

        fc_e8 = self.fc_layer(out_fc_e7, 128, 'fc_e8')
        out_fc_e8 = tf.nn.relu(self.batch_norm(fc_e8, is_training, 'bn_e8'))
        print(out_fc_e8)
        fc = out_fc_e8

        # concat fx, fc
        z = tf.concat([fx, fc], -1)

        z = self.fc_layer(z, 256, 'fc_e9')
        z = tf.nn.relu(self.batch_norm(z, is_training, 'bn_e9'))

        z = self.fc_layer(z, 256, 'fc_e10')
        z = tf.nn.relu(self.batch_norm(z, is_training, 'bn_e10'))
        print(z)

        #return out_e2, out_e3, out_e4, z
        return out_e2, out_e3, z


    def decoder_iobj(self, z, out_e2, out_e3, is_training=True):
        # z = self.fc_layer(z, 256*4*4*4, 'fc_d1')
        d1 = tf.reshape(z, (self.batch_size, 1, 1, 1, 256))
        d1 = self.conv3d_trans_layer(d1, 256, (self.batch_size, 4, 4, 4, 256), 'conv_trans_d1', d_h=1, d_w=1, d_d=1,
                                     padding='VALID')
        out_d1 = tf.nn.relu(d1)
        # out_d1 = tf.nn.relu((self.batch_norm(d1, is_training, 'bn_d1')))
        print(out_d1)

        #if self.skip >= 1:
        #    out_d1 = tf.concat([out_d1, out_e4], 4)
        conv_d2 = self.conv3d_trans_layer(out_d1, 128, (self.batch_size, 8, 8, 8, 128), 'conv_trans_d2')
        out_d2 = tf.nn.relu(self.batch_norm(conv_d2, is_training, 'bn_d2'))
        print(out_d2)

        #if self.skip >= 2:
        #    out_d2 = tf.concat([out_d2, out_e3], 4)
        conv_d3 = self.conv3d_trans_layer(out_d2, 64, (self.batch_size, 16, 16, 16, 64), 'conv_trans_d3')
        out_d3 = tf.nn.relu(self.batch_norm(conv_d3, is_training, 'bn_d3'))
        print(out_d3)

        #if self.skip >= 3:
        #    out_d3 = tf.concat([out_d3, out_e2], 4)
        if DIM_OUT == 32:
            conv_d4 = self.conv3d_trans_layer(out_d3, 2, (self.batch_size, 32, 32, 32, 2), 'conv_trans_d4')
            if self.output_layer == 'sigmoid':
                out_d4 = tf.nn.sigmoid(conv_d4)
            else:
                out_d4 = tf.nn.softmax(conv_d4, -1)
            print(out_d4)
            return out_d4

        conv_d4 = self.conv3d_trans_layer(out_d3, 32, (self.batch_size, 32, 32, 32, 32), 'conv_trans_d4')
        out_d4 = tf.nn.relu(self.batch_norm(conv_d4, is_training, 'bn_d4'))
        conv_d5 = self.conv3d_trans_layer(out_d4, 2, (self.batch_size, DIM_OUT, DIM_OUT, DIM_OUT, 2), 'conv_trans_d5')

        if self.output_layer == 'sigmoid':
            out_d5 = tf.nn.sigmoid(conv_d5)
        else:
            out_d5 = tf.nn.softmax(conv_d5, -1)
        print(out_d5)

        return out_d5



    def conv3d_trans_layer(self, inputs, out_dim, output_shape, name, k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2,
                           padding='SAME'):
        with tf.name_scope('conv_trans_layer'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(name=name + '/weights',
                                          shape=[k_d, k_h, k_w, out_dim, inputs.get_shape()[-1]],
                                          initializer=tf.contrib.layers.xavier_initializer())
                tf.summary.histogram(name + '/weights', weights)
            with tf.name_scope('conv_trans_out'):
                conv_trans = tf.nn.conv3d_transpose(inputs, weights, output_shape, strides=[1, d_d, d_h, d_w, 1],
                                                    padding=padding)
        return conv_trans

    def conv3d_layer(self, inputs, out_dim, name, k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, padding='SAME'):
        with tf.name_scope('conv_layer'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(name=name + '/weights',
                                          shape=[k_d, k_h, k_w, inputs.get_shape()[-1], out_dim],
                                          initializer=tf.contrib.layers.xavier_initializer())
                tf.summary.histogram(name + '/weights', weights)
            with tf.name_scope('conv_out'):
                conv = tf.nn.conv3d(inputs, weights, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        return conv

    def leaky_relu(self, x, alpha=0.2):
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        x = x - tf.constant(alpha, dtype=tf.float32) * negative_part
        return x

    def fc_layer(self, inputs, out_dim, name):
        assert len(inputs.get_shape()) == 2
        with tf.name_scope('fc_layer'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(name=name + '/weights', dtype=tf.float32,
                                          shape=[inputs.get_shape()[1], out_dim],
                                          initializer=tf.contrib.layers.xavier_initializer())
                tf.summary.histogram(name + '/weights', weights)
            with tf.name_scope('biases'):
                biases = tf.get_variable(name=name + '/biases', shape=[out_dim], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
                tf.summary.histogram(name + '/biases', biases)
            with tf.name_scope('fc_out'):
                fc = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
        return fc

    def fc_layer_init(self, inputs, out_dim, name):
        assert len(inputs.get_shape()) == 2
        with tf.name_scope('fc_layer_init'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(name=name + '/weights', dtype=tf.float32,
                                          shape=[inputs.get_shape()[1], out_dim],
                                          initializer=tf.constant_initializer(0.0))
                tf.summary.histogram(name + '/weights', weights)
            with tf.name_scope('biases'):
                biases = tf.get_variable(name=name + '/biases', shape=[out_dim], dtype=tf.float32,
                                         initializer=tf.constant_initializer([1, 0, 0, 0]))
                tf.summary.histogram(name + '/biases', biases)
            with tf.name_scope('fc_out'):
                fc = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
        return fc

    def batch_norm(self, x, is_training, scope):
        return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None,
                                            epsilon=1e-5, scale=True, is_training=is_training, scope=scope)

    def dis(self, o1, o2):
        eucd2 = tf.pow(tf.subtract(o1, o2), 2)
        eucd2 = tf.reduce_sum(eucd2)
        eucd = tf.sqrt(eucd2 + 1e-6, name='eucd')
        loss = tf.reduce_mean(eucd, name='loss')
        return loss
