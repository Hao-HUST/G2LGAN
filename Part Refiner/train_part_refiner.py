import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
import sys
import scipy.io as sio
import part_ae as nw
from utils import *
import argparse


#The hyper-parameter for trainig model
LEARNING_RATE = 0.0005
Z_SIZE = 64
EPOCHS = 8010
RATIO = '9-1'
TEST_SET = 'ICON2-crop-cube'


loss_type = 'crosse'
skip = '0'
output_layer = 'softmax'
weight = '-'  # 0.624-0.365-0.011


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))


parser = argparse.ArgumentParser()
parser.add_argument('--input_obj', default='chair', help='The input data category [default: chair]')
parser.add_argument('--num_semantic_parts', type=int, default=4, help='The amount of semantic parts on shape')
parser.add_argument('--GPU_ID', type=int, default=0, help='Choose which GPU device you want to use')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='The batch size of data feed into your network')
parser.add_argument('--is_training', type=bool, default=True, help='Set the flag as true if you need to train the model')
parser.add_argument('--voxel_resolution', type=int, default=64, help='The voxel resolution represent the output 3D shape')

FLAGS = parser.parse_args()

#choose your GPU device
device_ID = FLAGS.GPU_ID
obj = FLAGS.input_obj
label_num = FLAGS.num_semantic_parts
BATCH_SIZE = FLAGS.BATCH_SIZE
is_training = FLAGS.is_training
DIM = FLAGS.voxel_resolution



#The saving path for AE model
MODEL_SAVE = os.path.join(BASE_DIR, 'result/model/'+obj)

#The saving path for the refined shapes
SAMPLE_SAVE = os.path.join(BASE_DIR, 'result/training/rec/'+obj)

#The saving path for log files
LOG_PATH = os.path.join(BASE_DIR, 'result/log/'+obj)


if not os.path.exists(MODEL_SAVE):
    os.makedirs(MODEL_SAVE)
if not os.path.exists(SAMPLE_SAVE):
    os.makedirs(SAMPLE_SAVE)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)





def Loading_data(training_data_path, part_num):

    print 'Loading pair data for PR training'

    real_path_32 = training_data_path + '/real_32/'
    real_path_64 = training_data_path + '/real_64/'

    gen_path = training_data_path + '/gen_32/'
    pair_index_path = training_data_path+ '/pair_index/'

    part_pairs_index = {}
    pairs = np.empty((0, 3)) 


    print 'Loading Pair index'
    for ID in xrange(part_num):
    	part_pairs_index[ID] = np.array(load_data(pair_index_path + 'pair_index_part_'+str(ID)+'.npy'), dtype=int)
    # Adding label column
    for ID in xrange(part_num):
    	part_pairs_index[ID] = np.concatenate((part_pairs_index[ID], np.zeros((part_pairs_index[ID].shape[0], 1))), axis=1)


    #Loading real part with 32 resolution
    print 'Loading Real Parts 32'
    for ID in xrange(part_num):
        if ID==0:
    	    data= load_data(real_path_32 + 'part'+str(ID)+'.npy')
            for i in xrange(len(data)):
                pairs = np.concatenate((pairs, np.array([[i, i, ID]])), axis=0)
        else:
            data = np.concatenate((data, load_data(real_path_32 + 'part'+str(ID)+'.npy')), axis=0)
            pairs_len = len(pairs)
            for i in xrange(pairs_len, len(data)):
                pairs = np.concatenate((pairs, np.array([[i, i, ID]])), axis=0)




    print 'Loading Gen Parts'
    for ID in xrange(part_num):
    	part_pairs_index[ID][:, 0] += len(data)
        data = np.concatenate((data, load_data(gen_path + 'part'+str(ID)+'.npy')), axis=0)


    print 'Loading Real Parts_64'
    for ID in xrange(part_num):
        if ID==0:
    	    data_64= load_data(real_path_64 + 'part'+str(ID)+'.npy')

        else:
	    part_pairs_index[ID][:, 1] += len(data_64)
            data_64 = np.concatenate((data_64, load_data(real_path_64 + 'part'+str(ID)+'.npy')), axis=0)
     
    
          
    for ID in xrange(part_num):
        pairs = np.concatenate((pairs, part_pairs_index[ID]), axis=0)

    pairs = np.array(pairs, dtype=int)
    pairs = pairs.astype(int)



    print 'Loading Test Data'

    test_set = load_data(training_data_path + '/test/test_chair.npy')
    test_set_labels = load_data(training_data_path + '/test/test_label_chair.npy')
    print 'Finished loading data'


    return data, data_64, pairs, test_set, test_set_labels



def train():

    DATA_PATH = os.path.join(BASE_DIR, 'training_data/'+obj)

    #Loading traing data pair for PR
    data, data_64, pairs, test_set, test_set_labels =  Loading_data(DATA_PATH,label_num)

    # setup network
    network = nw.Network(BATCH_SIZE, weight, is_training, skip, label_num, loss_type, output_layer, device_ID)

    para_rec = [var for var in tf.trainable_variables()]


    # bn para
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('train'):
            optimizer_rec = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(network.rec_loss,
                                                                                             var_list=para_rec)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        ckpt = None
    
        if ckpt and ckpt.model_checkpoint_path:
            print ("Loading previous model...")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print ("Done.")
        else:
            print ("No saved model found.")

        for epoch in range(0, EPOCHS):
            x_obj, label, y_obj = load_batch(data, data_64, pairs, BATCH_SIZE,label_num)

            train_log, _, rec_loss = sess.run([merged, optimizer_rec, network.rec_loss],
                                          feed_dict={network.x_obj: x_obj, network.y_obj: y_obj, network.label: label})
            writer.add_summary(train_log, epoch)
            print ('Epoche :', epoch, 'loss:', rec_loss)

            if epoch % 100 == 1:
                saver.save(sess, save_path =(MODEL_SAVE + '/' + str(epoch) + '.cptk'))
                print(epoch, 'Model saved')

                rec = sess.run(network.iobj, feed_dict={network.x_obj: x_obj, network.y_obj: y_obj, network.label: label})


                # reconstruct the whole chair
                #solve the conflict of voxel
                ##### needs to change base on category !!! #######
                num_test_chairs = 16
                rec_parts = sess.run(network.x_rec, feed_dict={network.x_obj: test_set, network.label: test_set_labels})
                rec_chairs = np.zeros((num_test_chairs, DIM, DIM, DIM, label_num+1))
                for c in xrange(num_test_chairs):
                    base_idx = label_num * c
                    has_back = np.count_nonzero(test_set[base_idx, :, :, :, 1]) > 0
                    has_seat = np.count_nonzero(test_set[base_idx + 1, :, :, :, 1]) > 0
                    has_legs = np.count_nonzero(test_set[base_idx + 2, :, :, :, 1]) > 0
                    has_arm_rest = np.count_nonzero(test_set[base_idx + 3, :, :, :, 1]) > 0
                    for row in xrange(rec_parts.shape[1]):
                        for col in xrange(rec_parts.shape[2]):
                            for dep in xrange(rec_parts.shape[3]):
                                # seat
                                if has_seat and rec_parts[base_idx + 1, row, col, dep, 1] > rec_parts[base_idx + 1, row, col, dep, 0]:
                                    rec_chairs[c, row, col, dep] = [0, 0, 1, 0, 0]
                                # back
                                elif has_back and rec_parts[base_idx, row, col, dep, 1] > rec_parts[base_idx, row, col, dep, 0]:
                                    rec_chairs[c, row, col, dep] = [0, 1, 0, 0, 0]
                                # legs
                                elif has_legs and rec_parts[base_idx + 2, row, col, dep, 1] > rec_parts[base_idx + 2, row, col, dep, 0]:
                                    rec_chairs[c, row, col, dep] = [0, 0, 0, 1, 0]
                                # arm rests
                                elif has_arm_rest and rec_parts[base_idx + 3, row, col, dep, 1] > rec_parts[base_idx + 3, row, col, dep, 0]:
                                    rec_chairs[c, row, col, dep] = [0, 0, 0, 0, 1]
                                # background
                                else:
                                    rec_chairs[c, row, col, dep] = [1, 0, 0, 0, 0]
                np.save(SAMPLE_SAVE + str(epoch) + '_test', rec_chairs)











    
if __name__ == "__main__":
    train()


