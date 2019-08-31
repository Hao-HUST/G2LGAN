import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
import sys
import scipy.io as sio
import part_ae as nw
from utils import *
import argparse
from utils import separate_parts


#The hyper-parameter for trainig model
LEARNING_RATE = 0.0005
Z_SIZE = 64
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
MODEL_SAVE = os.path.join(BASE_DIR, 'result/model/'+obj+'/3901.cptk')

# Loading the initial shape from G2LGAN and seperate them into 
#different parts
#We assume the generated data are .npy files
def loading_and_seperate_gen_data(gen_data_path, part_num, dim):

    print 'Loading gen data for refining'
    gen_data_filelist =  os.listdir(gen_data_path)
    len_gen_data_filelist = len(gen_data_filelist)


    for i in xrange(len_gen_data_filelist):
        print 'File:', i
        if i==0:
            gen_data = load_data(gen_data_path + '/' + gen_data_filelist[i])
        else:
            gen_data = np.concatenate((gen_data, load_data(gen_data_path + '/' + gen_data_filelist[i])), axis=0)

    #Seperate the gen shapes into individual parts
    gen_data_parts = {}
    gen_data_parts = separate_parts(gen_data, dim, part_num)


    return gen_data_parts

def test():

    DATA_PATH = os.path.join(BASE_DIR, 'testing_data/initial_gen_data/'+obj)

    #The saving path for the refined shapes
    SAMPLE_SAVE = os.path.join(BASE_DIR, 'result/testing/rec/'+obj)
    if not os.path.exists(SAMPLE_SAVE):
        os.makedirs(SAMPLE_SAVE)

    #Loading traing data pair for PR
    gen_data_parts = {}
    gen_data_parts =  loading_and_seperate_gen_data(DATA_PATH,label_num, 32)


    # setup network
    network = nw.Network(BATCH_SIZE, weight, is_training, skip, label_num, loss_type, output_layer, device_ID)

    saver = tf.train.Saver()

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print ("Loading previous model...")
        saver.restore(sess, MODEL_SAVE)
        print ("Done.")


        for sample in xrange(len(gen_data_parts[0]) / 16):
            num_test_chairs = 16
            test_set = np.empty((0, 32, 32, 32, 2))
            test_set_labels = np.zeros((0, label_num))

            for j in xrange(sample * num_test_chairs, (sample + 1) * num_test_chairs):
                test_set = np.concatenate((test_set, np.reshape(gen_data_parts[0][j], np.concatenate(([1], gen_data_parts[0][j].shape)))))

                test_set_labels = np.concatenate((test_set_labels, [[1, 0, 0, 0]]))
                test_set = np.concatenate((test_set, np.reshape(gen_data_parts[1][j], np.concatenate(([1], gen_data_parts[1][j].shape)))))
                test_set_labels = np.concatenate((test_set_labels, [[0, 1, 0, 0]]))
                test_set = np.concatenate((test_set, np.reshape(gen_data_parts[2][j], np.concatenate(([1], gen_data_parts[2][j].shape)))))
                test_set_labels = np.concatenate((test_set_labels, [[0, 0, 1, 0]]))

                test_set = np.concatenate((test_set, np.reshape(gen_data_parts[3][j], np.concatenate(([1], gen_data_parts[3][j].shape)))))
                test_set_labels = np.concatenate((test_set_labels, [[0, 0, 0, 1]]))


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

            np.save(SAMPLE_SAVE + '/'+str(sample) + '_test', rec_chairs)
            #np.save(SAMPLE_SAVE + '/'+str(sample) + '_test_input', global_shape[sample * num_test_chairs: (sample + 1) * num_test_chairs])







    
if __name__ == "__main__":
    test()



