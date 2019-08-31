# we generated the training pair for Part Refiner is based on the shapes with res 32

import numpy as np
import scipy.io as sio
import os
import sys
import argparse
import itertools


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

parser = argparse.ArgumentParser()
parser.add_argument('--input_obj', default='chair', help='The input data path for seperating [default: None]')
parser.add_argument('--num_semantic_parts', type=int, default=4, help='The amount of semantic parts on shape')
FLAGS = parser.parse_args()

label_num = FLAGS.num_semantic_parts
obj = FLAGS.input_obj


def load_mat(matFile, cube):
    data = sio.loadmat(matFile)
    volume_size = (cube, cube, cube)
    array = np.ndarray(volume_size, np.int32)
    array = data['instance']
    return array


def load_real_data_part(real_part_path, label_num):
    real_parts = {}

    for i in xrange(label_num):
        real_part_path = os.path.join(real_part_path, 'part'+str(i))
        real_parts[i] = np.load(real_part_path)

    return real_part



def load_np_array(path):
    volume = np.load(path)
    return volume



def l1_distance(chair, data, part_num):
    dist = np.zeros(len(data))
    for i in xrange(len(data)):
        dist[i] = np.sum(np.abs(chair - data[i]))
    return np.argpartition(dist, part_num)[:part_num]


def intersection_over_union(chair, data, part_num):

    data = np.squeeze(data)
    dist = np.zeros(len(data))
    
    for i in xrange(len(data)):
        intersection = np.count_nonzero((data[i] - 0.5 * chair) == 0.5)
        union = np.count_nonzero(data[i] + chair)
        dist[i] = intersection / float(union)
    return np.argpartition(dist, -part_num)[-part_num:]



def load_real_part(input_path, part_num):

    print 'Loading data'
    loaded_real_parts = {}

    for ID in xrange(part_num):
    	loaded_real_parts[ID] = np.load(input_path + '/part'+str(ID)+'.npy'),
     
    return loaded_real_parts



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


def prepare_pair_for_PR(input_data_path, real_parts, dim, part_num):


    gen_parts = {}
    pairs = {}

    for ID in xrange(part_num):
    	gen_parts[ID] = np.empty([0, dim, dim, dim, 2])
        pairs[ID] = np.empty([0, 2])


    gen_data_filelist =  os.listdir(input_data_path)
    len_gen_data_filelist = len(gen_data_filelist)

    dataset_files_32 = []


    for i in xrange(len_gen_data_filelist):
    	print 'File:', i
        #we assume the initial generated shapes are .npy files
    	gen_shapes = load_np_array(input_data_path + '/' + gen_data_filelist[i])

    	# numpy convert to one-hot
    	print 'Converts to One-Hot'
    	max_idx = np.argmax(gen_shapes, axis=4)
    	max_idx = np.reshape(max_idx, (gen_shapes.shape[0] * gen_shapes.shape[1] * gen_shapes.shape[2] * gen_shapes.shape[3]))
    	tmp = np.zeros((gen_shapes.shape[0] * gen_shapes.shape[1] * gen_shapes.shape[2] * gen_shapes.shape[3], part_num+1))
    	tmp[np.arange(gen_shapes.shape[0] * gen_shapes.shape[1] * gen_shapes.shape[2] * gen_shapes.shape[3]), max_idx] = 1
    	gen_shapes = np.reshape(tmp, gen_shapes.shape)

    	# separates the generated parts
    	gen_shapes_parts = separate_parts(gen_shapes, 32, part_num)

        for j in xrange(label_num):
            print 'Category:', j
            for gen_part in gen_shapes_parts[j]:
                num_of_arm_rest_voxels = np.count_nonzero(gen_part)
                print(j)

                if num_of_arm_rest_voxels == 0:
                    continue
                elif num_of_arm_rest_voxels < 10:
                    match_idx = l1_distance(gen_part, real_parts[j], part_num)
                else:
                    match_idx = intersection_over_union(gen_part, real_parts[j], part_num)

                reshaped_part = np.reshape(gen_part, np.concatenate(([1], gen_part.shape)))
                gen_parts[j] = np.concatenate((gen_parts[j], reshaped_part), axis=0)

                g = len(gen_parts[j]) - 1

                for m in match_idx:
                    pair = np.array([g, m])
                    reshaped_pair = np.reshape(pair, np.concatenate(([1], pair.shape)))
                    pairs[j] = np.concatenate((pairs[j], reshaped_pair), axis=0)

    return pairs, gen_parts



def main():


    INITIAL_GEN_DATA_PATH = os.path.join(BASE_DIR, 'input_data/'+obj+'/gen_32')
    REAL_32_PART_DATA_PATH = os.path.join(BASE_DIR, 'seperate_result/'+obj+'/real_32')
    
    TRAINING_DATA_FOR_PR_SAVE_PATH = os.path.join(BASE_DIR, 'pair_index/'+obj)
    GEN_SHAPE_PART_SAVE_PATH = os.path.join(BASE_DIR, 'seperate_result/'+obj+'/gen_32')

    if not os.path.exists(TRAINING_DATA_FOR_PR_SAVE_PATH):
        os.makedirs(TRAINING_DATA_FOR_PR_SAVE_PATH)

    if not os.path.exists(GEN_SHAPE_PART_SAVE_PATH):
        os.makedirs(GEN_SHAPE_PART_SAVE_PATH)

    real_data_parts = load_real_part(REAL_32_PART_DATA_PATH, label_num)


    pair_index, gen_data_parts = prepare_pair_for_PR(INITIAL_GEN_DATA_PATH, real_data_parts, 32, label_num)

    for i in xrange(label_num):
        print 'Saving training data for PR', i

        np.save(TRAINING_DATA_FOR_PR_SAVE_PATH + '/pair_index_part_' + str(i), pair_index[i])
        np.save(GEN_SHAPE_PART_SAVE_PATH + '/part' + str(i), gen_data_parts[i])


if __name__ == "__main__":
    main()

