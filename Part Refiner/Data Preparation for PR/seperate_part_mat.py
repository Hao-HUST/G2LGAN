import os
import sys
import numpy as np
import scipy.io as sio
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))


parser = argparse.ArgumentParser()
parser.add_argument('--input_obj', default='chair', help='The input data path for seperating [default: None]')
parser.add_argument('--num_semantic_parts', type=int, default=4, help='The amount of semantic parts on shape')
FLAGS = parser.parse_args()

label_num = FLAGS.num_semantic_parts
obj = FLAGS.input_obj


def load_mat(matFile, cube_len):
    data = sio.loadmat(matFile)
    volume_size = (cube_len,cube_len,cube_len) 
    array = np.ndarray(volume_size, np.int32)
    array = data['instance']

    return array



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



def seperate_real_data(input_data_path, output_data_path, part_num):

    dataset_files_32 = []
    dataset_files_64 = []


    real_data_32_path = os.path.join(BASE_DIR, input_data_path+'/real_32')
    real_data_64_path = os.path.join(BASE_DIR, input_data_path+'/real_64')

    real_32_part_save_path = os.path.join(BASE_DIR, output_data_path+'/real_32')
    real_64_part_save_path = os.path.join(BASE_DIR, output_data_path+'/real_64')

    if not os.path.exists(real_32_part_save_path):
        os.makedirs(real_32_part_save_path)
    if not os.path.exists(real_64_part_save_path):
        os.makedirs(real_64_part_save_path)


    models_32 = os.listdir(real_data_32_path)

    volume_32 = np.zeros((len(models_32),32,32,32,part_num+1), dtype=np.float32)
    volume_64 = np.zeros((len(models_32),64,64,64,part_num+1), dtype=np.float32)


    for model in models_32:
        # We assume the real data with res_32 and res_64 share the same name 
        # on real_data_32_path and real_data_64_path, thus we can guarantee the 
        # sperated result for real_32 and real_64 are corresponding
        dataset_files_32.append(real_data_32_path + '/' +  model)
        dataset_files_64.append(real_data_64_path + '/' +  model)

        mp_32 = load_mat(dataset_files_32[-1], 32)
        for x in range(32):
            for y in range(32):
                for z in range(32):
                    volume_32[len(dataset_files_32) - 1, x, y, z, mp_32[x,y,z]] = 1

        mp_64 = load_mat(dataset_files_64[-1], 64)
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    volume_64[len(dataset_files_64) - 1, x, y, z, mp_64[x,y,z]] = 1


    # separates the generated parts
    real_shape_parts_32 = separate_parts(volume_32, 32, part_num)
    real_shape_parts_64 = separate_parts(volume_64, 64, part_num)


    empty_32 = np.zeros((3,32,32,32,2))
    empty_32[:,:,:,:,0] = 1
    empty_64 = np.zeros((3,64,64,64,2))
    empty_64[:,:,:,:,0] = 1

    for i in range(part_num):
        idx_to_remove = np.empty(0, dtype=int)
        print i, 'Before', len(real_shape_parts_32[i]), len(real_shape_parts_64[i])
        for j in xrange(len(real_shape_parts_32[i])):
            if np.count_nonzero(real_shape_parts_32[i][j, :, :, :, 1]) == 0:
                idx_to_remove = np.append(idx_to_remove, [j])	  


        np.delete(real_shape_parts_32[i], idx_to_remove, axis=0)
        np.delete(real_shape_parts_64[i], idx_to_remove, axis=0)
        real_shape_parts_32[i] = np.concatenate((real_shape_parts_32[i], empty_32), axis=0)
        real_shape_parts_64[i] = np.concatenate((real_shape_parts_64[i], empty_64), axis=0)
        print i, 'After', len(real_shape_parts_32[i]), len(real_shape_parts_64 [i])

        np.save(real_32_part_save_path+'/part'+str(i)+'.npy', real_shape_parts_32[i])
        np.save(real_64_part_save_path+'/part'+str(i)+'.npy', real_shape_parts_64[i])





def main():


    REAL_DATA_PATH = os.path.join(BASE_DIR, 'input_data/'+obj)

    REAL_SHAPE_PART_SAVE_PATH = os.path.join(BASE_DIR, 'seperate_result/'+obj)

    seperate_real_data(REAL_DATA_PATH, REAL_SHAPE_PART_SAVE_PATH, label_num)


    
if __name__ == "__main__":
    main()
