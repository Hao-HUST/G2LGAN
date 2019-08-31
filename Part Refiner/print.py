import tensorflow as tf
import os
import sys
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

'''
#The saving path for AE model
MODEL_SAVE = os.path.join(BASE_DIR, 'result/model/'+'chair/')


latest_ckp = tf.train.latest_checkpoint('/media/vcc/759604d1-150a-4a89-900a-b63a424584e8/GitHub/G2LGAN/result/chair/model/')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')


from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path = os.path.join(model_dir, "model.ckpt")
#The saving path for AE model
'''

#checkpoint_path = os.path.join(BASE_DIR, 'result/model/'+'chair'+'/3901.cptk')
checkpoint_path = '/media/vcc/759604d1-150a-4a89-900a-b63a424584e8/GitHub/G2LGAN/result/chair/model/20010.cptk'


reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(reader.get_tensor(key)) # Remove this is you want to print only variable names
