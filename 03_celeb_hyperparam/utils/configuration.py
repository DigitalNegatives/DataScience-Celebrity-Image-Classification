from hyperopt import hp
import tensorflow as tf

face_cascade_dir = './utils/haarcascade_frontalface_default.xml'
eye_cascade_dir = './utils/haarcascade_eye.xml'

data_base_dir = "../data/aligned_multi_gaus_mask_4_20"
# data_base_dir = "../data/aligned_multi_conv_5_20_mask_4_20"
corpus_dir = "{}/corpus".format(data_base_dir)
bottleneck_dir = "{}/bottleneck".format(data_base_dir)
log_dir = "./test/logs"
report_dir = "./results"

seed = 42
split = [0.8, 0.0, 0.2]

# Bottle training
bottleneck_train = False
# learning_rate_bottleneck = 0.0003
epochs_bottles = 200
#
# # Cosine Regularizer
beta = 0.1

# # Label training
# learning_rate = 0.003
# # learning_rate = 0.01
# # learning_rate = 0.001
epochs_labels = 100
# dropout_train = 0.75
dropout_test = 1.0
#
#
# #Conv parameters
# n_filters = [32, 32, 8]  #filter output sizes
# filter_sizes = [4, 4, 2]  #
filter_strides = [1, 2, 2, 1]
# #maxpool parameters
# ksize = [1,2,2,1]
# k_strides = [1,2,2,1]
# #FC parameter
n_nodes = 100
#
# #bottleneck layer
n_bottles = 2048


### Hyperparameter Parameters ###
config_n_filters = hp.choice('n_filters', [[32,32,16],[64,64,32]])
config_filter_sizes = hp.choice('filter_sizes', [[5,5,3],[4,4,2],[3,3,2]])
config_dropout_train = hp.choice('dropout_train', [0.5, 0.6, 0.7])
config_activation_conv = hp.choice('activation_conv', [tf.nn.tanh])
config_activation_fc = hp.choice('activation_fc', [tf.nn.tanh, tf.nn.relu])
config_learning_rate_bottleneck = hp.uniform('learning_rate_bottleneck', 0.0001, 0.0007)
config_learning_rate_label = hp.uniform('learning_rate_label', 0.0001, 0.0007)
