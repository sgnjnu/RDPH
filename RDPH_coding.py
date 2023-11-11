#utf-8
import tensorflow._api.v2.compat.v1 as tf
import scipy.io as sio
import numpy as np
from DataLoader import MyDataLoader
import os
tf.disable_v2_behavior()

feature_dim = 512
bit_length =16
batch_size = 1024
num_epoch = 200
display_step = 20
bit = '%d' % bit_length
HIDDEN_COUNT = 512
HIDDEN_COUNT1 = 512
method_name='RDPH'
dataset_name='mir'
data_dataset_name='mir' # the dataset extracted feature
n_classes = 24
feature_dim1=512


input_ix = tf.placeholder(tf.float32, [None, feature_dim])
input_tx = tf.placeholder(tf.float32, [None, feature_dim1])
input_iy = tf.placeholder(tf.float32, [None, feature_dim])
input_ty = tf.placeholder(tf.float32, [None, feature_dim1])
input_iz = tf.placeholder(tf.float32, [None, feature_dim])
input_tz = tf.placeholder(tf.float32, [None, feature_dim1])
input_ixB= tf.placeholder(tf.float32, [None, bit_length])
input_txB= tf.placeholder(tf.float32, [None, bit_length])
input_iyB= tf.placeholder(tf.float32, [None, bit_length])
input_tyB= tf.placeholder(tf.float32, [None, bit_length])
input_izB= tf.placeholder(tf.float32, [None, bit_length])
input_tzB= tf.placeholder(tf.float32, [None, bit_length])

pre_x = tf.placeholder(tf.float32, [None, n_classes])
pre_y = tf.placeholder(tf.float32, [None, n_classes])
pre_z = tf.placeholder(tf.float32, [None, n_classes])

pos_weight = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

Sxyz_=tf.placeholder(tf.float32, [None, ])
Sxyz=tf.reshape(Sxyz_, [-1, 1])
Syxz_=tf.placeholder(tf.float32, [None, ])
Syxz=tf.reshape(Syxz_, [-1, 1])
Szxy_=tf.placeholder(tf.float32, [None, ])
Szxy=tf.reshape(Szxy_, [-1, 1])

with tf.name_scope('label_hash_network') as scope:
    l_fc1w = tf.Variable(tf.truncated_normal([n_classes, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    l_fc1b = tf.Variable(tf.constant(0.0, shape=[HIDDEN_COUNT], dtype=tf.float32),
                         trainable=True, name='x_biases')
    l_fc2w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    l_fc3w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, bit_length],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')


with tf.name_scope('text_hash_network') as scope:
    t_fc1w = tf.Variable(tf.truncated_normal([feature_dim1, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    t_fc1b = tf.Variable(tf.constant(0.0, shape=[HIDDEN_COUNT], dtype=tf.float32),
                         trainable=True, name='x_biases')
    t_fc2w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    t_fc3w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, bit_length],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')


with tf.name_scope('image_hash_network') as scope:
    i_fc1w = tf.Variable(tf.truncated_normal([feature_dim, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    i_fc1b = tf.Variable(tf.constant(0.0, shape=[HIDDEN_COUNT], dtype=tf.float32),
                         trainable=True, name='x_biases')
    i_fc2w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, HIDDEN_COUNT],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
    i_fc3w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, bit_length],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')


with tf.name_scope('feature_discriminator_IT') as scope:
    f_dit_fc1w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    f_dit_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')

with tf.name_scope('hash_discriminator_IT') as scope:
    h_dit_fc1w = tf.Variable(tf.truncated_normal([bit_length, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    h_dit_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')

with tf.name_scope('feature_discriminator_IL') as scope:
    f_dil_fc1w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    f_dil_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')

with tf.name_scope('hash_discriminator_IL') as scope:
    h_dil_fc1w = tf.Variable(tf.truncated_normal([bit_length, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    h_dil_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')


with tf.name_scope('feature_discriminator_TL') as scope:
    f_dtl_fc1w = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    f_dtl_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')

with tf.name_scope('hash_discriminator_TL') as scope:
    h_dtl_fc1w = tf.Variable(tf.truncated_normal([bit_length, 1],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    h_dtl_fc1b = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                       trainable=True, name='biases')



fh_var =[t_fc1w,t_fc1b,t_fc2w,t_fc3w, i_fc1w, i_fc1b, i_fc2w,i_fc3w,l_fc1w,l_fc2w,l_fc1b]
dis_var_list=[f_dit_fc1w,f_dit_fc1b,h_dit_fc1w,h_dit_fc1b,f_dil_fc1w,f_dil_fc1b,h_dil_fc1w,h_dil_fc1b,f_dtl_fc1w,f_dtl_fc1b,h_dtl_fc1w,h_dtl_fc1b]



def image_hash(x_image):
    fc1l = tf.nn.bias_add(tf.matmul(x_image, i_fc1w), i_fc1b)
    fc1 = tf.nn.tanh(fc1l)
    fc2l = tf.matmul(fc1, i_fc2w)
    fc2 = tf.nn.tanh(fc2l)
    fc3l = tf.matmul(fc2, i_fc3w)
    hash = tf.nn.tanh(fc3l)
    return fc2,hash


def text_hash(x_text):
    fc1l = tf.nn.bias_add(tf.matmul(x_text, t_fc1w), t_fc1b)
    fc1 = tf.nn.tanh(fc1l)
    fc2l = tf.matmul(fc1, t_fc2w)
    fc2 = tf.nn.tanh(fc2l)
    fc3l = tf.matmul(fc2, t_fc3w)
    hash = tf.nn.tanh(fc3l)
    return fc2,hash

ixf,ix_hash = image_hash(input_ix)
txf,tx_hash = text_hash(input_tx)


'''prepare data'''
my_data_loader=MyDataLoader(data_dataset_name,batch_size,is_train=False)
code_path = './hash_code/' + dataset_name + '/' + method_name + '/' + str(
    bit_length) + 'bit/'
if not dataset_name==data_dataset_name:
    code_path=code_path+data_dataset_name+'/'

train_feat_path_I = code_path+'img_trn.mat'
train_feat_path_T = code_path+'txt_trn.mat'
test_feat_path_I = code_path+'img_tst.mat'
test_feat_path_T = code_path+'txt_tst.mat'
if not os.path.exists(code_path):
    os.makedirs(code_path)
train_feat_I = np.zeros([1, bit_length])
train_feat_T = np.zeros([1, bit_length])
test_feat_I = np.zeros([1, bit_length])
test_feat_T = np.zeros([1, bit_length])

saver = tf.train.Saver()
with tf.Session() as sess:
    save_name = "./model/" + dataset_name + "/" + str(bit_length)+ "bit/" + method_name + "/save_" + str(num_epoch) + "_pos.ckpt"
    saver.restore(sess,save_name)
    tst_batch_index = 1
    batch_index = 1
    while batch_index <= my_data_loader.train_batch_numbers:
        i1,t1,l1 = my_data_loader.fetch_train_data()
        tr_code_i,tr_code_t = sess.run([ix_hash,tx_hash], feed_dict={input_ix: i1,input_tx: t1})
        train_feat_I = np.vstack((train_feat_I, tr_code_i))
        train_feat_T = np.vstack((train_feat_T, tr_code_t))
        print(batch_index)
        batch_index += 1
    train_feat_I=train_feat_I[1:,:]
    train_feat_T=train_feat_T[1:,:]
    sio.savemat(train_feat_path_I, {'train_feat': train_feat_I})
    sio.savemat(train_feat_path_T, {'train_feat': train_feat_T})


    while tst_batch_index <= my_data_loader.test_batch_numbers:
        i1,t1,l1  = my_data_loader.fetch_test_data()
        ts_code_i,ts_code_t= sess.run([ix_hash,tx_hash], feed_dict={input_ix: i1,input_tx: t1})
        test_feat_I = np.vstack((test_feat_I, ts_code_i))
        test_feat_T = np.vstack((test_feat_T, ts_code_t))
        print (tst_batch_index)
        tst_batch_index += 1
    test_feat_I=test_feat_I[1:,:]
    test_feat_T=test_feat_T[1:,:]
    sio.savemat(test_feat_path_I, {'test_feat': test_feat_I})
    sio.savemat(test_feat_path_T, {'test_feat': test_feat_T})


