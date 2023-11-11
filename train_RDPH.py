# utf-8
import tensorflow._api.v2.compat.v1 as tf
import scipy.io as sio
import numpy as np
import readdatatools as rt
from DataLoader import MyDataLoader
import os

tf.disable_v2_behavior()

feature_dim = 512
bit_length = 16
batch_size = 128
num_epoch = 200
display_step = 20
bit = '%d' % bit_length
HIDDEN_COUNT = 512
HIDDEN_COUNT1 = 512
method_name = 'RDPH'
dataset_name = 'mir'

if dataset_name == 'mir':
    n_classes = 24
    feature_dim1 = 512
elif dataset_name == 'coco':
    n_classes = 80
    feature_dim1 = 512
elif dataset_name == 'nus':
    n_classes = 10
    feature_dim1 = 512
dropout = True

lambda1 = 20   # hyper-parameter lambda_1 in paper
lambda2 = 0.000001 # hyper-parameter lambda_2 in paper
lambda3 = 0.00001 # hyper-parameter lambda_3 in paper
lam = 1
alpha = 3.2 # hyper-parameter alpha in paper
c_margin = 0.05 # hyper-parameter m in paper

ood_nums = 64 # hyper-parameter M in paper

input_ix = tf.placeholder(tf.float32, [None, feature_dim])
input_io = tf.placeholder(tf.float32, [None, feature_dim])
input_to = tf.placeholder(tf.float32, [None, feature_dim1])
input_tx = tf.placeholder(tf.float32, [None, feature_dim1])
input_iy = tf.placeholder(tf.float32, [None, feature_dim])
input_ty = tf.placeholder(tf.float32, [None, feature_dim1])
input_iz = tf.placeholder(tf.float32, [None, feature_dim])
input_tz = tf.placeholder(tf.float32, [None, feature_dim1])
input_ixB = tf.placeholder(tf.float32, [None, bit_length])
input_txB = tf.placeholder(tf.float32, [None, bit_length])
input_lxB = tf.placeholder(tf.float32, [None, bit_length])
input_iyB = tf.placeholder(tf.float32, [None, bit_length])
input_tyB = tf.placeholder(tf.float32, [None, bit_length])
input_lyB = tf.placeholder(tf.float32, [None, bit_length])
input_izB = tf.placeholder(tf.float32, [None, bit_length])
input_tzB = tf.placeholder(tf.float32, [None, bit_length])
input_lzB = tf.placeholder(tf.float32, [None, bit_length])

pre_x = tf.placeholder(tf.float32, [None, n_classes])
pre_y = tf.placeholder(tf.float32, [None, n_classes])
pre_z = tf.placeholder(tf.float32, [None, n_classes])

pos_weight = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

Sxyz_ = tf.placeholder(tf.float32, [None, ])
Sxyz = tf.reshape(Sxyz_, [-1, 1])
Syxz_ = tf.placeholder(tf.float32, [None, ])
Syxz = tf.reshape(Syxz_, [-1, 1])
Szxy_ = tf.placeholder(tf.float32, [None, ])
Szxy = tf.reshape(Szxy_, [-1, 1])
Sxy_ = tf.placeholder(tf.float32, [None, ])
Sxy = tf.reshape(Sxy_, [-1, 1])
Sxz_ = tf.placeholder(tf.float32, [None, ])
Sxz = tf.reshape(Sxz_, [-1, 1])
Syz_ = tf.placeholder(tf.float32, [None, ])
Syz = tf.reshape(Syz_, [-1, 1])

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

fh_var = [t_fc1w, t_fc1b, t_fc2w, t_fc3w, i_fc1w, i_fc1b, i_fc2w, i_fc3w, l_fc1w, l_fc2w, l_fc1b]
dis_var_list = [f_dit_fc1w, f_dit_fc1b, h_dit_fc1w, h_dit_fc1b, f_dil_fc1w, f_dil_fc1b, h_dil_fc1w, h_dil_fc1b,
                f_dtl_fc1w, f_dtl_fc1b, h_dtl_fc1w, h_dtl_fc1b]


def label_hash(x_label):
    fc1l = tf.nn.bias_add(tf.matmul(x_label, l_fc1w), l_fc1b)
    fc1 = tf.nn.tanh(fc1l)
    fc2l = tf.matmul(fc1, l_fc2w)
    fc2 = tf.nn.tanh(fc2l)
    fc3l = tf.matmul(fc2, l_fc3w)
    hash = tf.nn.tanh(fc3l)
    return fc1, hash


def image_hash(x_image):
    fc1l = tf.nn.bias_add(tf.matmul(x_image, i_fc1w), i_fc1b)
    fc1 = tf.nn.tanh(fc1l)
    fc2l = tf.matmul(fc1, i_fc2w)
    fc2 = tf.nn.tanh(fc2l)
    fc3l = tf.matmul(fc2, i_fc3w)
    hash = tf.nn.tanh(fc3l)
    return fc2, hash


def text_hash(x_text):
    fc1l = tf.nn.bias_add(tf.matmul(x_text, t_fc1w), t_fc1b)
    fc1 = tf.nn.tanh(fc1l)
    fc2l = tf.matmul(fc1, t_fc2w)
    fc2 = tf.nn.tanh(fc2l)
    fc3l = tf.matmul(fc2, t_fc3w)
    hash = tf.nn.tanh(fc3l)
    return fc2, hash


def cosine(code1, code2):
    inner = tf.reduce_sum(tf.multiply(code1, code2), reduction_indices=1, keep_dims=True)
    c1_norm = tf.sqrt(tf.reduce_sum(tf.square(code1), reduction_indices=1, keep_dims=True))
    c2_norm = tf.sqrt(tf.reduce_sum(tf.square(code2), reduction_indices=1, keep_dims=True))
    return tf.divide(inner, tf.multiply(c1_norm, c2_norm) + 1e-6)


def feature_discriminators(input_feature, f_d_fc1w, f_d_fc1b):
    cls_out = tf.nn.bias_add(tf.matmul(input_feature, f_d_fc1w), f_d_fc1b)
    cls_prob = tf.nn.sigmoid(cls_out)
    return cls_prob


def hash_discriminators(hash_feature, h_d_fc1w, h_d_fc1b):
    cls_out = tf.nn.bias_add(tf.matmul(hash_feature, h_d_fc1w), h_d_fc1b)
    cls_prob = tf.nn.sigmoid(cls_out)
    return cls_prob


def dis_loss(p_pro, n_pro):
    loss = -tf.reduce_mean(tf.log(1e-6 + p_pro) + tf.log(1e-6 + 1 - n_pro))
    return loss


def quantization_loss(like_hash, code):
    q_square_loss = tf.reduce_sum(tf.square(like_hash - code), reduction_indices=1, keep_dims=True)
    return tf.reduce_mean(q_square_loss)



# NXN pairs
def cosine_sim(code1, code2):
    inner = tf.matmul(code1, code2, transpose_b=True)
    c1_norm = tf.sqrt(tf.reduce_sum(tf.square(code1), reduction_indices=1, keep_dims=True))
    c2_norm = tf.sqrt(tf.reduce_sum(tf.square(code2), reduction_indices=1, keep_dims=True))
    return tf.divide(inner, tf.matmul(c1_norm, c2_norm, transpose_b=True) + 1e-6)


def sim2(code1, code2):
    inner = tf.matmul(code1, code2, transpose_b=True)
    c1_len = tf.reduce_sum(code1, reduction_indices=1, keep_dims=True)
    z = tf.maximum(tf.log(c1_len - inner + 1), 1.0)
    return (tf.pow(alpha, inner)) / z


def concat_ID_OOD(id, ood):
    return tf.concat([id, ood], 0)


def semantic_consistency_loss(ci1, ci2, ci3, ct1, ct2, ct3, if1, if2, if3, tf1, tf2, tf3, lx, ly, lz, iof,tof, ioc, toc):
    # open-data label is zero vectors
    lo = tf.zeros(shape=[ood_nums, n_classes], dtype=tf.float32)

    lxo = concat_ID_OOD(lx, lo)
    lyo = concat_ID_OOD(ly, lo)
    lzo = concat_ID_OOD(lz, lo)

    if1o = concat_ID_OOD(if1, iof)
    if2o = concat_ID_OOD(if2, iof)
    if3o = concat_ID_OOD(if3, iof)

    tf1o = concat_ID_OOD(tf1, tof)
    tf2o = concat_ID_OOD(tf2, tof)
    tf3o = concat_ID_OOD(tf3, tof)

    ci1o = concat_ID_OOD(ci1, ioc)
    ci2o = concat_ID_OOD(ci2, ioc)
    ci3o = concat_ID_OOD(ci3, ioc)

    ct1o = concat_ID_OOD(ct1, toc)
    ct2o = concat_ID_OOD(ct2, toc)
    ct3o = concat_ID_OOD(ct3, toc)

    # intra-modal feature similarity

    Sx_yo_ii = cosine_sim(if1, if2o)
    Sx_zo_ii = cosine_sim(if1, if3o)
    Sy_zo_ii = cosine_sim(if2, if3o)
    Sy_xo_ii = cosine_sim(if2, if1o)
    Sz_xo_ii = cosine_sim(if3, if1o)
    Sz_yo_ii = cosine_sim(if3, if2o)

    Sx_yo_tt = cosine_sim(tf1, tf2o)
    Sx_zo_tt = cosine_sim(tf1, tf3o)
    Sy_zo_tt = cosine_sim(tf2, tf3o)
    Sy_xo_tt = cosine_sim(tf2, tf1o)
    Sz_xo_tt = cosine_sim(tf3, tf1o)
    Sz_yo_tt = cosine_sim(tf3, tf2o)

    # inter-modal label semantic similarity
    Sx_yo = sim2(lx, lyo)
    Sx_zo = sim2(lx, lzo)
    Sy_zo = sim2(ly, lzo)
    Sy_xo = sim2(ly, lxo)
    Sz_xo = sim2(lz, lxo)
    Sz_yo = sim2(lz, lyo)

    # inter-modal ranking loss by semantic
    Dx_yo_it=cosine_sim(ci1,ct2o)
    Dx_yo_ti=cosine_sim(ct1,ci2o)
    Dx_zo_it=cosine_sim(ci1,ct3o)
    Dx_zo_ti=cosine_sim(ct1,ci3o)
    Dy_zo_it=cosine_sim(ci2,ct3o)
    Dy_zo_ti=cosine_sim(ct2,ci3o)
    Dy_xo_it=cosine_sim(ci2,ct1o)
    Dy_xo_ti=cosine_sim(ct2,ci1o)
    Dz_xo_it=cosine_sim(ci3,ct1o)
    Dz_xo_ti=cosine_sim(ct3,ci1o)
    Dz_yo_it=cosine_sim(ci3,ct2o)
    Dz_yo_ti=cosine_sim(ct3,ci2o)



    l_it_xyo=semantic_consistency_sub_loss(Dx_yo_it,Sx_yo)
    l_ti_xyo=semantic_consistency_sub_loss(Dx_yo_ti,Sx_yo)
    l_it_xzo=semantic_consistency_sub_loss(Dx_zo_it,Sx_zo)
    l_ti_xzo=semantic_consistency_sub_loss(Dx_zo_ti,Sx_zo)
    l_it_yzo=semantic_consistency_sub_loss(Dy_zo_it,Sy_zo)
    l_ti_yzo=semantic_consistency_sub_loss(Dy_zo_ti,Sy_zo)

    l_it_yxo=semantic_consistency_sub_loss(Dy_xo_it,Sy_xo)
    l_ti_yxo=semantic_consistency_sub_loss(Dy_xo_ti,Sy_xo)
    l_it_zxo=semantic_consistency_sub_loss(Dz_xo_it,Sz_xo)
    l_ti_zxo=semantic_consistency_sub_loss(Dz_xo_ti,Sz_xo)
    l_it_zyo=semantic_consistency_sub_loss(Dz_yo_it,Sz_yo)
    l_ti_zyo=semantic_consistency_sub_loss(Dz_yo_ti,Sz_yo)

    l_inter_semantic_ranking=l_it_xyo+l_ti_xyo+l_it_xzo+l_ti_xzo+l_it_yzo+l_ti_yzo+l_it_yxo+l_ti_yxo+l_it_zxo+l_ti_zxo+l_it_zyo+l_ti_zyo

    # intra-modal ranking loss by feature
    Dx_yo_ii = cosine_sim(ci1, ci2o)
    Dx_yo_tt = cosine_sim(ct1, ct2o)
    Dx_zo_ii = cosine_sim(ci1, ci3o)
    Dx_zo_tt = cosine_sim(ct1, ct3o)
    Dy_zo_ii = cosine_sim(ci2, ci3o)
    Dy_zo_tt = cosine_sim(ct2, ct3o)
    Dy_xo_ii = cosine_sim(ci2, ci1o)
    Dy_xo_tt = cosine_sim(ct2, ct1o)
    Dz_xo_ii = cosine_sim(ci3, ci1o)
    Dz_xo_tt = cosine_sim(ct3, ct1o)
    Dz_yo_ii = cosine_sim(ci3, ci2o)
    Dz_yo_tt = cosine_sim(ct3, ct2o)

    l_ii_xyo = semantic_consistency_sub_loss_open(Dx_yo_ii, Sx_yo_ii)
    l_ii_xzo = semantic_consistency_sub_loss_open(Dx_zo_ii, Sx_zo_ii)
    l_ii_yzo = semantic_consistency_sub_loss_open(Dy_zo_ii, Sy_zo_ii)
    l_tt_xyo = semantic_consistency_sub_loss_open(Dx_yo_tt, Sx_yo_tt)
    l_tt_xzo = semantic_consistency_sub_loss_open(Dx_zo_tt, Sx_zo_tt)
    l_tt_yzo = semantic_consistency_sub_loss_open(Dy_zo_tt, Sy_zo_tt)

    l_ii_yxo = semantic_consistency_sub_loss_open(Dy_xo_ii, Sy_xo_ii)
    l_ii_zxo = semantic_consistency_sub_loss_open(Dz_xo_ii, Sz_xo_ii)
    l_ii_zyo = semantic_consistency_sub_loss_open(Dz_yo_ii, Sz_yo_ii)
    l_tt_yxo = semantic_consistency_sub_loss_open(Dy_xo_tt, Sy_xo_tt)
    l_tt_zxo = semantic_consistency_sub_loss_open(Dz_xo_tt, Sz_xo_tt)
    l_tt_zyo = semantic_consistency_sub_loss_open(Dz_yo_tt, Sz_yo_tt)

    l_intra = l_ii_xyo + l_ii_xzo + l_ii_yzo + l_tt_xyo + l_tt_xzo + l_tt_yzo + \
              l_ii_yxo + l_ii_zxo + l_ii_zyo + l_tt_yxo + l_tt_zxo + l_tt_zyo

    l_inter_consist=semantic_consistency_sub_loss3(Dx_yo_it,Dx_yo_ti)+semantic_consistency_sub_loss3(Dx_zo_it,Dx_zo_ti)+semantic_consistency_sub_loss3(Dy_zo_it,Dy_zo_ti)\
    + semantic_consistency_sub_loss3(Dy_xo_it,Dy_xo_ti)+semantic_consistency_sub_loss3(Dz_xo_it,Dz_xo_ti)+semantic_consistency_sub_loss3(Dz_yo_it,Dz_yo_ti)

    return l_inter_semantic_ranking, l_inter_consist, l_intra


def semantic_consistency_sub_loss(c12, s12):
    p = tf.nn.softmax(s12)
    q = tf.nn.softmax(c12 - c_margin)
    return Js_divergerence(p, q)


def semantic_consistency_sub_loss3(c12, s12):
    p = tf.nn.softmax(s12 - c_margin)
    q = tf.nn.softmax(c12 - c_margin)
    return Js_divergerence(p, q)


def semantic_consistency_sub_loss_open(c12, s12):
    p = tf.nn.softmax(s12)
    q = tf.nn.softmax(c12)
    return Js_divergerence(p, q)


def Js_divergerence(pijk, qijk):
    per_loss = pijk * tf.log(2 * pijk / (pijk + qijk) + 1e-7) + qijk * tf.log(2 * qijk / (qijk + pijk) + 1e-7)
    return tf.reduce_mean(tf.reduce_sum(per_loss, reduction_indices=1, keep_dims=True))


def adv_loss(f1, f2, h1, h2, f_d_fc1w, f_d_fc1b, h_d_fc1w, h_d_fc1b):
    f1_pro = feature_discriminators(f1, f_d_fc1w, f_d_fc1b)
    f2_pro = feature_discriminators(f2, f_d_fc1w, f_d_fc1b)
    h1_pro = hash_discriminators(h1, h_d_fc1w, h_d_fc1b)
    h2_pro = hash_discriminators(h2, h_d_fc1w, h_d_fc1b)
    return dis_loss(f1_pro, f2_pro) + dis_loss(h1_pro, h2_pro)


lxf, lx_hash = label_hash(pre_x)
lyf, ly_hash = label_hash(pre_y)
lzf, lz_hash = label_hash(pre_z)
ixf, ix_hash = image_hash(input_ix)
iyf, iy_hash = image_hash(input_iy)
izf, iz_hash = image_hash(input_iz)
iof, io_hash = image_hash(input_io)
txf, tx_hash = text_hash(input_tx)
tyf, ty_hash = text_hash(input_ty)
tzf, tz_hash = text_hash(input_tz)
tof, to_hash = text_hash(input_to)

# dis for I-T
adv_itx_loss = adv_loss(ixf, txf, ix_hash, tx_hash, f_dit_fc1w, f_dit_fc1b, h_dit_fc1w, h_dit_fc1b)
adv_ity_loss = adv_loss(iyf, tyf, iy_hash, ty_hash, f_dit_fc1w, f_dit_fc1b, h_dit_fc1w, h_dit_fc1b)
adv_itz_loss = adv_loss(izf, tzf, iz_hash, tz_hash, f_dit_fc1w, f_dit_fc1b, h_dit_fc1w, h_dit_fc1b)
# dis for I-L
adv_ilx_loss = adv_loss(ixf, lxf, ix_hash, lx_hash, f_dil_fc1w, f_dil_fc1b, h_dil_fc1w, h_dil_fc1b)
adv_ily_loss = adv_loss(iyf, lyf, iy_hash, ly_hash, f_dil_fc1w, f_dil_fc1b, h_dil_fc1w, h_dil_fc1b)
adv_ilz_loss = adv_loss(izf, lzf, iz_hash, lz_hash, f_dil_fc1w, f_dil_fc1b, h_dil_fc1w, h_dil_fc1b)
# dis for T-L
adv_tlx_loss = adv_loss(txf, lxf, tx_hash, lx_hash, f_dtl_fc1w, f_dtl_fc1b, h_dtl_fc1w, h_dtl_fc1b)
adv_tly_loss = adv_loss(tyf, lyf, ty_hash, ly_hash, f_dtl_fc1w, f_dtl_fc1b, h_dtl_fc1w, h_dtl_fc1b)
adv_tlz_loss = adv_loss(tzf, lzf, tz_hash, lz_hash, f_dtl_fc1w, f_dtl_fc1b, h_dtl_fc1w, h_dtl_fc1b)

Lsc, Lssc, l_INTRA = semantic_consistency_loss(ix_hash, iy_hash, iz_hash, tx_hash, ty_hash, tz_hash, input_ix, input_iy,
                                               input_iz, input_tx, input_ty, input_tz, pre_x, pre_y, pre_z, input_io,
                                               input_to,io_hash, to_hash)
Lqan = quantization_loss(ix_hash, input_ixB) + quantization_loss(iy_hash, input_iyB) + quantization_loss(iz_hash,
                                                                                                         input_izB) \
       + quantization_loss(tx_hash, input_txB) + quantization_loss(ty_hash, input_tyB) + quantization_loss(tz_hash,
                                                                                                           input_tzB) \
       + (quantization_loss(lx_hash, input_lxB) + quantization_loss(ly_hash, input_lyB) + quantization_loss(lz_hash,
                                                                                                            input_lzB))
Ladv = adv_itx_loss + adv_ity_loss + adv_itz_loss + adv_ilx_loss + adv_ily_loss + adv_ilz_loss + adv_tlx_loss + adv_tly_loss + adv_tlz_loss

Lgen = 10 * (Lsc) + lambda3 * Lqan + 10 * Lssc + lambda1 * l_INTRA
Ltolfh = Lgen - lambda2 * Ladv

learning_rate = 0.001
opt_dis = tf.train.AdamOptimizer(learning_rate).minimize(Ladv, var_list=dis_var_list)
opt_fh = tf.train.AdamOptimizer(learning_rate).minimize(Ltolfh, var_list=fh_var)
my_opts = [opt_dis, opt_fh]

bit = '%d' % bit_length
log_dir = "./model/" + dataset_name + "/" + bit + "bit/" + method_name
train_writer = tf.summary.FileWriter(log_dir + '/train')
test_writer = tf.summary.FileWriter(log_dir + '/test')


def coding_process(session, data_loader, Wi, Wt, Wl, is_first=True):
    code_i = np.zeros([1, bit_length])
    code_t = np.zeros([1, bit_length])
    code_l = np.zeros([1, bit_length])
    batch_index = 1
    data_num = data_loader.train_data_num
    perm, locations = rt.spilt_locations_non_perm(data_num, 400)
    while batch_index <= (len(locations) - 1):
        data_indexs = perm[locations[batch_index - 1]:locations[batch_index]]
        batch_is = data_loader.image_traindata[data_indexs, :]
        batch_ts = data_loader.text_traindata[data_indexs, :]
        batch_ls = data_loader.train_label_list[data_indexs, :]
        i_code, t_code, l_code = session.run([ix_hash, tx_hash, lx_hash],
                                             feed_dict={input_ix: batch_is, input_tx: batch_ts, pre_x: batch_ls,
                                                        keep_prob: 1.0})
        code_i = np.vstack((code_i, i_code))
        code_t = np.vstack((code_t, t_code))
        code_l = np.vstack((code_l, l_code))
        batch_index += 1
    code_i = code_i[1:, :]
    code_t = code_t[1:, :]
    code_l = code_l[1:, :]
    if is_first:
        iB = np.sign(code_i)
        tB = np.sign(code_t)
        lB = np.sign(code_l)
    else:
        l_i_B = np.matmul(data_loader.train_label_list, Wi)
        l_t_B = np.matmul(data_loader.train_label_list, Wt)
        l_l_B = np.matmul(data_loader.train_label_list, Wl)
        iB = half_coding(learning_rate * code_i + l_i_B)
        tB = half_coding(learning_rate * code_t + l_t_B)
        lB = half_coding(learning_rate * code_l + l_l_B)

    return iB, tB, lB


def half_coding(x):
    median = np.median(x, axis=0)
    same_sign = 2 * np.asarray(x >= median, dtype=np.float32) - 1.0
    return same_sign


def getW(B, data_loader, N, lm):
    L = data_loader.train_label_list
    A = np.matmul(np.transpose(L), L)
    noise = np.eye(N)
    nW = np.matmul(np.matmul(np.linalg.inv(A + lm * noise), np.transpose(L)), B)
    return nW


def train(epochs, session, opts, data_loader, trn_step, Bi, Bt, Bl):
    train_batch_numbers = data_loader.train_batch_numbers
    data_loader.shuffle_train_data()

    a_train_step = trn_step
    # opt hash
    batch_index = 1
    while batch_index <= train_batch_numbers:
        ix, tx, lx, iy, ty, ly, iz, tz, lz = data_loader.fetch_train_triplets()
        ood_tdata = np.random.uniform(-1.0, 1.0, size=(ood_nums, feature_dim))
        ood_idata = 0.5*ix[0:ood_nums,:]+0.5*ix[(batch_size-ood_nums):,:]
        iBx, iBy, iBz = data_loader.get_xyz_B(Bi)
        tBx, tBy, tBz = data_loader.get_xyz_B(Bt)
        lBx, lBy, lBz = data_loader.get_xyz_B(Bl)
        s123, s213, s312, s12, s13, s23 = rt.RDPH_triplet_weights(lx, ly, lz)
        for opt in opts:
            session.run(opt,
                        feed_dict={input_ix: ix, input_iy: iy, input_iz: iz, input_tx: tx, input_ty: ty, input_tz: tz,
                                   pre_x: lx, pre_y: ly, pre_z: lz, input_ixB: iBx, input_iyB: iBy, input_izB: iBz,
                                   input_txB: tBx,
                                   input_tyB: tBy, input_tzB: tBz, input_lxB: lBx, input_lyB: lBy, input_lzB: lBz,
                                   Sxyz_: s123, Syxz_: s213, Szxy_: s312
                            , Sxy_: s12, Sxz_: s13, Syz_: s23, input_to: ood_tdata,input_io: ood_idata})

        if batch_index % display_step == 0:
            trn_cost, adv_cost, Ltr_ = session.run([Lgen, Ladv, Lsc],
                                                   feed_dict={input_ix: ix, input_iy: iy, input_iz: iz, input_tx: tx,
                                                              input_ty: ty, input_tz: tz,
                                                              pre_x: lx, pre_y: ly, pre_z: lz, input_ixB: iBx,
                                                              input_iyB: iBy, input_izB: iBz, input_txB: tBx,
                                                              input_tyB: tBy, input_tzB: tBz, input_lxB: lBx,
                                                              input_lyB: lBy, input_lzB: lBz, Sxyz_: s123, Syxz_: s213,
                                                              Szxy_: s312
                                                       , Sxy_: s12, Sxz_: s13, Syz_: s23, input_to: ood_tdata,input_io: ood_idata})
            rt.print_results_new(batch_index, 0, epochs, trn_cost, Ltr_, 0, 0, 0, 0)

        batch_index += 1
    if epochs % 100 == 0:
        ee = '%d' % epochs
        bit = '%d' % bit_length
        save_name = "./model/" + dataset_name + "/" + bit + "bit/" + method_name + "/save_" + ee + "_pos.ckpt"
        saver.save(session, save_name)
        print("-------Save Finished!---------")
    return a_train_step


'''prepare data'''


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


init = tf.initialize_all_variables()
my_data_loader = MyDataLoader(dataset_name, batch_size)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    train_step = 1
    epochs = 1
    iB, tB, lB = coding_process(sess, my_data_loader, 0, 0, 0, is_first=True)
    tic()
    while epochs <= num_epoch:

        a_train_step = train(epochs, sess, my_opts, my_data_loader, train_step, iB, tB, lB)
        train_step = a_train_step
        if epochs % 2 == 0:
            Win = getW(iB, my_data_loader, n_classes, lam)
            Wtn = getW(tB, my_data_loader, n_classes, lam)
            Wln = getW(lB, my_data_loader, n_classes, lam)
            iB, tB, lB = coding_process(sess, my_data_loader, Win, Wtn, Wln, is_first=False)
        epochs += 1
    toc()
    code_path = './hash_code/' + dataset_name + '/' + method_name + '/' + str(bit_length) + 'bit/'
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    train_feat_path_I = code_path + 'img_trn.mat'
    train_feat_path_T = code_path + 'txt_trn.mat'
    sio.savemat(train_feat_path_I, {'train_feat': iB})
    sio.savemat(train_feat_path_T, {'train_feat': tB})

train_writer.close()
test_writer.close()
