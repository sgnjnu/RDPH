# coding=utf-8
import numpy as np


def spilt_locations(num_data, batch_size):
    perm = np.random.permutation(num_data)
    if num_data % batch_size != 0:
        perm2 = np.random.permutation(num_data)
        perm2 = perm2 - 1
        lack_num = batch_size - (num_data % batch_size)
        perm = np.append(perm, perm2[0:lack_num])
        print (len(perm))
    locations = []

    for ii in range(0, len(perm) - 1):
        if ii % batch_size == 0:
            locations.append(ii)

    locations.append(len(perm))
    return perm, locations


def spilt_locations_non_perm(num_data, batch_size):
    perm = np.arange(num_data)
    """if num_data % batch_size != 0:
        perm2 = np.random.permutation(num_data)
        perm2 = perm2 - 1
        lack_num = batch_size - (num_data % batch_size)
        perm = np.append(perm, perm2[0:lack_num])
        print len(perm)"""
    locations = []

    for ii in range(0, len(perm) - 1):
        if ii % batch_size == 0:
            locations.append(ii)

    locations.append(len(perm))
    return perm, locations


def print_results_new(batch_index,task_flag,epochs,trn_cost,mem_cost,trn_p_acc,trn_acc,m_p_acc,m_acc):
    print("task_" + str(task_flag) + ",Iter/epochs " + str(batch_index) + "/" + str(
        epochs) + "\n loss= " + "{:.5f}".format(
        trn_cost) + ",mem Loss= " + "{:.5f}".format(
        mem_cost)  + ",trn acc= " + "{:.5f}".format(
        trn_acc) + ",trn p acc= " + "{:.5f}".format(
        trn_p_acc) + ",m acc= " + "{:.5f}".format(
        m_acc) + ",m p acc= " + "{:.5f}".format(
        m_p_acc))


def RDPH_triplet_weights(labels1, labels2, labels3):
    """Reads tag labele1, labels2, labels3
    Args:
    tags: labels1, labels2, Labels3 share the same dimension [N x L]
    Returns:
    s12,s13,s23= |l1 n L2|,|l1 n L2|, in [0,L]
    s123=sign(s12-s13), s213=sign(s12-s23), s312=sign(s13-s23)
    """
    n1 = np.sum(np.asarray(labels1,dtype=np.float32), axis=1)
    n2 = np.sum(np.asarray(labels2,dtype=np.float32), axis=1)
    n3 = np.sum(np.asarray(labels3,dtype=np.float32), axis=1)
    max_n=np.maximum(np.maximum(n1,n2),n3)
    n12_ = np.sum(np.asarray(labels1*labels2,
                             dtype=np.float32), axis=1)
    n13_ = np.sum(np.asarray(labels1*labels3,
                             dtype=np.float32), axis=1)
    n23_ = np.sum(np.asarray(labels2*labels3,
                             dtype=np.float32), axis=1)

    z1=idealDCG(n1)
    z2=idealDCG(n2)
    z3=idealDCG(n3)
    sig_123=np.asarray(np.not_equal(n12_,n13_),dtype=np.float32)
    sig_213=np.asarray(np.not_equal(n12_,n23_),dtype=np.float32)
    sig_312=np.asarray(np.not_equal(n13_,n23_),dtype=np.float32)
    sim_123=sig_123*((np.power(2,n12_)-1)/np.maximum(np.log2(max_n-n12_+1),1)-(np.power(2,n13_)-1)/np.maximum(np.log2(max_n-n13_+1),1))/z1

    sim_213=sig_213*((np.power(2,n12_)-1)/np.maximum(np.log2(max_n-n12_+1),1)-(np.power(2,n23_)-1)/np.maximum(np.log2(max_n-n23_+1),1))/z2

    sim_312=sig_312*((np.power(2,n13_)-1)/np.maximum(np.log2(max_n-n13_+1),1)-(np.power(2,n23_)-1)/np.maximum(np.log2(max_n-n23_+1),1))/z3
    s12=cosine_sim(labels1,labels2)
    s13 = cosine_sim(labels1, labels3)
    s23 = cosine_sim(labels2, labels3)
    s12=np.reshape(s12,newshape=[-1,])
    s13=np.reshape(s13,newshape=[-1,])
    s23=np.reshape(s23,newshape=[-1,])
    return sim_123,sim_213,sim_312,s12,s13,s23


def idealDCG(q_label_counts):
    """Reads query_labesl, database_labels
        Args:
        tags: query_labels [N_q, L]
        Returns:
        idealDCG (N_q,1)
    """
    q_idealDCG=np.ones_like(q_label_counts)
    for ii in range(len(q_label_counts)):
        q_idealDCG[ii]=DCG(q_label_counts[ii])
    return  q_idealDCG


def DCG(max_similairty):
    a=0
    for ii in range(np.int(max_similairty)):
       a=a+(np.power(2,max_similairty-ii)-1 )/np.log2(ii+1+1)
    return a


def cosine_sim(c1,c2):
    inner = np.sum(np.multiply(c1, c2), axis=1, keepdims=True)
    c1_norm = np.sqrt(np.sum(np.square(c1), axis=1, keepdims=True))
    c2_norm = np.sqrt(np.sum(np.square(c2), axis=1, keepdims=True))
    return np.divide(inner, np.multiply(c1_norm, c2_norm))
