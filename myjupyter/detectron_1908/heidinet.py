# from fine_tune_model import fine_tuned_model, data_generator
# from keras.datasets import cifar10


from keras import backend as K
K.set_image_data_format('channels_last')
from keras.callbacks import ModelCheckpoint


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

print('config.gpu_options.per_process_gpu_memory_fraction:',config.gpu_options.per_process_gpu_memory_fraction)



import os
import sys
import time

import cv2
import numpy as np
from scipy import spatial


import imutils
from itertools import permutations

from imutils.video import WebcamVideoStream
import random
# import matplotlib.pyplot as plt
# %matplotlib inline

# import os
# cwd = os.getcwd()

from os.path import join
import glob
from PIL import Image
# import matplotlib.pyplot as plt



from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.layers import Reshape, Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate, merge, Dropout, Activation
from keras.layers.normalization import BatchNormalization


def generate_triplet(x,y,testsize=0.3,ap_pairs=10,an_pairs=10):
    data_xy = tuple([x,y])

    trainsize = 1-testsize

    triplet_train_pairs = []
    triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        #print(len(same_class_idx),same_class_idx)
        #print(len(list(permutations(same_class_idx,2))))
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        
        #print(len(list(permutations(same_class_idx,2))))
        if ap_pairs > len(list(permutations(same_class_idx,2))):
            tap_pairs = len(list(permutations(same_class_idx,2)))
        else:    
            tap_pairs = ap_pairs
        
        if an_pairs > (len(list(diff_class_idx))):
            tan_pairs = (len(list(diff_class_idx)))
        else:
            tan_pairs = an_pairs
        A_P_pairs = random.sample(list(permutations(same_class_idx,2)),k=tap_pairs) #Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx),k=tan_pairs)
        

        #train
        A_P_len = len(A_P_pairs)
        Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len*trainsize)]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_train_pairs.append([Anchor,Positive,Negative])               
        #test
        for ap in A_P_pairs[int(A_P_len*trainsize):]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_test_pairs.append([Anchor,Positive,Negative])    
                
    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)



def pick_x_with_person_num(num_labels, X_data, Y_data, person_num=5, ls=0):
    #person_num = 5
    #ls = 0
    last_label = len(num_labels)-1
    if ls> last_label-1:
        ls = 0
        
    le = ls+person_num
    
    pick_idx_array = None
    for p_idx in num_labels[ls:le]:
        one_id_idx = np.where(Y_data == p_idx)[0]
        
        if pick_idx_array is None:
            pick_idx_array = one_id_idx
        else:
            pick_idx_array = np.append(pick_idx_array, one_id_idx)
        #print(pick_idx_array.shape)
            
    pick_x = X_data[pick_idx_array]#.shape
    pick_y = Y_data[pick_idx_array]
    
    return pick_x, pick_y


def triplet_loss_v1(y_true, y_pred):
        margin = K.constant(0.2)
        return K.mean(K.maximum(K.constant(0), 
                                K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def triplet_loss_v2(y_true, y_pred, margin_type =  'softplus'):
        margin = K.constant(0.2)

        #loss = K.sum(K.square(y_pred[:,0,0]), axis=1) - K.sum(K.square(y_pred[:,1,0]), axis=1)
        loss = K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0])
        if margin_type == 'maxplus':
            loss = K.maximum(0.0, margin + loss)
        elif margin_type == 'softplus':
            loss = K.log(1 + K.exp(loss))

        return K.mean(loss)
    

def softmax_weights(dist,axis=-1):
    #ndim = K.ndim(dist)
#     ndim=3
#     if ndim == 2:
#         return K.softmax(dist)
#     elif ndim > 2:
    e = K.exp(dist - K.max(dist, axis=axis, keepdims=True))
    s = K.sum(e, axis=axis, keepdims=True)
    return e / (s+ 1e-6)

def weighted_triplet_loss(y_true, y_pred, margin_type =  'maxplus'):
    pos = y_pred[:,0,:]
    neg = y_pred[:,1,:]
    
    m_constant = 0.2#0.2

    
    w1 = softmax_weights(pos)
    w2 = softmax_weights(-neg)
    
    w1 = tf.cast(w1, tf.float32)
    w2 = tf.cast(w2, tf.float32)

    margin = m_constant * (K.square(pos * w1))

    loss_p = (K.square(pos * w1))
    loss_n = (K.square(neg * w2))

    loss = loss_p - loss_n + margin

    if margin_type == 'maxplus':
        L3 = K.maximum(0.0, loss)
    elif margin_type == 'softplus':
        L3 = K.softplus(loss)  # K.log(1 + K.exp(loss+margin)) #

    return K.mean(L3)

def dtw_distance(vects):
    x, y = vects
    return caculate_two_imgs(x,y) 

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def empty_list_mat(n=8,m=8):
    test_list = []
    for i in range(n):
        test_list2 = []
        for j in range(m):
            test_list2.append(None)
        test_list.append(test_list2)
    return test_list


def caculate_two_imgs(a1, a2):
    tensor_shape = a1.shape
    dtw_range = tensor_shape[1]
    dist = []#np.zeros([dtw_range,dtw_range]) #np.zeros((ash[0],ash[1],ash[1]))

    for i in range(dtw_range):
        dist_a = []
        temp_feat1 = a1[:,i,:]
        for j in range(dtw_range):
            temp_feat2 = a2[:,j,:]
            testa =euclidean_distance([temp_feat1,temp_feat2])
            dist_a.append(testa)
        dist.append(dist_a)
    d = dtw2(dist)
    return d


def dtw2(dist_mat):
    m = len(dist_mat)
    n = len(dist_mat[0])
    dist = empty_list_mat(n,m)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i][j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j-1] + dist_mat[i][j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i-1][j] + dist_mat[i][j]
            else:
                dist_min = K.min(K.stack([dist[i - 1][j], dist[i][j - 1]],axis=0),axis=0,keepdims=False)
                dist[i][j] =dist_min+ dist_mat[i][j] 
    return dist[-1][-1]/sum((n,m))



from keras.models import model_from_json

## python 3.5.1
# load json and create model
cwd = '/heidi_net'
json_file = open(os.path.join(cwd,"model_best/mars_train_zip6_200_200_ori_model.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={"caculate_two_imgs": caculate_two_imgs})

# load weights into new model
loaded_model.load_weights(os.path.join(cwd,"model_best/triplet_0424_ori_1_972.h5"))
print("Loaded model from disk")

adam_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
loaded_model.compile(optimizer=adam_optim, loss=weighted_triplet_loss, metrics=[accuracy])  #triplet_loss_v1

## heidi net 분할

## resnet 추출
heidi_resnet_model = Model(input = loaded_model.input[0], output = loaded_model.get_layer('resnet_model').get_output_at(-3))

## triplet 추출
X1 = Input(shape=(128,), name='lambda_input1')
X2 = Input(shape=(128,), name='lambda_input2')
X3 = Input(shape=(128,), name='lambda_input3')

# positive_dist = Lambda(dtw_distance, name='pos_dist')([X1, X2]) 
# negative_dist = Lambda(dtw_distance, name='neg_dist')([X1, X3])
positive_dist = Lambda(euclidean_distance, name='pos_dist')([X1, X2]) 
negative_dist = Lambda(euclidean_distance, name='neg_dist')([X1, X3])
#print(positive_dist.shape)

stacked_dists = Lambda( 
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists')([positive_dist, negative_dist])
#print('stacked_dists:',stacked_dists.shape)

triplet_model = Model(inputs=[X1,X2,X3], outputs=stacked_dists)
