# qtlib.py
#
# Utiliti functions for training CNN in DeepDTI.
#
# (c) Qiyuan Tian, Harvard, 2021

import numpy as np
import keras.backend as K

def extract_block(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
    
    return blocks

def mean_squared_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights # use last channel from grouth-truth data to weight loss from each voxel from first n-1 channels
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)

def mean_absolute_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    
    return K.mean(K.abs(y_pred_weighted - y_true_weighted), axis=-1)