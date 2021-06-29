# s_DeepDTI_trainCNN.py
#
#   A script for trian the convolutional neural network in DeepDTI.
#
#   Source code:
#       https://github.com/qiyuantian/DeepDTI/blob/main/s_DeepDTI_trainCNN.py
#
#   Reference:
#       [1] Tian Q, Bilgic B, Fan Q, Liao C, Ngamsombat C, Hu Y, Witzel T,
#       Setsompop K, Polimeni JR, Huang SY. DeepDTI: High-fidelity
#       six-direction diffusion tensor imaging using deep learning.
#       NeuroImage. 2020;219:117017. 
#
#       [2] Tian Q, Li Z, Fan Q, Ngamsombat C, Hu Y, Liao C, Wang F,
#       Setsompop K, Polimeni JR, Bilgic B, Huang SY. SRDTI: Deep
#       learning-based super-resolution for diffusion tensor MRI. arXiv
#       preprint. 2021; arXiv:2102.09069.
#
# (c) Qiyuan Tian, Harvard, 2021

# %% load modual

import os
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# %% load DnCNN model and utility library

import qtlib as qtlib
from dncnn import dncnn_model

# %% set up root path

dpRoot = os.path.dirname(os.path.abspath('s_DeepDTI_trainCNN.py'));
os.chdir(dpRoot)

# %% load data 

fpData = os.path.join(dpRoot, 'cnn_inout.mat'); 
tmp = sio.loadmat(fpData)
diff_gt = tmp['diff_gt'] * 1.0 # convert uint16 to floating point
diff_input1 = tmp['diff_input1'] * 1.0
diff_input2 = tmp['diff_input2'] * 1.0
diff_input3 = tmp['diff_input3'] * 1.0
diff_input4 = tmp['diff_input4'] * 1.0
diff_input5 = tmp['diff_input5'] * 1.0
mask = tmp['mask'] * 1.0

# concatenate input data
diff_input1 = np.expand_dims(diff_input1, 4) # input in this example do not contain t1w and t2w data, can simply concatenate them with the several diffusion images
diff_input2 = np.expand_dims(diff_input2, 4)
diff_input3 = np.expand_dims(diff_input3, 4)
diff_input4 = np.expand_dims(diff_input4, 4)
diff_input5 = np.expand_dims(diff_input5, 4)   
diff_input = np.concatenate((diff_input1, diff_input2, diff_input3, diff_input4, diff_input5), axis=4)

# %% standardize image intensity

diff_input_stand = np.zeros(diff_input.shape)
diff_output_stand = np.zeros(diff_input.shape)

for ii in np.arange(0, diff_input.shape[-1]):   

    imgs_in = diff_input[:, :, :, :, ii]
    imgs_out = diff_gt

    imgs_in_stand = np.zeros(imgs_in.shape)
    imgs_out_stand = np.zeros(imgs_out.shape)

    for jj in np.arange(0, imgs_in.shape[-1]):  
        
        img = imgs_in[:, :, :, jj];
        imgmean = np.mean(img[np.nonzero(mask)])
        imgstd = np.std(img[np.nonzero(mask)])
        
        imgs_in_stand[:, :, :, jj] = (img - imgmean) / imgstd * mask; # normalize by substracting mean then dividing the std dev of brain voxels in input images
        imgs_out_stand[:, :, :, jj] = (imgs_out[:, :, :, jj] - imgmean) / imgstd * mask;
    
    diff_input_stand[:, :, :, :, ii] = imgs_in_stand;
    diff_output_stand[:, :, :, :, ii] = imgs_out_stand;
  
res_output_stand = diff_output_stand - diff_input_stand; # residual btw input and output

# display standardized b=0 images
plt.imshow(diff_input_stand[:, :, 35, 0, 3], clim=(-2., 2.), cmap='gray')
plt.imshow(diff_output_stand[:, :, 35, 0, 3], clim=(-2., 2.), cmap='gray')
plt.imshow(res_output_stand[:, :, 35, 0, 3], clim=(-1., 1.), cmap='bwr')

# display standardized dwis
plt.imshow(diff_input_stand[:, :, 35, 1, 3], clim=(-2., 2.), cmap='gray')
plt.imshow(diff_output_stand[:, :, 35, 1, 3], clim=(-2., 2.), cmap='gray')
plt.imshow(res_output_stand[:, :, 35, 1, 3], clim=(-1., 1.), cmap='bwr')
    
# %% divide brain volume to blocks

# find indices of smallest block that covers whole brain
tmp = np.nonzero(mask);
xind = tmp[0] * 1.0;
yind = tmp[1] * 1.0;
zind = tmp[2] * 1.0;

xmin = np.min(xind); xmax = np.max(xind);
ymin = np.min(yind); ymax = np.max(yind);
zmin = np.min(zind); zmax = np.max(zind);
ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]; 

# calculate number of blocks along each dimension
sz_block = 64
xlen = xmax - xmin + 1;
ylen = ymax - ymin + 1;
zlen = zmax - zmin + 1;

nx = int(np.ceil(xlen / sz_block));
ny = int(np.ceil(ylen / sz_block));
nz = int(np.ceil(zlen / sz_block));

# determine starting and ending indices of each block
xstart = xmin;
ystart = ymin;
zstart = zmin;

xend = xmax - sz_block + 1;
yend = ymax - sz_block + 1;
zend = zmax - sz_block + 1;

xind_block = np.round(np.linspace(xstart, xend, nx));
yind_block = np.round(np.linspace(ystart, yend, ny));
zind_block = np.round(np.linspace(zstart, zend, nz));

ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
count = 0
for ii in np.arange(0, xind_block.shape[0]):
    for jj in np.arange(0, yind_block.shape[0]):
        for kk in np.arange(0, zind_block.shape[0]):
            ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
            count = count + 1

ind_block = ind_block.astype(int);

# display block indices
print(ind_block[0:6, :])
print(ind_block[6:12, :])

# %% prepare input and output data for CNN

img_block_train = np.zeros(1)
img_block_val = np.zeros(1)

imgres_block_train = np.zeros(1)
imgres_block_val = np.zeros(1)

mask_block_train = np.zeros(1)
mask_block_val = np.zeros(1)

for ii in np.arange(0, diff_input.shape[-1]):    
    
    img = diff_input_stand[:, :, :, :, ii];
    imgres = res_output_stand[:, :, :, :, ii];
    mask_expand = np.expand_dims(mask, 3);        
    
    # extract blocks
    img_block = qtlib.extract_block(img, ind_block);
    imgres_block = qtlib.extract_block(imgres, ind_block);
    mask_block = qtlib.extract_block(mask_expand, ind_block);
    
    imgres_block = np.concatenate((imgres_block, mask_block), axis=-1); # last channel is brain mask, which is used to weigth loss from each voxel
    
    if np.mod(ii, 5) == 0: # use 20% data for validation, should use data from 1 out of 5 subjects for validation in actual implementation 
        print('validation sets')
        if imgres_block_val.any():
            img_block_val = np.concatenate((img_block_val, img_block), axis=0)
            imgres_block_val = np.concatenate((imgres_block_val, imgres_block), axis=0)
            mask_block_val = np.concatenate((mask_block_val, mask_block), axis=0)
        else:
            img_block_val = img_block
            imgres_block_val = imgres_block
            mask_block_val = mask_block
    else:
        print('trainging sets')
        if imgres_block_train.any():
            img_block_train = np.concatenate((img_block_train, img_block), axis=0)
            imgres_block_train = np.concatenate((imgres_block_train, imgres_block), axis=0)
            mask_block_train = np.concatenate((mask_block_train, mask_block), axis=0)
        else:
            img_block_train = img_block
            imgres_block_train = imgres_block
            mask_block_train = mask_block

print(img_block_train.shape)
print(img_block_val.shape)
print(imgres_block_train.shape)
print(imgres_block_val.shape)

# display input and output pair for training 
plt.imshow(img_block_train[0, :, :, 30, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(imgres_block_train[0, :, :, 30, 1], clim=(-1., 1.), cmap='bwr')

# display input and output pair for validation 
plt.imshow(img_block_val[0, :, :, 30, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(imgres_block_val[0, :, :, 30, 1], clim=(-1., 1.), cmap='bwr')

# %% set up DnCNN model

nlayer = 10; 
nfilter = 128;
nin = 7; # should be 9 if t1w and t2w data are included
nout = 7; 

dncnn = dncnn_model(nin, nout, layer_num=nlayer, filter_num=nfilter, kinit_type='he_normal', bnorm_flag=True);
dncnn.summary()

# %% set up adam optimizer

adam_opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0);
dncnn.compile(loss = qtlib.mean_squared_error_weighted, optimizer = adam_opt);
# qtlib.mean_squared_error_weighted is a custom loss function that weights the loss from each voxel
# in this example, only the loss within the brain mask
    
# %% train DnCNN model

nbatch = 1; # should adapte to different datasets and GPUs
nepoch = 100; # should adapte to different datasets, train for 100 epochs in this example

fnCp = 'deepdti_nb1_ep' + np.str(nepoch)
fpCp = os.path.join(dpRoot, fnCp + '.h5') 
checkpoint = ModelCheckpoint(fpCp, monitor='val_loss', save_best_only = True)

history = dncnn.fit(x = [img_block_train, mask_block_train], 
                    y = imgres_block_train, 
                    validation_data = ([img_block_val, mask_block_val], imgres_block_val),
                    batch_size = nbatch, 
                    epochs = nepoch, 
                    callbacks = [checkpoint],
                    verbose = 1, 
                    shuffle = True) 
                    
# save loss
fpLoss = os.path.join(dpRoot, fnCp + '.mat') 
sio.savemat(fpLoss, {'loss_train':history.history['loss'], 'loss_val':history.history['val_loss']})    

# display loss
plt.plot(history.history['loss'],label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.ylim([0, 0.1])

# %% apply DnCNN model

imgres_block_pred = dncnn.predict([img_block_val, mask_block_val]); # predicted residual images
imgres_block_pred = imgres_block_pred[:, :, :, :, :-1] * mask_block_val; # remove last channel
img_block_pred = (img_block_val + imgres_block_pred) * mask_block_val; # denoised images
img_block_gt = img_block_val + imgres_block_val[:, :, :, :, :-1]; # ground-truth images

# display input, output and ground-truth images
plt.imshow(img_block_val[0, :, :, 30, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(img_block_pred[0, :, :, 30, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(img_block_gt[0, :, :, 30, 1], clim=(-2., 2.), cmap='gray')

# display residual images and report mean absolute difference
plt.imshow(imgres_block_val[0, :, :, 30, 1], clim=(-1., 1.), cmap='bwr')
plt.imshow(img_block_gt[0, :, :, 30, 1] - img_block_pred[0, :, :, 30, 1], clim=(-1., 1.), cmap='bwr')

print(np.mean(np.abs(imgres_block_val[0, :, :, 30, 1]))) # mean absolute difference
print(np.mean(np.abs(img_block_gt[0, :, :, 30, 1] - img_block_pred[0, :, :, 30, 1])))

# %% apply DnCNN in actual implementation

# DnCNN can be applied to the whole brain volume data with standardized image intensities
# in order to avoid assembly denoised blocks into whole brain.
# The denoised volume data from DnCNN should be transformed to the normal range by multiplying
# the standard deviation and addin the mean image intensity.
# The finaly denoised results can be then used for fitting diffusion tensor model or other purposes.









