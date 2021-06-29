# dncnn.py
#
# DnCNN model from Zhang et al., 2016 for denoising diffusion MRI data in DeepDTI from Tian et al., 2020. 
# 
# (c) Qiyuan Tian, Harvard, 2021

from keras.models import Model
from keras.layers import Input, Conv3D, BatchNormalization, Activation, concatenate
from keras.utils import plot_model

def dncnn_model(input_ch, output_ch=1, layer_num=20, filter_num=64, bnorm_flag=True, kinit_type='he_normal', tag='dncnn'):

    inputs = Input((None, None, None, input_ch))    
    inputs_weight = Input((None, None, None, 1))    
    
    recon = dncnn_inout(inputs, 
                        inputs_weight, 
                        output_ch=output_ch, 
                        layer_num=layer_num, 
                        filter_num=filter_num, 
                        kinit_type=kinit_type, 
                        tag=tag)

    model = Model(inputs=[inputs, inputs_weight], outputs=[recon])
    plot_model(model, to_file='%s.png' % tag, show_shapes=True)

    return model


def dncnn_inout(inputs, inputs_weight, output_ch=1, layer_num=20, filter_num=64, bnorm_flag=True, kinit_type='he_normal', tag='dncnn'):
    
    conv = inputs
    
    # input layer
    layer_count = 1
    conv = Conv3D(filter_num, (3,3,3), padding='same', 
                  activation='relu', 
                  kernel_initializer=kinit_type, 
                  name = 'conv'+str(layer_count))(conv)
    
    # 2nd to N-1 layers
    for ii in range(layer_num-2):
        layer_count += 1      
        
        if bnorm_flag:
            conv = Conv3D(filter_num, (3,3,3), padding='same', 
                          kernel_initializer=kinit_type, 
                          name = 'conv'+str(layer_count))(conv)
            conv = BatchNormalization()(conv)        
            conv = Activation('relu', name = 'relu'+str(layer_count))(conv)
        else:
            conv = Conv3D(filter_num, (3,3,3), padding='same', 
                          activation='relu', 
                          kernel_initializer=kinit_type, 
                          name = 'conv'+str(layer_count))(conv)
            
    # output layer
    layer_count += 1
    num_ch = output_ch
    conv = Conv3D(num_ch, (3,3,3), padding='same', 
                  kernel_initializer=kinit_type, 
                  name = 'conv'+str(layer_count))(conv)

    # concat layer
    # also add in an additional channel such that for only using loss within brain mask
    # the last channel is used in custom loss function to weight loss from each voxel to only use loss within brain mask
    conv = concatenate([conv, inputs_weight])
    return conv



















