'''

Model definition for building VGG16 with or without Imagenet weights, as well as functionality for altering pretrained weights from 3 channels to 1.


'''



import numpy as np
import os
import tensorflow as tf
# from pyleaves.utils import set_visible_gpus
# set_visible_gpus([4])

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dropout, Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.keras import datasets, layers, models
import itertools
import matplotlib.pyplot as plt

import pyleaves
from pyleaves.train.metrics import METRICS


class VGG16:
    '''
    Example usage:
    
    '''
    
    def __init__(self, model_config):

        self.config = model_config
        
        self.num_classes = model_config.num_classes
        self.frozen_layers = model_config.frozen_layers
        self.input_shape = model_config.input_shape
        self.base_learning_rate = model_config.base_learning_rate
        self.regularization = model_config.regularization
        
    def build_base(self):
        
        if weights=='imagenet':
            input_shape=(224,224,3)
        else:
            input_shape=self.input_shape
            
        base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                        include_top=False,
                                                        input_tensor=Input(shape=input_shape))
        
        for layer in base.layers[self.frozen_layers[0]:self.frozen_layers[1]]:
            layer.trainable = False
            
        return base
    
    def build_head(self, base):
        
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        dense1 = tf.keras.layers.Dense(2048,activation='relu',name='dense1')
        dense2 = tf.keras.layers.Dense(512,activation='relu',name='dense2')
        prediction_layer = tf.keras.layers.Dense(self.num_classes,activation='softmax')
        model = tf.keras.Sequential([
            base,
            global_average_layer,dense1,dense2,
            prediction_layer
            ])
        
        return model
    
    def add_regularization(self, model):

        if self.regularization is not None:
            if 'l2' in self.regularization:
                regularizer = tf.keras.regularizers.l2(self.regularization['l2'])
            elif 'l1' in self.regularization:
                regularizer = tf.keras.regularizers.l1(self.regularization['l1'])
        
        model = pyleaves.models.base_model.add_regularization(model, regularizer)
        
        return model
    
    def build_model(self):
        
        base = self.build_base()
        
        model = self.build_head(base)
            
        model = self.add_regularization(model)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.base_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=METRICS)        
        
        return model
    
    
    
    
class VGG16Grayscale(VGG16):
    def __init__(self, model_config):
        
        if 'weights_dir' in model_config:
            self.weights_dir = model_config.weights_dir
        else:
            self.weights_dir = os.path.expanduser(r'~/.keras/models/')
        self.weights_path = os.path.join(self.weights_dir,'vgg16_grayscale.h5')
        
        super().__init__(model_config)
        
            
    def build_base(self):
        
        base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                 include_top=False,
                                                 input_tensor=Input(shape=(224,224,3)))

        # Convert Block1_conv1 weights 
        # From shape[3, 3, 3, 64] -> this is for RGB images
        # To shape [3, 3, 1, 64]. Weighted average of the features has to be calculated across channels.
        # RGB weights: Red 0.2989, Green 0.5870, Blue 0.1140

        block1_conv1 = base.get_layer('block1_conv1').get_weights()
        weights, biases = block1_conv1

        # :weights shape = [3, 3, 3, 64] - (0, 1, 2, 3)
        # convert :weights shape to = [64, 3, 3, 3] - (3, 2, 0, 1)
        weights = np.transpose(weights, (3, 2, 0, 1))

        kernel_out_channels, kernel_in_channels, kernel_rows, kernel_columns = weights.shape
        
        # initialize 1 channel weights
        grayscale_weights = np.zeros((kernel_out_channels, 1, kernel_rows, kernel_columns))

        for i in range(kernel_out_channels):

            # get kernel for every out_channel
            get_kernel = weights[i, :, :, :]
            temp_kernel = np.zeros((3, 3))

            # :get_kernel shape = [3, 3, 3]
            # axis, dims = (0, in_channel), (1, row), (2, col)

            # calculate weighted average across channel axis
            in_channels, in_rows, in_columns = get_kernel.shape
            for in_row in range(in_rows):
                for in_col in range(in_columns):
                    feature_red = get_kernel[0, in_row, in_col]
                    feature_green = get_kernel[1, in_row, in_col]
                    feature_blue = get_kernel[2, in_row, in_col]
                    # weighted average for RGB filter
                    total = (feature_red * 0.2989) + (feature_green * 0.5870) + (feature_blue * 0.1140)

                    temp_kernel[in_row, in_col] = total

            temp_kernel = np.expand_dims(temp_kernel, axis=0)
            # Now, :temp_kernel shape is [1, 3, 3]
            grayscale_weights[i, :, :, :] = temp_kernel

        # Dimension of :grayscale_weights is [64, 1, 3, 3]

        # dimension, axis of :grayscale_weights = (out_channels: 0), (in_channels: 1), (rows: 2), (columns: 3)
        # tf format of weights = (rows: 0), (columns: 1), (in_channels: 2), (out_channels: 3)

        # Go from (0, 1, 2, 3) to (2, 3, 1, 0)
        grayscale_weights = np.transpose(grayscale_weights, (2, 3, 1, 0)) # (3, 3, 1, 64)

        # combine :grayscale_weights and :biases
        new_block1_conv1 = [grayscale_weights, biases]


        # Reconstruct the layers of VGG16 but replace block1_conv1 weights with :grayscale_weights

        # get weights of all the layers starting from 'block1_conv2'
        vgg16_weights = {}
        for layer in base.layers[2:]:
            if "conv" in layer.name:
                vgg16_weights[layer.name] = base.get_layer(layer.name).get_weights()

        del base

        # Custom build VGG16
        input_layer = Input(shape=(self.input_shape[0],self.input_shape[1], 1), name='input')
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 1), data_format="channels_last", name='block1_conv1')(input_layer)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((8, 8), strides=(8, 8), name='block5_pool')(x)

        base_model = Model(inputs=input_layer, outputs=x)

        base_model.get_layer('block1_conv1').set_weights(new_block1_conv1)
        for layer in base_model.layers[2:]:
            if 'conv' in layer.name:
                base_model.get_layer(layer.name).set_weights(vgg16_weights[layer.name])

#         base_model.summary()
        base_model.save(self.weights_path)
    
        return base_model
        
        
        
        
        