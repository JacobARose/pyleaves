'''

Model definition for building VGG16 with or without Imagenet weights, as well as functionality for altering pretrained weights from 3 channels to 1.


'''


from collections import OrderedDict
import numpy as np
import os
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# from pyleaves.utils import set_visible_gpus
# set_visible_gpus([4])

from tensorflow.keras.applications import resnet, resnet_v2
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dropout, Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.keras import datasets, layers, models
import itertools
import matplotlib.pyplot as plt

import pyleaves
from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig, ModelConfig
from pyleaves.models.base_model import BaseModel
from pyleaves.train.metrics import METRICS


class ResNet(BaseModel):
    '''
    Example usage:
    
    '''
    
    _MODELS = {
              'resnet_50':resnet.ResNet50,
              'resnet_101':resnet.ResNet101,
              'resnet_152':resnet.ResNet152,
              'resnet_50_v2':resnet_v2.ResNet50V2,
              'resnet_101_v2':resnet_v2.ResNet101V2,
              'resnet_152_v2':resnet_v2.ResNet152V2
             }
    

    
    def __init__(self, model_config, name=None):
        assert model_config.model_name in self._MODELS
        
        if name is None:
            name = model_config.model_name
        
        self.base_model = self._MODELS[model_config.model_name]
        
        super().__init__(model_config, name=name)
        
    def get_weights(self, model):
        weights = OrderedDict()
        for layer in model.layers:
            w = layer.get_weights()
            if len(w)==2:
                print(layer.name, w[0].shape, w[1].shape)
                weights.update({layer.name:w})
        return weights
            
    def build_base(self):
            
        base = self.base_model(weights='imagenet',
                               include_top=False,
                               input_tensor=Input(shape=(224,224,3)))

        if self.frozen_layers is not None:
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

#     def get_config(self, model):
#         config = [] # OrderedDict()
        
#         for i, layer in enumerate(model.layers):
#             config.append(
#                           {
#                            'layer_name':layer.name,
#                            'layer_contents':
#                                             {
#                                              'layer_number': i,
#                                              'layer_function': type(layer),
#                                              'layer_config': layer.get_config()
#                                             }
#                           }
#                          )

#         return config
    
    
#     def build_model(self):
        
#         base = self.build_base()
        
#         model = self.build_head(base)
            
#         model = self.add_regularization(model)
#         model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.base_learning_rate),
#                       loss='categorical_crossentropy',
#                       metrics=METRICS)        
        
#         return model
    

    
# config = ExperimentConfig()

# config = ModelConfig(model_name='resnet_50_v2')

# res_model = ResNet(config)


# model = res_model.build_base()

# weights = res_model.get_weights(model)

# res_model_config = res_model.get_config(model)

# configs[:5] = res_model_config

class ResNetGrayScale(ResNet):
    def __init__(self, model_config, name=None):
        
        print('WARNING: ResNetGrayScale Not yet implemented')
        return None
    
#         if name is None:
#             name = model_config.model_name+'_grayscale'
#         super().__init__(model_config)
        
            
    def build_base(self):
        
        base = self.base_model(weights='imagenet',
                               include_top=False,
                               input_tensor=Input(shape=(224,224,3)))

        # Convert Block1_conv1 weights 
        # From shape[7, 7, 3, 64] -> this is for RGB images
        # To shape [7, 7, 1, 64]. Weighted average of the features has to be calculated across channels.
        # RGB weights: Red 0.2989, Green 0.5870, Blue 0.1140

        conv1_conv = base.get_layer('conv1_conv').get_weights()
        weights, biases = conv1_conv

        # :weights shape = [7, 7, 3, 64] - (0, 1, 2, 3)
        # convert :weights shape to = [64, 3, 7, 7] - (3, 2, 0, 1)
        weights = np.transpose(weights, (3, 2, 0, 1))

        kernel_out_channels, kernel_in_channels, kernel_rows, kernel_columns = weights.shape
        
        # initialize 1 channel weights
        grayscale_weights = np.zeros((kernel_out_channels, 1, kernel_rows, kernel_columns))

        for i in range(kernel_out_channels):
            # calculate weighted average across channel axis
            
            get_kernel = weights[i, :, :, :]
            
            in_channels, in_rows, in_columns = get_kernel.shape
            temp_kernel = np.zeros((in_rows, in_columns))
            # :get_kernel shape = [3, 7, 7]
            # axis, dims = (0, in_channel), (1, row), (2, col)
            
            
            for in_row in range(in_rows):
                for in_col in range(in_columns):
                    feature_red = get_kernel[0, in_row, in_col]
                    feature_green = get_kernel[1, in_row, in_col]
                    feature_blue = get_kernel[2, in_row, in_col]
                    # weighted average for RGB filter
                    total = (feature_red * 0.2989) + (feature_green * 0.5870) + (feature_blue * 0.1140)

                    temp_kernel[in_row, in_col] = total

            temp_kernel = np.expand_dims(temp_kernel, axis=0)
            # Now, :temp_kernel shape is [1, 7, 7]
            grayscale_weights[i, :, :, :] = temp_kernel

        # Dimension of :grayscale_weights is [64, 1, 7, 7]

        # dimension, axis of :grayscale_weights = (out_channels: 0), (in_channels: 1), (rows: 2), (columns: 3)
        # tf format of weights = (rows: 0), (columns: 1), (in_channels: 2), (out_channels: 3)

        # Go from (0, 1, 2, 3) to (2, 3, 1, 0)
        grayscale_weights = np.transpose(grayscale_weights, (2, 3, 1, 0)) # (7, 7, 1, 64)

        # combine :grayscale_weights and :biases
        new_conv1_conv = [grayscale_weights, biases]

        # Reconstruct the layers of ResNet but replace block1_conv weights with :grayscale_weights

        # get weights of all the layers starting from 'block1_conv2'
        resnet_weights = {}
        for layer in base.layers[2:]:
            if "conv" in layer.name:
                resnet_weights[layer.name] = layer.get_weights()
#                 resnet_weights[layer.name] = base.get_layer(layer.name).get_weights()
        resnet_config = self.get_config(base)
    
        del base

        
        # TBD 3/4/20
        
        # Custom build ResNet
#         input_layer = Input(shape=(self.input_shape[0],self.input_shape[1], 1), name='input')
        
#         l1_config = resnet_config[1]
#         call_layer = l1_config['layer_contents']['layer_function']
#         layer_config = l1_config['layer_contents']['layer_config']        
#         x = [call_layer(**layer_config)(input_layer)]
#         # Block 1
#         i=0
#         for l in resnet_config[2:]:
#             print(l['layer_contents'])
#             call_layer = l['layer_contents']['layer_function']
#             layer_config = l['layer_contents']['layer_config']
#             print('layer ', i)
#             i+=1
#             x = call_layer(**layer_config)(x)
            
#         base_model = Model(inputs=input_layer, outputs=x)

#         base_model.get_layer('conv1_conv').set_weights(new_conv1_conv)
#         for layer in base_model.layers[2:]:
#             if 'conv' in layer.name:
#                 base_model.get_layer(layer.name).set_weights(vgg16_weights[layer.name])

# #         base_model.summary()
#         base_model.save(self.weights_path)
    
#         return base_model
        
        
        
        
        