'''
Model definitions for tensorflow.keras model architectures
'''

import numpy as np
import os
import tensorflow as tf
# from pyleaves.utils import set_visible_gpus
# set_visible_gpus([4])
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import  Dropout, Input
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
import itertools
import matplotlib.pyplot as plt

from pyleaves.train.metrics import METRICS
from pyleaves.models.base_model import add_regularization

def vgg16_base(input_shape=(224,224,3), frozen_layers=(0,-4)):
    print('printing vgg16 base with input_shape=',input_shape)
    vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                        include_top=False, input_tensor=Input(shape=input_shape))
    for layer in vgg16_model.layers[frozen_layers[0]:frozen_layers[1]]:
        layer.trainable = False
    return vgg16_model
def xception_base(num_classes=10000,frozen_layers=(0,-4)):
    xception = tf.keras.applications.xception.Xception(include_top=False,
                       weights='imagenet', input_tensor=None,
                       input_shape=None, pooling=None, classes=num_classes)
    for layer in xception.layers[frozen_layers[0]:frozen_layers[1]]:
        layer.trainable = False
    return xception

def resnet_50_v2_base(num_classes=10000,frozen_layers=(0,-4)):
    model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False,
                       weights='imagenet', input_tensor=None,
                       input_shape=None, pooling=None, classes=num_classes)
    for layer in model.layers[frozen_layers[0]:frozen_layers[1]]:
        layer.trainable = False
    return model

def resnet_101_v2_base(num_classes=10000,frozen_layers=(0,-4)):
    model= tf.keras.applications.resnet_v2.ResNet101V2(include_top=False,
                       weights='imagenet', input_tensor=None,
                       input_shape=None, pooling=None, classes=num_classes)
    for layer in model.layers[frozen_layers[0]:frozen_layers[1]]:
        layer.trainable = False
    return model

def shallow(input_shape=(224,224,3)):

    model = models.Sequential()
    model.add(layers.Conv2D(64, (7, 7), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (7, 7), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64*2, activation='relu'))
    return model


def build_model(model_name='shallow',
                num_classes=10000,
                frozen_layers=(0,-4),
                input_shape=(224,224,3),
                base_learning_rate=0.0001,
                regularization=None,
                **kwargs):

    if 'name' in kwargs:
        print("keyword 'name' is deprecated in function build_model(), please use 'model_name' instead.")
        return None
    
    print('building model: ',name)
    
    if name == 'shallow':
        base = shallow(input_shape)
    elif name == 'vgg16':
        base = vgg16_base(input_shape, frozen_layers)
    elif name == 'xception':
        base = xception_base(num_classes, frozen_layers)
    elif name == 'resnet_50_v2':
        base = resnet_50_v2_base(num_classes, frozen_layers)
    elif name == 'resnet_101_v2':
        base = resnet_101_v2_base(num_classes, frozen_layers)

    if name != 'shallow':
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#         conv1 = tf.keras.layers.Dense(2048,activation='relu')
#         conv2 = tf.keras.layers.Dense(512,activation='relu')
#         prediction_layer = tf.keras.layers.Dense(num_classes,activation='softmax')
#         model = tf.keras.Sequential([
#             base,
#             global_average_layer,conv1,conv2,
#             prediction_layer
#             ])
        conv = tf.keras.layers.Dense(1024,activation='relu')
        prediction_layer = tf.keras.layers.Dense(num_classes,activation='softmax')
        model = tf.keras.Sequential([
            base,
            global_average_layer,conv,
            prediction_layer
            ])        
        
    else:
        prediction_layer = tf.keras.layers.Dense(num_classes,activation='softmax')
        model = tf.keras.Sequential([
            base,
            prediction_layer
            ])

        
    if regularization is not None:
        if 'l2' in regularization:
            regularizer = tf.keras.regularizers.l2(regularization['l2'])
            model = add_regularization(model, regularizer)
        
        
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=METRICS)

    return model
