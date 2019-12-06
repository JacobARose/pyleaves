'''
Model definitions for tensorflow.keras model architectures
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import  Dropout, Input
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
import itertools
import matplotlib.pyplot as plt


def vgg16_base(input_shape=(224,224,3), frozen_layers=(0,-4)):
	vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', 
						include_top=False, input_tensor=Input(shape=input_shape))
	for layer in vgg16_model.layers[frozen_layers[0]:frozen_layers[1]]:
		layer.trainable = False
	return vgg16_model
def xception_base(classes=10000,frozen_layers=(0,-4)): 
	xception = tf.keras.applications.xception.Xception(include_top=False,
	                   weights='imagenet', input_tensor=None, 
	                   input_shape=None, pooling=None, num_classes=num_classes)
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
		