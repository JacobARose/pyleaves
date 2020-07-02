# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: vgg16.py


'''

Model definition for building VGG16 with or without Imagenet weights, as well as functionality for altering pretrained weights from 3 channels to 1.


'''
from pyleaves.models.base_model import add_regularization

import io
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
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
# from pyleaves.models.base_model import BaseModel
from pyleaves.base.base_model import BaseModel
from pyleaves.train.metrics import METRICS


class VGG16(BaseModel):
    '''
    e.g.
    model_params = {'num_classes':32,
                    'frozen_layers':(0,-4),
                    'input_shape':(224,224,3),
                    'base_learning_rate':1e-5,
                    'regularization':{"l1": 1e-4}
                    }
    Example usage:
    '''
    def __init__(self, model_params, name='VGG16'):
        super().__init__(model_params, name=name)

    def build_base(self):

        base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                 include_top=False,
                                                 input_tensor=Input(shape=(224,224,3)))

        if self.frozen_layers is not None:
            for layer in base.layers[self.frozen_layers[0]:self.frozen_layers[1]]:
                layer.trainable = False

        return base

    def build_head(self, base):

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        dense1 = tf.keras.layers.Dense(2048,activation='relu',name='dense1')#, kernel_initializer=tf.initializers.GlorotNormal())
        dense2 = tf.keras.layers.Dense(512,activation='relu',name='dense2')#, kernel_initializer=tf.initializers.GlorotNormal())
        prediction_layer = tf.keras.layers.Dense(self.num_classes,activation='softmax')#, kernel_initializer=tf.initializers.GlorotNormal())
        model = tf.keras.Sequential([
            base,
            global_average_layer,dense1,dense2,
            prediction_layer
            ])

        return model

def get_layers_by_index(model, layer_indices=[2, 5, 9, 13, 17]):
    if type(model.layers[0])==tf.python.keras.engine.training.Model:
        #In case model was constructed from a base model
        model = model.layers[0]
    if layer_indices == 'all':
        layer_indices = list(range(len(model.layers)))
    layers = [l for idx, l in enumerate(model.layers) if idx in layer_indices]
    return layers

def get_conv_block(model, block_number=1):
    layers = get_layers_by_index(model, layer_indices='all')
    block_layers = [l for l in layers if l.name.startswith('block'+str(block_number)) and ('conv' in l.name)]
    return block_layers

def get_vgg16_conv_block_outputs(model, index=True):
    """Get the output tensors from the 5 layers at the end of each convolutional block in vgg16.

    Returns
    -------
    type
        Description of returned object.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    block_out_layer_idx = [2, 5, 9, 13, 17]
    if index:
        outputs = [(i, model.layers[i].output) for i in block_out_layer_idx]
    else:
        outputs = [model.layers[i+1].output for i in block_out_layer_idx]
    return outputs

def is_square_num(num):
    return np.sqrt(num)*np.sqrt(num)==num

@tf.function
def plot_activation(activation_array, filter_size=64):
    assert is_square_num(filter_size)

    rows = cols = int(np.sqrt(filter_size))

    fig, axes = plt.subplots(rows,cols, figsize=(24,24))

    for i in range(filter_size):
        axes[int(i/rows),i%cols].imshow(activation_array[:,:,i])
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    grid_imgs = tf.image.decode_png(buffer.getvalue(), channels=4)
    buffer.close()
    grid_imgs = tf.expand_dims(grid_imgs, 0)
    return grid_imgs

# @tf.function
def visualize_activations(input_images, model,  sess=None, graph=None, group='vgg16_conv_block_outputs'):

    if group=='vgg16_conv_block_outputs':
        outputs = get_vgg16_conv_block_outputs(model, index=True)
        num_outputs = 2#len(outputs)
        filter_size=64
        num_input_images = 3

    output_layers = [o[1] for o in outputs][:3]
    output_layer_indices = [o[0] for o in outputs][:3]
    model = Model(inputs=model.inputs, outputs=output_layers)
    layer_output_predictions = model.predict(input_images, steps=num_input_images)
    print(len(layer_output_predictions))
    print(num_outputs)

    output_grids = []
    layer_names = [l.name for l in get_layers_by_index(model, layer_indices=output_layer_indices)]
    print(f'logging {num_outputs} outputs for {num_input_images} input images')

    # with graph.as_default():
    if True:
        K.set_session(sess)

        for i in range(num_input_images):
            for j in range(num_outputs):
                name = layer_names[j]
                output_grids.append((
                                    name,
                                    plot_activation(sess.run(layer_output_predictions[j][i]), filter_size=filter_size)
                                    ))
    return output_grids





class VGG16GrayScale(VGG16):
    def __init__(self, model_params, name='vgg16_grayscale'):
        super().__init__(model_params, name=name)

    def build_base(self):
        # import tensorflow.keras.backend as K
        # K.clear_session()
        # K.reset_default_graph()
        if os.path.isfile(self.base_model_filepath):
            return self.load(self.base_model_filepath)

        base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                 include_top=False)#,
                                                 # input_tensor=Input(shape=(224,224,3)))

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

        # base_model.save(self.weights_filepath)
        self.save(self.base_model_filepath, model=base_model)

        if self.frozen_layers is not None:
            for layer in base_model.layers[self.frozen_layers[0]:self.frozen_layers[1]]:
                layer.trainable = False

        return base_model
