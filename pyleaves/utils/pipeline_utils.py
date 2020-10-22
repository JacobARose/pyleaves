# @Author: Jacob A Rose
# @Date:   Fri September 18th 2020, 7:30 pm
# @Email:  jacobrose@brown.edu
# @Filename: pipeline_utils.py

"""

Functions designed for use by pyleaves.pipelines.pipeline_simple.py

Mostly refactored versions of functions originally defined in pyleaves.train.paleoai_train

"""
# from pyleaves.datasets import base_dataset
from paleoai_data.dataset_drivers import base_dataset
from paleoai_data.utils.kfold_cross_validation import DataFold
from typing import List, Union, Tuple
import random
import numpy as np
from more_itertools import unzip
from tqdm import trange

from pathlib import Path
from functools import partial
from more_itertools import unzip
import neptune
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pprint import pprint
import pyleaves
from pyleaves.models import resnet, vgg16
# from pyleaves.models import base_model
from pyleaves.base import base_model
from pyleaves.utils import ensure_dir_exists, img_aug_utils as iau
from pyleaves.utils.tf_utils import BalancedAccuracyMetric
from paleoai_data.utils.kfold_cross_validation import DataFold
from tfrecord_utils.encoders import TFRecordCoder

##########################################################################
##########################################################################

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.resnet_v2 import preprocess_input
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LambdaCallback
import wandb

import importlib

def load_img(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def rgb2gray_3channel(img, label):
    '''
    Convert rgb image to grayscale, but keep num_channels=3
    '''
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.grayscale_to_rgb(img)
    return img, label

def rgb2gray_1channel(img, label):
    '''
    Convert rgb image to grayscale, num_channels from 3 to 1
    '''
    img = tf.image.rgb_to_grayscale(img)
    return img, label

def rotate(x, y, seed=None):
    """Rotation augmentation

    Args:
        x,     tf.Tensor: Image
        y,     tf.Tensor: arbitrary tensor, passes through unchanged

    Returns:
        Augmented image, y
    """
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32,seed=seed)), y

def flip(x, y, seed=None):
    """Flip augmentation

    Args:
        x,     tf.Tensor: Image to flip
        y,     tf.Tensor: arbitrary tensor, passes through unchanged
    Returns:
        Augmented image, y
    """
    x = tf.image.random_flip_left_right(x, seed=seed)
    x = tf.image.random_flip_up_down(x, seed=seed)

    return x, y

def color(x, y, seed=None):
    """Color, Saturation, Brightness and Contrast augmentation

    Args:
        x,     tf.Tensor: Image
        y,     tf.Tensor: arbitrary tensor, passes through unchanged

    Returns:
        Augmented image, y
    """
    x = tf.image.random_hue(x, 0.08, seed=seed)
    x = tf.image.random_saturation(x, 0.6, 1.6, seed=seed)
    x = tf.image.random_brightness(x, 0.05, seed=seed)
    x = tf.image.random_contrast(x, 0.7, 1.3, seed=seed)
    return x, y


def sat_bright_con(x, y, seed=None):
    """Saturation, Brightness and Contrast augmentation

    Args:
        x,     tf.Tensor: Image
        y,     tf.Tensor: arbitrary tensor, passes through unchanged

    Returns:
        Augmented image, y
    """
    x = tf.image.random_saturation(x, 0.6, 1.6, seed=seed)
    x = tf.image.random_brightness(x, 0.05, seed=seed)
    x = tf.image.random_contrast(x, 0.7, 1.3, seed=seed)
    return x, y





def _cond_apply(x, y, func, prob, seed=None):
    """Conditionally apply func to x and y with probability prob."""
    return tf.cond((tf.random.uniform([], 0, 1, seed=seed) >= (1.0 - prob)), lambda: func(x,y,seed=seed), lambda: (x,y))

def augment_sample(x, y, prob=1.0, seed=None):
    x, y = _cond_apply(x, y, flip, prob, seed=seed)
    x, y = _cond_apply(x, y, rotate, prob, seed=seed)
    x, y = _cond_apply(x, y, color, prob, seed=seed)
    return x, y

def resize_image(image, shape=(512,512,3), resize_buffer_size=128, training=False, seed=None):
    """Short summary.

    Parameters
    ----------
    image : tf.Tensor
        Tensor with 3 dimensions (h,w,c)
    shape : tuple(int,int,int)
        The desired target size for output images
    resize_buffer_size : int
        Number of pixels of buffer room for _aspect_preserving_resize() to conserve prior to random crop.
        A higher number results in a more diverse set of possible random crops.
    training : bool
        If set to True, first resizes image while preserving aspect ratio, then performs a random crop
        If set to False, deterministically crop image to dimensions described by 'shape'
    seed : int
        Seed for setting RNG

    Returns
    -------
    image : tf.Tensor
        Resized image with shape == 'shape' variable
    """
    
    if training:
        smallest_side = tf.minimum(shape[0], shape[1])
        image = iau._aspect_preserving_resize(image, smallest_side = smallest_side + resize_buffer_size)
        image = tf.image.random_crop(image, shape, seed=seed)
    else:
        image = tf.image.resize_with_pad(image, target_height=shape[0], target_width=shape[1])

    return image

# def apply_preprocess(x, y, num_classes, preprocess_config):
    
#     preprocess_module = importlib.import_module(preprocess_config._target_)

#     preprocess_input = preprocess_module.preprocess_input

#     return hydra.utils.call(preprocess_config, x), tf.one_hot(y, depth=num_classes)
    # return preprocess_input(x), tf.one_hot(y, depth=num_classes)

def get_preprocess_func(from_module: str="tensorflow.keras.applications.imagenet_utils", **kwargs):

    preprocess_module = importlib.import_module(from_module)
    preprocess_input = preprocess_module.preprocess_input
    preprocess_input(tf.zeros([4, 32, 32, 3]))

    # preprocess_input = lambda x: x

    def preprocess_func(x):
        return preprocess_input(x)
    _temp = tf.zeros([4, 32, 32, 3])
    preprocess_func(_temp)

    return preprocess_func


def apply_preprocess(dataset, num_classes, preprocessing_module):
    
    preprocess_input = get_preprocess_func(from_module=preprocessing_module)

    dataset = dataset.map(lambda x,y: (preprocess_input(x), tf.one_hot(y, depth=num_classes)),
                          num_parallel_calls=-1)
    return dataset






def prep_dataset(dataset,
                 preprocess_module: DictConfig,
                 batch_size=32,
                 buffer_size=100,
                 shuffle=False,
                 target_size=(512,512),
                 num_channels=3,
                 color_mode='grayscale',
                 num_classes=10,
                 augmentations=[{}],
                 aug_prob=1.0,
                 training=False,
                 cache_dir: str=None,
                 seed=None):

    
    
    resize = partial(resize_image, shape=(*target_size, num_channels), training=training, seed=seed)
    dataset = dataset.map(lambda x,y: (resize(x), y),
                          num_parallel_calls=-1)

    dataset = apply_preprocess(dataset, num_classes, preprocessing_module=preprocess_module)

    # preprocess_input = partial(apply_preprocess, num_classes=num_classes, preprocess_config=preprocess_config)
    # dataset = dataset.map(lambda x,y: preprocess_input(x,y),
    #                       num_parallel_calls=-1)

    if cache_dir:
        dataset = dataset.cache(cache_dir)


    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    #Note 9/18/2020
    # I just switched the above two steps from the order they've been in
    #Previously:
    # batch->repeat
    #Now:
    # repeat->batch

    for aug in augmentations:
        if 'flip' in aug:
            dataset = dataset.map(lambda x, y: _cond_apply(x, y, flip, prob=aug['flip'], seed=seed), num_parallel_calls=-1)
        if 'rotate' in aug:
            dataset = dataset.map(lambda x, y: _cond_apply(x, y, rotate, prob=aug['rotate'], seed=seed), num_parallel_calls=-1)
        if 'color' in aug:
            dataset = dataset.map(lambda x, y: _cond_apply(x, y, color, prob=aug['color'], seed=seed), num_parallel_calls=-1)

    if color_mode=='grayscale':
        if num_channels==3:
            dataset = dataset.map(lambda x,y: rgb2gray_3channel(x, y), num_parallel_calls=-1)
        elif num_channels==1:
            dataset = dataset.map(lambda x,y: rgb2gray_1channel(x, y), num_parallel_calls=-1)


    dataset = dataset.prefetch(1)
    return dataset




def extract_data(fold: DataFold,
                 encoder: base_dataset.LabelEncoder=None,
                 class_names: List[str]=None,
                 exclude_classes=[],
                 include_classes=[],
                 val_split: float=0.0,
                 subsets=['train','val','test'],
                 seed: int=None):



    train_data, test_data = fold.train_data, fold.test_data
    if encoder is None:
        class_names = class_names or fold.metadata.class_names
        encoder = base_dataset.LabelEncoder(class_names)
        reset_encodings = True
        alphabetize_classes = True
    else:
        reset_encodings = False
        alphabetize_classes = False
    encoder.update_with_new_classes(include_classes=include_classes,
                                    exclude_classes=exclude_classes,
                                    reset_encodings=reset_encodings,
                                    alphabetize_classes=alphabetize_classes)
    # classes = list((set(encoder.classes)-set(exclude_classes)).union(set(include_classes)))

    train_dataset, _ = fold.train_dataset.enforce_class_whitelist(class_names=encoder.classes)
    test_dataset, _ = fold.test_dataset.enforce_class_whitelist(class_names=encoder.classes)

    if 'train+test' in subsets:
        train_dataset = train_dataset + test_dataset

    split_data = {}
    split_datasets = {}

    if val_split > 0.0:

        train_dataset, val_dataset = train_dataset.split(val_split, seed=seed)
        # val_x = [str(p) for p in list(val_dataset.data['path'].values)]
        # val_y = np.array(encoder.encode(val_dataset.data['family']))
        # val_data = (val_x, val_y)
        val_data = val_dataset.get_encoded_data(encoder=encoder, shuffle=False)
        split_data['val'] = val_data
        split_datasets['val'] = val_dataset

    # train_x = [str(p) for p in list(train_dataset.data['path'].values)]
    # train_y = np.array(encoder.encode(train_dataset.data['family']))
    # train_data = list(zip(train_x,train_y))
    # shuffle_idx = random.sample(range(len(train_x)), len(train_x))
    # train_data = [train_data[k] for k in shuffle_idx]
    # train_data = [list(i) for i in unzip(train_data)]

    # np.any(['train' in subset for subset in ['train+test','val','test']])

    if np.any(['train' in subset for subset in subsets]):
        train_data = train_dataset.get_encoded_data(encoder=encoder, shuffle=True)
        split_data['train'] = train_data
        split_datasets['train'] = train_dataset


    # test_x = [str(p) for p in list(test_dataset.data['path'].values)]
    # test_y = np.array(encoder.encode(test_dataset.data['family']))
    # test_data = (test_x, test_y)
    if 'test' in subsets:
        test_data = test_dataset.get_encoded_data(encoder=encoder, shuffle=False)
        split_data['test'] = test_data
        split_datasets['test'] = test_dataset

    return split_data, split_datasets, encoder







def load_data_from_tfrecords(tfrecord_dir: str,
                             data: Tuple[List[str],List[int]]=None,
                             target_shape: Tuple[int]=(768,768,3),
                             samples_per_shard: int=400,
                             subset_key: str='train', #'val',
                             num_classes: int=None,
                             num_parallel_calls: int=-1):

    assert isinstance(data, tuple)

    data = pd.DataFrame({'source_path':data[0],'label':data[1]})

    print(f'Creating TFRecordCoder with samples_per_shard={samples_per_shard}')

    coder = TFRecordCoder(data = data,
                          output_dir = tfrecord_dir,
                          columns={'source_path':'source_path','target_path':'target_path', 'label':'label'},
                          subset=subset_key,
                          target_shape=target_shape,
                          samples_per_shard=samples_per_shard,
                          num_classes=num_classes)

    coder.execute_convert()
    tfrecord_files = [os.path.join(tfrecord_dir,f) for f in os.listdir(tfrecord_dir) if subset_key in f]

    subset_data = tf.data.Dataset.from_tensor_slices(tfrecord_files) \
                                                    .cache() \
                                                    .shuffle(100) \
                                                    .interleave(tf.data.TFRecordDataset) \
                                                    .map(coder.decode_example, num_parallel_calls=num_parallel_calls)
    return subset_data




def load_data_from_tensor_slices(data: List[List], cache: Union[bool,str], training=False, seed=None):

    num_samples = len(data[0])

    def load_img(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    x_data = tf.data.Dataset.from_tensor_slices(data[0])
    y_data = tf.data.Dataset.from_tensor_slices(data[1])
    data = tf.data.Dataset.zip((x_data, y_data))
    if training:
        data = data.shuffle(num_samples,seed=seed, reshuffle_each_iteration=True)

    if cache:
        if isinstance(cache, str):
            data = data.cache(cache)
        else:
            data = data.cache()

    data = data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=tf.uint8),y), num_parallel_calls=-1)

    return data


def extract_and_load_data(data_fold: DataFold,
                          encoder: base_dataset.LabelEncoder=None,
                          class_names: List[str]=None,
                          exclude_classes=[],
                          include_classes=[],
                          val_split=0.0,
                          subsets=['train','val','test'],
                          cache: Union[bool,str]=True,
                          use_tfrecords: bool=True,
                          tfrecord_dir: str=None,
                          samples_per_shard: int=400,
                          seed=None):


    extracted_data, split_datasets, encoder = extract_data(fold=data_fold,
                                                           encoder=encoder,
                                                           class_names=class_names,
                                                           exclude_classes=exclude_classes,
                                                           include_classes=include_classes,
                                                           val_split=val_split,
                                                           subsets=subsets,
                                                           seed=seed)
                                                       
    # subset_keys = [k for k in extracted_data if extracted_data[k] is not None]
    loaded_data = {}
    for subset_key, data in extracted_data.items():
        training = bool('train' in subset_key)
        if use_tfrecords:            
            loaded_data[subset_key] = load_data_from_tfrecords(tfrecord_dir=tfrecord_dir,
                                                               data=data,
                                                               samples_per_shard=samples_per_shard,
                                                               subset_key=subset_key,
                                                               num_classes=len(encoder.classes))
        else:
            loaded_data[subset_key] = load_data_from_tensor_slices(data, cache=cache, training=training, seed=seed)

    return loaded_data, extracted_data, split_datasets, encoder


def create_dataset(data_fold: DataFold,
                   data_config: DictConfig,
                   preprocess_config: DictConfig,
                   encoder: base_dataset.LabelEncoder=None,
                   class_names: List[str]=None,
                   subsets=['train','val','test'],
                   cache: Union[bool,str]=True,
                   cache_image_dir: str=None,
                   use_tfrecords=False,
                   tfrecord_dir:str = '',
                   samples_per_shard: int=300,
                   seed: int=None):

    # dataset, split_datasets, encoder
    loaded_data, extracted_data, split_datasets, encoder = extract_and_load_data(data_fold=data_fold,
                                                             encoder=encoder,
                                                             class_names=class_names,
                                                             exclude_classes=data_config.extract.exclude_classes,
                                                             include_classes=data_config.extract.include_classes,
                                                             val_split=data_config.extract.val_split,
                                                             subsets=subsets,
                                                             use_tfrecords=use_tfrecords,
                                                             tfrecord_dir=tfrecord_dir,
                                                             samples_per_shard=samples_per_shard,
                                                             cache=cache,
                                                             seed=seed)
    num_classes = encoder.num_classes
    split_data = {}
    if np.any(['train' in subset for subset in loaded_data.keys()]):
    # if 'train' in loaded_data.keys():
        # np.any(['train' in subset for subset in ['train+test','val','test']])
        # data_subs = {'train': [0,2,3], 'test':[4,5,6], 'train+test':[7,8,9]}
        keys = np.array(list(loaded_data.keys()))
        keys = keys[['train' in subset for subset in keys]]
        print('keys = ', keys)

        if 'train+test' in subsets:
            subset_key = 'train+test'
            print(f"Proceeding to train on subset concatenation '{subset_key}'")
        elif 'train' in subsets:
            subset_key = 'train'
            print(f"Proceeding to train on subset {subset_key}")


        split_data['train'] = prep_dataset(loaded_data['train'],
                                           preprocess_module=preprocess_config._target_,
                                           batch_size=data_config.training.batch_size,
                                           buffer_size=data_config.training.buffer_size,
                                           shuffle=True,
                                           target_size=data_config.training.target_size,
                                           num_channels=data_config.extract.num_channels,
                                           color_mode=data_config.extract.color_mode,
                                           num_classes=num_classes,
                                           augmentations=data_config.training.augmentations,
                                           training=True,
                                           cache_dir=cache_image_dir,
                                           seed=seed)
    if 'val' in loaded_data.keys():
        print(f"Proceeding to validate with subset 'val', val_split={data_config.extract.val_split}")
        split_data['val'] = prep_dataset(loaded_data['val'],
                                         preprocess_module=preprocess_config._target_,
                                         batch_size=data_config.training.batch_size,
                                         target_size=data_config.training.target_size,
                                         num_channels=data_config.extract.num_channels,
                                         color_mode=data_config.extract.color_mode,
                                         num_classes=num_classes,
                                         training=False,
                                         cache_dir=cache_image_dir,
                                         seed=seed)

    if 'test' in loaded_data.keys():
        print("Proceeding to test with subset 'test'")
        split_data['test'] = prep_dataset(loaded_data['test'],
                                          preprocess_module=preprocess_config._target_,
                                          batch_size=data_config.training.batch_size,
                                          target_size=data_config.training.target_size,
                                          num_channels=data_config.extract.num_channels,
                                          color_mode=data_config.extract.color_mode,
                                          num_classes=num_classes,
                                          training=False,
                                          cache_dir=cache_image_dir,
                                          seed=seed)

    return split_data, extracted_data, split_datasets, encoder



################################################################################################
################################################################################################
## MODELS
################################################################################################
################################################################################################



# def weighted_categorical_crossentropy(weights):
#     """
#     A weighted version of keras.objectives.categorical_crossentropy
    
#     Variables:
#         weights: numpy array of shape (C,) where C is the number of classes
    
#     Usage:
#         weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#         loss = weighted_categorical_crossentropy(weights)
#         model.compile(loss=loss,optimizer='adam')
#     """
    
#     weights = K.variable(weights)
        
#     def loss(y_true, y_pred):
#         # scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         # clip to prevent NaN's and Inf's
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         # calc
#         loss = y_true * K.log(y_pred) * weights
#         loss = -K.sum(loss, -1)
#         return loss
    
#     return loss






def freeze_batchnorm_layers(model, verbose=False):
    layer_nums = []
    for i, layer in enumerate(model.layers):
        if layer.name.endswith('bn'):
            layer_nums.append(i)
            layer.trainable = False
            if verbose: print(f'layer {i}', layer.name)
    return model, layer_nums






def build_base_vgg16_RGB(weights="imagenet", input_shape=(224,224,3), frozen_layers: Tuple[int]=None):

    base = tf.keras.applications.vgg16.VGG16(weights=weights,
                                             include_top=False,
                                             input_tensor=Input(shape=input_shape))

    if (frozen_layers is not None) and (type(frozen_layers) is not str):
        for layer in base.layers[frozen_layers[0]:frozen_layers[1]]:
            layer.trainable = False

    return base


def build_head(base, num_classes=10, head_layers: List[int]=None, pool_size=(2,2), kernel_l2=1e-2):
    if head_layers is None:
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        dense1 = tf.keras.layers.Dense(2048,activation='relu',name='dense1')
        dense2 = tf.keras.layers.Dense(512,activation='relu',name='dense2')
        prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
        model = tf.keras.Sequential([
            base,
            global_average_layer,dense1,dense2,
            prediction_layer
            ])

    else:
        layers = [base] # ToDo try adding all base layers one by one
        # layers = [layer for layer in base.layers] # ToDo try adding all base layers one by one
        layers.append(tf.keras.layers.GlobalAveragePooling2D())#pool_size=pool_size))
        for layer_num, layer_units in enumerate(head_layers):
            layers.append(tf.keras.layers.Dense(layer_units,activation='relu',name=f'dense{layer_num}', kernel_regularizer=tf.keras.regularizers.l2(kernel_l2)))
        
        layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
        model = tf.keras.Sequential(layers)

    return model




def build_lightweight_nets(model_name="mobile_net_v2", weights="imagenet", input_shape=(224,224,3), frozen_layers: Tuple[int]=None):


    if model_name=="mobile_net_v2":
        model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="inception_v3":
        model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="xception":
        model = tf.keras.applications.Xception(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B0":
        model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B1":
        model = tf.keras.applications.EfficientNetB1(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B2":
        model = tf.keras.applications.EfficientNetB2(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B3":
        model = tf.keras.applications.EfficientNetB3(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B4":
        model = tf.keras.applications.EfficientNetB4(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B5":
        model = tf.keras.applications.EfficientNetB5(input_shape=input_shape, include_top=False, weights=weights)


    if type(frozen_layers) is tuple:
        for layer in model.layers[frozen_layers[0]:frozen_layers[1]]:
            layer.trainable = False

    return model




def build_model(model_config, load_saved_model=False):
    '''

    Minimum config:
        model_config = {
                'model_name': None,
                'optimizer':"Adam",
                'num_classes':32,
                'weights': None,
                'frozen_layers':None,
                'input_shape':(224,224,3),
                'lr':1e-5,
                'lr_momentum': None,
                'regularization':None,
                'loss':'categorical_crossentropy',
                'METRICS':['accuracy']
                'head_layers': None,
                }


    Example config:

        model_config = {
                        'model_name': "resnet_50_v2",
                        'optimizer':"Adam",
                        'num_classes':32,
                        'weights': "imagenet",
                        'frozen_layers':(0,-4),
                        'input_shape':(224,224,3),
                        'lr':1e-5,
                        'lr_momentum':0.9,
                        'regularization':{"l2": 1e-4},
                        'loss':'categorical_crossentropy',
                        'METRICS':['accuracy'],
                        'head_layers': [256,128]
                        }
    '''
    if wandb.run.resumed:
        # restore the best model
        model = tf.keras.models.load_model(wandb.restore("model-best.h5").name)
    else:
        if load_saved_model and os.path.isfile(model_config['saved_model_path']):
            print(f"Loading found saved model from {model_config['saved_model_path']}")
            model =  tf.keras.models.load_model(model_config['saved_model_path'])
        else:
            if model_config['model_name']=='vgg16':
                if model_config['num_channels']==1:
                    model_builder = vgg16.VGG16GrayScale(model_config)
                    build_base = model_builder.build_base
                else:
                    build_base = partial(build_base_vgg16_RGB, weights=model_config.weights, input_shape=model_config.input_shape, frozen_layers=model_config.frozen_layers)

            elif model_config['model_name'].startswith('resnet'):
                model_builder = resnet.ResNet(model_config)
                build_base = partial(model_builder.build_base, weights=model_config.weights, input_shape=model_config.input_shape)

            elif model_config['model_name'] in ["mobile_net_v2",
                                                "inception_v3",
                                                "xception"]:#,
                                #   "efficient_net_B0",
                                #   "efficient_net_B1",
                                #   "efficient_net_B2",
                                #   "efficient_net_B3",
                                #   "efficient_net_B4",
                                #   "efficient_net_B5"]:
                build_base = partial(build_lightweight_nets,
                                    model_name=model_config.model_name,
                                    weights=model_config.weights,
                                    input_shape=model_config.input_shape,
                                    frozen_layers=model_config.frozen_layers)

                base = build_base()
                model = build_head(base, num_classes=model_config.num_classes, head_layers=model_config.head_layers, pool_size=model_config.pool_size, kernel_l2=model_config.kernel_l2)


    base = model.layers[0]
    if model_config.frozen_layers=='bn':
        base, frozen_layers = freeze_batchnorm_layers(base, verbose=False)
        model_config.frozen_layers = frozen_layers

    base = base_model.Model.add_regularization(base, **model_config.regularization)
    model = base_model.Model.add_regularization(model, l2=model_config.kernel_l2)
    
    
    # model = base_model.Model.add_regularization(model, **model_config.regularization)

    if model_config.optimizer == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=model_config.lr, momentum=model_config.lr_momentum)#, decay=model_config.lr_decay)
    elif model_config.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=model_config.lr, momentum=model_config.lr_momentum)
    elif model_config.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.lr)

    if model_config.loss=='categorical_crossentropy':
        loss = 'categorical_crossentropy'
    # elif model_config.loss=="weighted_categorical_crossentropy":
    #     loss = weighted_categorical_crossentropy(model_config.class_weights)
    METRICS = []
    if 'f1' in model_config['METRICS']:
        METRICS.append(tfa.metrics.F1Score(num_classes=model_config['num_classes'],
                                           average='weighted',
                                           name='weighted_f1'))
    if 'accuracy' in model_config['METRICS']:
        METRICS.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))
        METRICS.append(tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top-3_accuracy'))
        METRICS.append(tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top-5_accuracy'))
    if 'balanced_accuracy' in model_config['METRICS']:
        METRICS.append(BalancedAccuracyMetric(model_config.num_classes))
    if 'precision' in model_config['METRICS']:
        METRICS.append(tf.keras.metrics.Precision())
    if 'recall' in model_config['METRICS']:
        METRICS.append(tf.keras.metrics.Recall())

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=METRICS)

    return model


def tf_data2np(data: tf.data.Dataset, num_batches: int=4):
    # TODO Move this to within the confusion matrix callback
    x_val, y_val = [], []
    data_iter = iter(data)
    print(f'Loading {num_batches} batches into memory for confusion matrix callback')
    for _ in trange(num_batches):
        x, y = next(data_iter)
        x_val.append(x)
        y_val.append(y)
    return np.vstack(x_val), np.vstack(y_val)



def get_callbacks(config, model_config, model, csv_path: str, train_data=None, val_data=None, encoder=None, experiment=None):
    from neptunecontrib.monitoring.keras import NeptuneMonitor
    # from pyleaves.train.paleoai_train import EarlyStopping, CSVLogger
    from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
    from pyleaves.utils.callback_utils import BackupAndRestore, NeptuneVisualizationCallback,ReduceLROnPlateau,TensorBoard
    from pyleaves.utils.neptune_utils import ImageLoggerCallback

    experiment = experiment or neptune

    # reduce_lr = ReduceLROnPlateau(monitor=config.callbacks.reduce_lr_on_plateau.monitor, factor=0.5,
    #                               patience=config.callbacks.reduce_lr_on_plateau.patience, min_lr=model_config.lr*0.1)
    
    # backup_callback = BackupAndRestore(config.run_dirs.checkpoints_path)
    # backup_callback.set_model(model)

    tensorboard_callback = TensorBoard(log_dir=config.run_dirs.log_dir, profile_batch=0)

    print('building callbacks')
    callbacks = [tensorboard_callback, #backup_callback,               reduce_lr, #           NeptuneMonitor(),
                 CSVLogger(csv_path, separator=',', append=True),
                 EarlyStopping(monitor=config.callbacks.early_stopping.monitor, patience=config.callbacks.early_stopping.patience, min_delta=config.callbacks.early_stopping.min_delta, verbose=1, restore_best_weights=config.callbacks.early_stopping.restore_best_weights)]#True)

                #  EarlyStopping(monitor='val_loss', patience=1, min_delta=0.1, verbose=1, restore_best_weights=False)]


    if config.callbacks.log_images and (train_data is not None):
        callbacks.append(ImageLoggerCallback(data=train_data, 
                                             freq=10, 
                                             max_images=-1,
                                             name='train',
                                             encoder=encoder,
                                             experiment=experiment,
                                             include_predictions=True,
                                             log_epochs=config.callbacks.log_epochs))

    if config.callbacks.log_images and (val_data is not None):
        callbacks.append(ImageLoggerCallback(data=val_data,
                                             freq=10,
                                             max_images=-1,
                                             name='val',
                                             encoder=encoder,
                                             experiment=experiment,
                                             include_predictions=True,
                                             log_epochs=config.callbacks.log_epochs))

    # TODO move the below 2 blocks of code also into neptune visualizer definition
    if config.callbacks.confusion_matrix.log_train and (train_data is not None):
        if config.callbacks.confusion_matrix.num_batches=='all':
            num_batches = config.dataset.params.training.steps_per_epoch
        elif type(config.callbacks.confusion_matrix.num_batches)==int:
            num_batches = config.callbacks.confusion_matrix.num_batches
        else:
            num_batches = 10
            print(f'invalid value for config.callbacks.confusion_matrix.num_batches={config.callbacks.confusion_matrix.num_batches}.\nContinuing with 10 batches')
        # train_data_np = tf_data2np(data=train_data, num_batches=num_batches)
				 
        train_neptune_visualization_callback = NeptuneVisualizationCallback(train_data, #train_data_np,
                                                                            num_classes=model_config.num_classes,
                                                                            text_labels = encoder.classes,
                                                                            steps=num_batches,
                                                                            subset_prefix='train',
                                                                            experiment=experiment)
        callbacks.append(train_neptune_visualization_callback)

    if config.callbacks.confusion_matrix.log_val and (val_data is not None):
        if config.callbacks.confusion_matrix.num_batches=='all':
            num_batches = config.dataset.params.training.validation_steps
        elif type(config.callbacks.confusion_matrix.num_batches)==int:
            num_batches = config.callbacks.confusion_matrix.num_batches
        else:
            num_batches = 10
            print(f'invalid value for config.callbacks.confusion_matrix.num_batches={config.callbacks.confusion_matrix.num_batches}.\nContinuing with 10 batches')
        # validation_data_np = tf_data2np(data=val_data, num_batches=num_batches)
        val_neptune_visualization_callback = NeptuneVisualizationCallback(val_data,
                                                                    num_classes=model_config.num_classes,
                                                                    text_labels = encoder.classes,
                                                                    steps=num_batches,
                                                                    subset_prefix='val',
                                                                    experiment=experiment)
        # val_neptune_visualization_callback = NeptuneVisualizationCallback(validation_data_np, num_classes=model_config.num_classes, experiment=experiment)
        callbacks.append(val_neptune_visualization_callback)

    if config.orchestration.debug:
        print(f'Built callbacks: ')
        pprint(callbacks)
        print('callback_config:')
        pprint(dict(config.callbacks))
        
    return callbacks


################################################################################################
################################################################################################
## MODELS
################################################################################################
################################################################################################





from sklearn.metrics import classification_report#, confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

def evaluate_performance(model, x, y=None, num_samples=None, batch_size=None, labels: List[int]=None, target_names: List[str]=None, output_dict: bool=True):
    steps = int(np.ceil(num_samples/batch_size))
    if y is None:
        # y = x.map(lambda x,y: y).batch(num_samples).take(1)
        y = x.map(lambda x,y: y).take(num_samples).batch(batch_size)
        y = np.vstack([i for i in y])

    x = x.take(num_samples).batch(batch_size)
    probs = model.predict(x, steps=steps, verbose=1)
    print('probs.shape = ', probs.shape)
    y_hat = probs.argmax(axis=1)
    print('y_hat.shape = ', y_hat.shape)
    y = y.argmax(axis=1)
    print('y.shape = ', y.shape)
    # print(f'labels.shape = {labels.shape}, target_names.shape = {target_names.shape}')
    try:
        report = classification_report(y, y_hat, labels=labels, target_names=target_names, output_dict=output_dict)
        # report = classification_report(labels, preds)
        if type(report)==dict:
            report = pd.DataFrame(report)
    except Exception as e:
        import pdb; pdb.set_trace()
        print(e)

    return report

# y = x.map(lambda x,y: y).batch(num_samples).take(1)
# x = x.take(num_samples).batch(batch_size)

# The above resulted in y.shape == 2078
# while
# x.shape == 2048

# Clearly, the batch operation is truncating to multiples of batch_size
# Likely the best option is for both x and y to utilize the same order of operations



# def load_and_eval_model(x, y=None, steps=None, text_labels=None):
    
#     model_path = "/media/data/jacob/simplified-baselines/Leaves_in_PNAS_family_100_resnet_50_v2_[512, 512]/task-9_2020-09-22_23-25-32/model_dir/saved_model"
#     model = tf.keras.models.load_model(model_path)

#     if y is None:
#         y = x.map(lambda x,y: y)

#     probs = model.predict(x, steps=steps)
#     print(probs.shape)
#     y_hat = probs.argmax(axis=1)
#     report = classification_report(y, y_hat, target_names=text_labels)
#     # report = classification_report(labels, preds)

#     return report








