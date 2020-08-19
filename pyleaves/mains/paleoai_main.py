# @Author: Jacob A Rose
# @Date:   Tue August 18th 2020, 11:53 pm
# @Email:  jacobrose@brown.edu
# @Filename: paleoai_main.py


'''
Script built off of configurable_train_pipeline.py


python '/home/jacob/projects/pyleaves/pyleaves/mains/paleoai_main.py'

'''

from collections import OrderedDict
from stuf import stuf
import copy
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
from pyleaves.utils.callback_utils import BackupAndRestore
from pprint import pprint
from pathlib import Path

from pyleaves.utils import setGPU, set_tf_config

from pyleaves.datasets import leaves_dataset, fossil_dataset, pnas_dataset, base_dataset
from pyleaves.datasets.base_dataset import BaseDataset
import neptune
import arrow
from pyleaves.utils import ensure_dir_exists, img_aug_utils as iau
from tfrecord_utils.encoders import TFRecordCoder
from more_itertools import unzip
from functools import partial
import pyleaves
import hydra
from omegaconf import DictConfig, OmegaConf

from paleoai_data.utils.kfold_cross_validation import DataFold
from paleoai_data.utils.kfold_cross_validation import generate_KFoldDataset, export_folds_to_csv, KFoldLoader #, prep_dataset

CONFIG_DIR = str(Path(pyleaves.RESOURCES_DIR,'..','..','configs','hydra'))
##########################################################################
##########################################################################

date_format = '%Y-%m-%d_%H-%M-%S'



import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from pyleaves.models import resnet, vgg16








def initialize_experiment(cfg, experiment_start_time=None):

    # if 'stage_1' in cfg.pipeline:
    #     for stage in cfg.pipeline:
    #         cfg.experiment.experiment_name = '_'.join([config.dataset.dataset_name, config.model.model_name for config in ])
    # else:

    cfg_0 = cfg.stage_0
    cfg.experiment.experiment_name = '_'.join([cfg_0.dataset.dataset_name, cfg_0.model.model_name])
    cfg.experiment.experiment_dir = os.path.join(cfg.experiment.neptune_experiment_dir, cfg.experiment.experiment_name)

    cfg.experiment.experiment_start_time = experiment_start_time or datetime.now().strftime(date_format)
    cfg.update(log_dir = os.path.join(cfg.experiment.experiment_dir, 'log_dir__'+cfg.experiment.experiment_start_time))
    cfg.update(model_dir = os.path.join(cfg.log_dir,'model_dir'))
    cfg.stage_0.update(tfrecord_dir = os.path.join(cfg.log_dir,'tfrecord_dir'))
    cfg.update(tfrecord_dir = cfg.stage_0.tfrecord_dir)
    cfg.saved_model_path = str(Path(cfg.model_dir) / Path('saved_model'))
    cfg.checkpoints_path = str(Path(cfg.model_dir) / Path('checkpoints'))
    for k,v in cfg.items():
        if '_dir' in k:
            ensure_dir_exists(v)

def restore_or_initialize_experiment(cfg, restore_last=False, prefix='log_dir__', verbose=0):
#     date_format = '%Y-%m-%d_%H-%M-%S'
    cfg = copy.deepcopy(cfg)
    cfg.experiment.experiment_name = '_'.join([cfg.stage_0.dataset.dataset_name, cfg.stage_0.model.model_name])
    cfg.experiment.experiment_dir = os.path.join(cfg.experiment.neptune_experiment_dir, cfg.experiment.experiment_name)
    ensure_dir_exists(cfg.experiment.experiment_dir)

    if restore_last:
        experiment_files = [(exp_name.split(prefix)[-1], exp_name) for exp_name in os.listdir(cfg.experiment.experiment_dir)]
        keep_files = []
        for i in range(len(experiment_files)):
            exp = experiment_files[i]
            try:
                keep_files.append((datetime.strptime(exp[0], date_format), exp[1]))
                if verbose >= 1: print(f'Found previous experiment {exp[1]}')
            except ValueError:
                if verbose >=2: print(f'skipping invalid file {exp[1]}')
                pass

        experiment_files = sorted(keep_files, key= lambda exp: exp[0])
        if type(experiment_files)==list and len(experiment_files)>0:
            experiment_file = experiment_files[-1]
            cfg.experiment.experiment_start_time = experiment_file[0].strftime(date_format)
            initialize_experiment(cfg, experiment_start_time=cfg.experiment.experiment_start_time)
            if verbose >= 1: print(f'Continuing experiment with start time =', cfg.experiment.experiment_start_time)
            return cfg
        else:
            print('No previous experiment in',cfg.experiment.experiment_dir, 'with prefix',prefix)

    cfg.experiment.experiment_start_time = datetime.now().strftime(date_format)
    initialize_experiment(cfg, experiment_start_time=cfg.experiment.experiment_start_time)
    if verbose >= 1: print('Initializing new experiment at time:', cfg.experiment.experiment_start_time )
    return cfg



def log_data(logs):
    for k, v in logs.items():
        neptune.log_metric(k, v)




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
    """Color augmentation

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
    tf.print(type(shape), shape, type(tf.cast(shape[0], dtype=tf.int32)), tf.cast(shape[0], dtype=tf.int32))

    
    if training:
        smallest_side = tf.minimum(shape[0], shape[1])
        image = iau._aspect_preserving_resize(image, smallest_side = smallest_side + resize_buffer_size)
        image = tf.image.random_crop(image, shape, seed=seed)
    else:
        image = tf.image.resize_with_pad(image, target_height=shape[0], target_width=shape[1])

    return image
# shape = (tf.cast(shape[0], dtype=tf.int32), tf.cast(shape[1], dtype=tf.int32))
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
preprocess_input(tf.zeros([4, 224, 224, 3]))
def apply_preprocess(x, y, num_classes=10):
    return preprocess_input(x), tf.one_hot(y, depth=num_classes)


def prep_dataset(dataset,
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
                 seed=None):

    print(type(target_size),target_size)
    
    resize = partial(resize_image, shape=(*target_size, num_channels), training=training, seed=seed)
    dataset = dataset.map(lambda x,y: (resize(x), y),
                          num_parallel_calls=-1)

    dataset = dataset.map(lambda x,y: apply_preprocess(x,y,num_classes),
                          num_parallel_calls=-1)

    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()

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


def partition_data(data, partitions=OrderedDict({'train':0.5,'test':0.5})):
    '''
    Split data into named partitions by fraction

    Example:
    --------
    >> split_data = partition_data(data, partitions=OrderedDict({'train':0.4,'val':0.1,'test':0.5}))
    '''
    num_rows = len(data)
    output={}
    taken = 0.0
    for k,v in partitions.items():
        idx = (int(taken*num_rows),int((taken+v)*num_rows))
        output.update({k:data[idx[0]:idx[1]]})
        taken+=v
    assert taken <= 1.0
    return output

def decode_example(serialized_example):
    feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'path': tf.io.FixedLenFeature([], tf.string, default_value='')
                            }
    features = tf.io.parse_single_example(serialized_example,features=feature_description)

    img = tf.image.decode_jpeg(features['image'], channels=3)
    label = tf.cast(features['label'], tf.int32)
    return img, label
# img = tf.image.convert_image_dtype(load_img(img)*255.0,dtype=tf.uint8)
# img = tf.image.convert_image_dtype(img, dtype=tf.float32)



def initialize_data_from_leavesdb(dataset_name='PNAS',
                                  splits={'train':0.7,'validation':0.3},
                                  threshold=50,
                                  exclude_classes=[],
                                  include_classes=[]):
    datasets = {
            'PNAS': pnas_dataset.PNASDataset(),
            'Leaves': leaves_dataset.LeavesDataset(),
            'Fossil': fossil_dataset.FossilDataset()
            }
    data_files = datasets[dataset_name]

    data_files.exclude_rare_classes(threshold=threshold)
    encoder = base_dataset.LabelEncoder(data_files.classes)
    classes = list((set(encoder.classes)-set(exclude_classes)).union(set(include_classes)))
    data_files, excluded_data_files = data_files.enforce_class_whitelist(class_names=classes)

    x = list(data_files.data['path'].values)
    y = np.array(encoder.encode(data_files.data['family']))

    shuffled_data = list(zip(x,y))
    random.shuffle(shuffled_data)
    partitioned_data = partition_data(data=shuffled_data,
                                      partitions=OrderedDict(splits))
    split_data = {k:v for k,v in partitioned_data.items() if len(v)>0}

    for subset, subset_data in split_data.items():
        split_data[subset] = [list(i) for i in unzip(subset_data)]

    return split_data, data_files, excluded_data_files, encoder



def initialize_data_from_paleoai(fold: DataFold,
                                 exclude_classes=[],
                                 include_classes=[]):

    train_data, test_data = fold.train_data, fold.test_data

    encoder = base_dataset.LabelEncoder(fold.metadata.class_names)
    classes = list((set(encoder.classes)-set(exclude_classes)).union(set(include_classes)))
    # data_files, excluded_data_files = fold.full_dataset.enforce_class_whitelist(class_names=classes)
    train_dataset, _ = fold.train_dataset.enforce_class_whitelist(class_names=classes)
    test_dataset, _ = fold.test_dataset.enforce_class_whitelist(class_names=classes)

    train_x = [str(p) for p in list(train_dataset.data['path'].values)]
    train_y = np.array(encoder.encode(train_dataset.data['family']))

    test_x = [str(p) for p in list(test_dataset.data['path'].values)]
    test_y = np.array(encoder.encode(test_dataset.data['family']))

    train_data = list(zip(train_x,train_y))
    random.shuffle(train_data)
    train_data = [list(i) for i in unzip(train_data)]
    test_data = (test_x, test_y)

    split_data = {'train':train_data, 'test':test_data}

    return split_data, train_dataset, test_dataset, encoder

    



def load_data_from_tfrecords(tfrecord_dir,
                             data=None,
                             target_shape=(768,768,3),
                             samples_per_shard=800,
                             subset_keys=['train','validation'],
                             num_classes=None):

    if data:
        for k,v in data.items():
            data[k] = pd.DataFrame({'source_path':v[0],'label':v[1]})

        train_coder = TFRecordCoder(data = data[subset_keys[0]],
                                    output_dir = tfrecord_dir,
                                    subset=subset_keys[0],
                                    target_shape=target_shape,
                                    samples_per_shard=samples_per_shard,
                                    num_classes=num_classes)


        val_coder = TFRecordCoder(data = data[subset_keys[1]],
                                  output_dir = tfrecord_dir,
                                  subset=subset_keys[1],
                                  target_shape=target_shape,
                                  samples_per_shard=samples_per_shard,
                                  num_classes=num_classes)
        train_coder.execute_convert()
        val_coder.execute_convert()

    files = {subset_keys[0]: [os.path.join(tfrecord_dir,f) for f in os.listdir(tfrecord_dir) if subset_keys[0] in f],
             subset_keys[1]: [os.path.join(tfrecord_dir,f) for f in os.listdir(tfrecord_dir) if subset_keys[1] in f]}

    # import pdb;pdb.set_trace()
    split_datasets = {}
    for subset, tfrecord_paths in files.items():
        split_datasets[subset] = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
                                                .cache() \
                                                .shuffle(100) \
                                                .interleave(tf.data.TFRecordDataset) \
                                                .map(decode_example, num_parallel_calls=-1)
    return split_datasets


def load_data_from_tensor_slices(split_data, shuffle_train=True, seed=None):
    train_x = tf.data.Dataset.from_tensor_slices(split_data['train'][0])
    train_y = tf.data.Dataset.from_tensor_slices(split_data['train'][1])
    num_train_samples = len(split_data['train'][0])
    train_data = tf.data.Dataset.zip((train_x, train_y))
    if shuffle_train:
        train_data = train_data.shuffle(int(num_train_samples),seed=seed, reshuffle_each_iteration=True)
    train_data = train_data.cache()
    train_data = train_data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=tf.uint8),y), num_parallel_calls=-1)

    test_x = tf.data.Dataset.from_tensor_slices(split_data['test'][0])
    test_y = tf.data.Dataset.from_tensor_slices(split_data['test'][1])
    validation_data = tf.data.Dataset.zip((test_x, test_y))
    validation_data = validation_data.cache()
    validation_data = validation_data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=tf.uint8),y), num_parallel_calls=-1)

    return train_data, validation_data


def load_data_old(dataset_name='PNAS',
                  splits={'train':0.7,'validation':0.3},
                  threshold=50,
                  exclude_classes=[],
                  include_classes=[],
                  use_tfrecords=False,
                  tfrecord_dir=None,
                  samples_per_shard=800,
                  shuffle_train=True,
                  seed=None):
    '''
    This function relies on leavesdb rather than newer paleoai_data-based data sources.
    Deprecated as of (8/18/2020)

    Use newly named load_data function

    '''
    split_data, data_files, excluded_data_files, encoder = initialize_data_from_leavesdb(dataset_name=dataset_name,
                                                                                splits=splits,
                                                                                threshold=threshold,
                                                                                exclude_classes=exclude_classes,
                                                                                include_classes=include_classes)

    if use_tfrecords:
        split_datasets = load_data_from_tfrecords(tfrecord_dir=tfrecord_dir,
                                                  data=split_data,
                                                  samples_per_shard=samples_per_shard,
                                                  num_classes=len(encoder.classes))
        train_data, validation_data = split_datasets['train'], split_datasets['validation']

    else:
        train_data, validation_data = load_data_from_tensor_slices(split_data, shuffle_train=shuffle_train, seed=seed)

    return {'train':train_data,
            'validation':validation_data}, data_files, excluded_data_files

def load_data(data_fold: DataFold,
              exclude_classes=[],
              include_classes=[],
              use_tfrecords=False,
              tfrecord_dir=None,
              samples_per_shard=800,
              shuffle_train=True,
              seed=None):

    split_data, train_dataset, test_dataset, encoder = initialize_data_from_paleoai(fold=data_fold,
                                                                                    exclude_classes=exclude_classes,
                                                                                    include_classes=include_classes)
                                                                                    # subset_keys=['train','test'],
    if use_tfrecords:
        split_datasets = load_data_from_tfrecords(tfrecord_dir=tfrecord_dir,
                                                  data=split_data,
                                                  samples_per_shard=samples_per_shard,
                                                  subset_keys=['train','test'],
                                                  num_classes=len(encoder.classes))
        train_data, test_data = split_datasets['train'], split_datasets['test']

    else:
        train_data, test_data = load_data_from_tensor_slices(split_data, shuffle_train=shuffle_train, seed=seed)

    return {'train':train_data,
            'test':test_data}, train_dataset, test_dataset,  encoder



def create_dataset(data_fold: DataFold,
                   batch_size=32,
                   buffer_size=200,
                   exclude_classes=[],
                   include_classes=[],
                   target_size=(512,512),
                   num_channels=1,
                   color_mode='grayscale',
                   augmentations=[{}],
                   seed=None,
                   use_tfrecords=False,
                   tfrecord_dir=None,
                   samples_per_shard=800):

    dataset, train_dataset, test_dataset, encoder = load_data(data_fold=data_fold,
                                                              exclude_classes=exclude_classes,
                                                              include_classes=include_classes,
                                                              use_tfrecords=use_tfrecords,
                                                              tfrecord_dir=tfrecord_dir,
                                                              samples_per_shard=samples_per_shard,
                                                              seed=seed)
    if type(target_size)=='str':
        target_size = tuple(map(int, target_size.strip('()').split(',')))
    train_data = prep_dataset(dataset['train'],
                              batch_size=batch_size,
                              buffer_size=buffer_size,
                              shuffle=True,
                              target_size=target_size,
                              num_channels=num_channels,
                              color_mode=color_mode,
                              num_classes=train_dataset.num_classes,
                              augmentations=augmentations,
                              training=True,
                              seed=seed)

    test_data = prep_dataset(dataset['test'],
                            batch_size=batch_size,
                            target_size=target_size,
                            num_channels=num_channels,
                            color_mode=color_mode,
                            num_classes=test_dataset.num_classes,
                            training=False,
                            seed=seed)

    return train_data, test_data, train_dataset, test_dataset, encoder


# def create_dataset(dataset_name='PNAS',
#                    threshold=50,
#                    batch_size=32,
#                    buffer_size=200,
#                    exclude_classes=[],
#                    include_classes=[],
#                    target_size=(512,512),
#                    num_channels=1,
#                    color_mode='grayscale',
#                    splits={'train':0.7,'validation':0.3},
#                    augmentations=[{}],
#                    seed=None,
#                    use_tfrecords=False,
#                    tfrecord_dir=None,
#                    samples_per_shard=800):

#     dataset, data_files, excluded_data_files = load_data(dataset_name=dataset_name,
#                                                          splits=splits,
#                                                          threshold=threshold,
#                                                          seed=seed,
#                                                          exclude_classes=exclude_classes,
#                                                          include_classes=include_classes,
#                                                          use_tfrecords=use_tfrecords,
#                                                          tfrecord_dir=tfrecord_dir,
#                                                          samples_per_shard=samples_per_shard)
#     train_data = prep_dataset(dataset['train'],
#                               batch_size=batch_size,
#                               buffer_size=buffer_size,#int(data_files.num_samples*splits['train']),
#                               shuffle=True,
#                               target_size=target_size,
#                               num_channels=num_channels,
#                               color_mode=color_mode,
#                               num_classes=data_files.num_classes,
#                               augmentations=augmentations,
#                               training=True,
#                               seed=seed)
#     val_data = prep_dataset(dataset['validation'],
#                             batch_size=batch_size,
#                             target_size=target_size,
#                             num_channels=num_channels,
#                             color_mode=color_mode,
#                             num_classes=data_files.num_classes,
#                             training=False,
#                             seed=seed)
#     return train_data, val_data, data_files, excluded_data_files


##########################################################################
##########################################################################


def build_base_vgg16_RGB(cfg):

    base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                             include_top=False,
                                             input_tensor=Input(shape=(*cfg['target_size'],3)))

    return base


def build_head(base, num_classes=10):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense1 = tf.keras.layers.Dense(2048,activation='relu',name='dense1')
    dense2 = tf.keras.layers.Dense(512,activation='relu',name='dense2')
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    model = tf.keras.Sequential([
        base,
        global_average_layer,dense1,dense2,
        prediction_layer
        ])
    return model


def build_model(cfg):
    '''
    model_params = {
                'num_classes':cfg['num_classes'],
                'frozen_layers':cfg['frozen_layers'],
                'input_shape':(*cfg['target_size'],cfg['num_channels']),
                'base_learning_rate':cfg['lr'],
                'regularization':cfg['regularization'],
                'loss':'categorical_crossentropy'.
                'METRICS':['accuracy']
                }
    '''

    if cfg['model_name']=='vgg16':
        if cfg['num_channels']==1:
            model_builder = vgg16.VGG16GrayScale(cfg)
            build_base = model_builder.build_base
        else:
            build_base = partial(build_base_vgg16_RGB, cfg=cfg)

    elif cfg['model_name'].startswith('resnet'):
        model_builder = resnet.ResNet(cfg)
        build_base = model_builder.build_base

    base = build_base()
    model = build_head(base, num_classes=cfg['num_classes'])


    initial_learning_rate = cfg['lr']
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )


        # optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
    if cfg['optimizer']=='Adam':

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    if cfg['loss']=='categorical_crossentropy':
        loss = 'categorical_crossentropy'

    METRICS = []
    if 'accuracy' in cfg['METRICS']:
        METRICS.append('accuracy')
    if 'precision' in cfg['METRICS']:
        METRICS.append(tf.keras.metrics.Precision())
    if 'recall' in cfg['METRICS']:
        METRICS.append(tf.keras.metrics.Recall())


    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=METRICS)

    return model


##########################################################################
##########################################################################

def plot_sample(sample, num_res=1):
    num_samples = min(64, len(sample[0]))

    grid = gridspec.GridSpec(num_res, num_samples)
    grid.update(left=0, bottom=0, top=1, right=1, wspace=0.01, hspace=0.01)
    fig = plt.figure(figsize=[num_samples, num_res])
    for x in range(num_res):
        images = sample[x].numpy() #this converts the tensor to a numpy array
        images = np.squeeze(images)
        for y in range(num_samples):
            ax = fig.add_subplot(grid[x, y])
            ax.set_axis_off()
            ax.imshow((images[y] + 1.0)/2, cmap='gray')
    plt.show()

# neptune_logger = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs))
neptune_logger = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: log_data(logs),
                                                   on_epoch_end=lambda epoch, logs: log_data(logs))

from pyleaves.utils.neptune_utils import ImageLoggerCallback

# @hydra.main(config_path=Path(CONFIG_DIR,'PNAS_config.yaml'),config_name="PNAS")
# def train_pyleaves_dataset(cfg : DictConfig) -> None:
#     print(cfg.pretty())
#     import pdb; pdb.set_trace()
#     cfg_0 = cfg.stage_0
#     ensure_dir_exists(cfg['log_dir'])
#     ensure_dir_exists(cfg['model_dir'])
#     neptune.append_tag(cfg_0.dataset.dataset_name)
#     neptune.append_tag(cfg_0.model.model_name)
#     neptune.append_tag(str(cfg_0.dataset.target_size))
#     neptune.append_tag(cfg_0.dataset.num_channels)
#     neptune.append_tag(cfg_0.dataset.color_mode)
#     K.clear_session()
#     tf.random.set_seed(cfg_0.misc.seed)

#     train_dataset, validation_dataset, STAGE1_data_files, excluded = create_dataset(dataset_name=cfg_0.dataset.dataset_name,
#                                                                                     threshold=cfg_0.dataset.threshold,
#                                                                                     batch_size=cfg_0.training.batch_size,
#                                                                                     buffer_size=cfg_0.training.buffer_size,
#                                                                                     exclude_classes=cfg_0.dataset.exclude_classes,
#                                                                                     target_size=cfg_0.dataset.target_size,
#                                                                                     num_channels=cfg_0.dataset.num_channels,
#                                                                                     color_mode=cfg_0.dataset.color_mode,
#                                                                                     splits=cfg_0.dataset.splits,
#                                                                                     augmentations=cfg_0.training.augmentations,
#                                                                                     seed=cfg_0.misc.seed,
#                                                                                     use_tfrecords=cfg_0.misc.use_tfrecords,
#                                                                                     tfrecord_dir=cfg_0.dataset.tfrecord_dir,
#                                                                                     samples_per_shard=cfg_0.misc.samples_per_shard)

#     cfg_0.num_classes = STAGE1_data_files.num_classes
#     cfg['splits_size'] = {'train':{},
#                              'validation':{}}
#     cfg['splits_size']['train'] = int(STAGE1_data_files.num_samples*cfg['splits']['train'])
#     cfg['splits_size']['validation'] = int(STAGE1_data_files.num_samples*cfg['splits']['validation'])

#     cfg['steps_per_epoch'] = cfg['splits_size']['train']//cfg['BATCH_SIZE']
#     cfg['validation_steps'] = cfg['splits_size']['validation']//cfg['BATCH_SIZE']

#     neptune.set_property('num_classes',cfg['num_classes'])
#     neptune.set_property('steps_per_epoch',cfg['steps_per_epoch'])
#     neptune.set_property('validation_steps',cfg['validation_steps'])

#     # TODO: log encoder contents as dict
#     encoder = base_dataset.LabelEncoder(STAGE1_data_files.classes)

#     cfg['base_learning_rate'] = cfg['lr']
#     cfg['input_shape'] = (*cfg['target_size'],cfg['num_channels'])

#     # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
#     # with strategy.scope():
#     model = build_model(cfg)

#     # model = build_or_restore_model(cfg)
#     model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
#     pprint(cfg)

#     backup_callback = BackupAndRestore(cfg['checkpoints_path'])
#     backup_callback.set_model(model)
#     callbacks = [neptune_logger,
#                  backup_callback,
#                  EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)]#,
#     #              ImageLoggerCallback(data=train_dataset, freq=1000, max_images=-1, name='train', encoder=encoder),
#     #              ImageLoggerCallback(data=validation_dataset, freq=1000, max_images=-1, name='val', encoder=encoder),

#     history = model.fit(train_dataset,
#                         epochs=cfg['num_epochs'],
#                         callbacks=callbacks,
#                         validation_data=validation_dataset,
#                         shuffle=True,
#                         steps_per_epoch=cfg['steps_per_epoch'],
#                         validation_steps=cfg['validation_steps'])
#     #                     initial_epoch=0,

#     # TODO: Change build_model to build_or_load_model
#     model.save(cfg['saved_model_path'] + '-stage 1')
#     for k,v in cfg.items():
#         neptune.set_property(str(k),str(v))

#     if cfg['transfer_to_PNAS'] or cfg['transfer_to_Fossil']:
#         cfg['include_classes'] = STAGE1_data_files.classes
#         train_dataset, validation_dataset, STAGE2_data_files, STAGE2_excluded = create_dataset(dataset_name=cfg['stage_2']['dataset_name'], #cfg['dataset_name'],
#                                                                                                threshold=cfg['threshold'],
#                                                                                                batch_size=cfg['BATCH_SIZE'],
#                                                                                                buffer_size=cfg['buffer_size'],
#                                                                                                exclude_classes=cfg['exclude_classes'],
#                                                                                                include_classes=cfg['include_classes'],
#                                                                                                target_size=cfg['target_size'],
#                                                                                                num_channels=cfg['num_channels'],
#                                                                                                color_mode=cfg['color_mode'],
#                                                                                                splits=cfg['splits'],
#                                                                                                augmentations=cfg['augmentations'],
#                                                                                                seed=cfg['seed'])

#         cfg['num_classes'] = STAGE2_data_files.num_classes
#         cfg['splits_size'] = {'train':{},
#                                  'validation':{}}
#         cfg['splits_size']['train'] = int(STAGE2_data_files.num_samples*cfg['splits']['train'])
#         cfg['splits_size']['validation'] = int(STAGE2_data_files.num_samples*cfg['splits']['validation'])

#         cfg['steps_per_epoch'] = cfg['splits_size']['train']//cfg['BATCH_SIZE']
#         cfg['validation_steps'] = cfg['splits_size']['validation']//cfg['BATCH_SIZE']

#         backup_callback = BackupAndRestore(cfg['checkpoints_path'])
#         backup_callback.set_model(model)
#         callbacks = [neptune_logger,
#                      backup_callback,
#                      EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)]#,

#         history = model.fit(train_dataset,
#                             epochs=cfg['num_epochs'],
#                             callbacks=callbacks,
#                             validation_data=validation_dataset,
#                             shuffle=True,
#                             steps_per_epoch=cfg['steps_per_epoch'],
#                             validation_steps=cfg['validation_steps'])
#     return history


def log_config(cfg: DictConfig, verbose: bool=False):
    if verbose: print(cfg.pretty())

    cfg_0 = cfg.stage_0
    ensure_dir_exists(cfg['log_dir'])
    ensure_dir_exists(cfg['model_dir'])
    neptune.append_tag(cfg_0.dataset.dataset_name)
    neptune.append_tag(cfg_0.model.model_name)
    neptune.append_tag(str(cfg_0.dataset.target_size))
    neptune.append_tag(cfg_0.dataset.num_channels)
    neptune.append_tag(cfg_0.dataset.color_mode)


def log_dataset(cfg: DictConfig, train_dataset: BaseDataset, test_dataset: BaseDataset):
    cfg['num_classes'] = train_dataset.num_classes
    cfg['splits_size'] = {'train':{},
                          'test':{}}
    cfg['splits_size']['train'] = int(train_dataset.num_samples)
    cfg['splits_size']['test'] = int(test_dataset.num_samples)

    cfg['steps_per_epoch'] = cfg['splits_size']['train']//cfg['BATCH_SIZE']
    cfg['validation_steps'] = cfg['splits_size']['test']//cfg['BATCH_SIZE']

    neptune.set_property('num_classes',cfg['num_classes'])
    neptune.set_property('steps_per_epoch',cfg['steps_per_epoch'])
    neptune.set_property('validation_steps',cfg['validation_steps'])



def train_single_fold(fold: DataFold, cfg : DictConfig, verbose: bool=True) -> None:
    setGPU()
    set_tf_config()

    
    cfg.tfrecord_dir = os.path.join(cfg.tfrecord_dir,fold.fold_name)
    ensure_dir_exists(cfg.tfrecord_dir)
    if verbose:
        print('='*20)
        print(cfg.tfrecord_dir)
        print('='*20)

    K.clear_session()
    train_data, test_data, train_dataset, test_dataset, encoder = create_dataset(data_fold=fold,
                                                                                batch_size=cfg.training.batch_size,
                                                                                buffer_size=cfg.training.buffer_size,
                                                                                exclude_classes=cfg.dataset.exclude_classes,
                                                                                include_classes=cfg.dataset.include_classes,
                                                                                target_size=cfg.dataset.target_size,
                                                                                num_channels=cfg.dataset.num_channels,
                                                                                color_mode=cfg.dataset.color_mode,
                                                                                augmentations=cfg.training.augmentations,
                                                                                seed=cfg.misc.seed,
                                                                                use_tfrecords=cfg.misc.use_tfrecords,
                                                                                tfrecord_dir=cfg.tfrecord_dir,
                                                                                samples_per_shard=cfg.misc.samples_per_shard)

    if verbose: print(f'Starting fold {fold.fold_id}')
    log_dataset(cfg=cfg, train_dataset=train_dataset, test_dataset=test_dataset)

    cfg['base_learning_rate'] = cfg['lr']
    cfg['input_shape'] = (*cfg['target_size'],cfg['num_channels'])

    model = build_model(cfg)

    model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
    pprint(cfg)

    backup_callback = BackupAndRestore(cfg['checkpoints_path'])
    backup_callback.set_model(model)
    callbacks = [neptune_logger,
                backup_callback,
                EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True),
                ImageLoggerCallback(data=train_dataset, freq=1000, max_images=-1, name='train', encoder=encoder),
                ImageLoggerCallback(data=test_dataset, freq=1000, max_images=-1, name='val', encoder=encoder)]

    history = model.fit(train_data,
                        epochs=cfg['num_epochs'],
                        callbacks=callbacks,
                        validation_data=test_data,
                        shuffle=True,
                        steps_per_epoch=cfg['steps_per_epoch'],
                        validation_steps=cfg['validation_steps'])
    return history


# from keras.wrappers.scikit_learn import KerasClassifier
# from tune_sklearn import TuneGridSearchCV
from joblib import Parallel, delayed


def train_paleoai_dataset(cfg : DictConfig, n_jobs: int=1, verbose: bool=False) -> None:

    cfg_0 = cfg.stage_0
    cfg_1 = cfg.stage_1

    log_config(cfg=cfg, verbose=verbose)
    # log_config(cfg=cfg_1, verbose=verbose)
    # import tensorflow as tf

    # tf.random.set_seed(cfg_0.misc.seed)

    kfold_loader = KFoldLoader(root_dir=cfg_0.dataset.fold_dir)

    kfold_iter = kfold_loader.iter_folds(repeats=1)
    histories = Parallel(n_jobs=n_jobs)(delayed(train_single_fold)(fold=fold, cfg=copy.deepcopy(cfg_0)) for i, fold in enumerate(kfold_iter))

    import pdb; pdb.set_trace()

    return histories

    # for i, fold in enumerate(kfold_iter):

    # model.save(cfg['saved_model_path'] + '-stage 0')
    # for k,v in cfg.items():
    #     neptune.set_property(str(k),str(v))

    #     cfg['include_classes'] = STAGE1_data_files.classes
    #     train_dataset, validation_dataset, STAGE2_data_files, STAGE2_excluded = create_dataset(dataset_name=cfg['stage_2']['dataset_name'], #cfg['dataset_name'],
    #                                                                                            threshold=cfg['threshold'],
    #                                                                                            batch_size=cfg['BATCH_SIZE'],
    #                                                                                            buffer_size=cfg['buffer_size'],
    #                                                                                            exclude_classes=cfg['exclude_classes'],
    #                                                                                            include_classes=cfg['include_classes'],
    #                                                                                            target_size=cfg['target_size'],
    #                                                                                            num_channels=cfg['num_channels'],
    #                                                                                            color_mode=cfg['color_mode'],
    #                                                                                            splits=cfg['splits'],
    #                                                                                            augmentations=cfg['augmentations'],
    #                                                                                            seed=cfg['seed'])

    #     cfg['num_classes'] = STAGE2_data_files.num_classes
    #     cfg['splits_size'] = {'train':{},
    #                              'validation':{}}
    #     cfg['splits_size']['train'] = int(STAGE2_data_files.num_samples*cfg['splits']['train'])
    #     cfg['splits_size']['validation'] = int(STAGE2_data_files.num_samples*cfg['splits']['validation'])

    #     cfg['steps_per_epoch'] = cfg['splits_size']['train']//cfg['BATCH_SIZE']
    #     cfg['validation_steps'] = cfg['splits_size']['validation']//cfg['BATCH_SIZE']

    #     backup_callback = BackupAndRestore(cfg['checkpoints_path'])
    #     backup_callback.set_model(model)
    #     callbacks = [neptune_logger,
    #                  backup_callback,
    #                  EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)]#,

    #     history = model.fit(train_dataset,
    #                         epochs=cfg['num_epochs'],
    #                         callbacks=callbacks,
    #                         validation_data=validation_dataset,
    #                         shuffle=True,
    #                         steps_per_epoch=cfg['steps_per_epoch'],
    #                         validation_steps=cfg['validation_steps'])
    # return history








@hydra.main(config_path=Path(CONFIG_DIR,'Leaves-PNAS.yaml'))
def train(cfg : DictConfig) -> None:

    cfg = restore_or_initialize_experiment(cfg, restore_last=True, prefix='log_dir__', verbose=2)

    neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name), params=params):
        # train_pyleaves_dataset(cfg)
        train_paleoai_dataset(cfg=cfg, n_jobs=1, verbose=True)



if __name__=='__main__':

    train()







    # PARAMS = {'neptune_project_name':'jacobarose/sandbox',
    #           'neptune_experiment_dir':'/media/data/jacob/sandbox_logs',
    #           'optimizer':'Adam',
    #           'loss':'categorical_crossentropy',
    #           'lr':1e-4,
    #           'color_mode':'grayscale',
    #           'num_channels':3,
    #           'BATCH_SIZE':16,
    #           'buffer_size':200,
    #           'num_epochs':150,
    #           'dataset_name':'Leaves',#'Fossil',#'PNAS',#
    #           'threshold':2,
    #           'frozen_layers':None,
    #           'model_name':'resnet_50_v2',#'vgg16',#
    #           'splits':{'train':0.5,'validation':0.5},
    #           'seed':45,
    #           'use_tfrecords':True,
    #           'samples_per_shard':400}
    #
    #
    # PARAMS['transfer_to_PNAS']=False
    # PARAMS['transfer_to_Fossil']=False#True

    # PARAMS['stage-2'] = {}#'lr':1e-5,
                         # 'color_mode':'grayscale',
                         # 'num_channels':3,
                         # 'BATCH_SIZE':8,
                         # 'buffer_size':200,
                         # 'num_epochs':150,
                         # 'dataset_name':'Fossil',#'PNAS',
                         # 'threshold':2,
                         # 'frozen_layers':None,
                         # 'model_name':'vgg16',#'resnet_50_v2',
                         # 'splits':{'train':0.5,'validation':0.5},
                         # 'seed':45}

    # PARAMS = stuf(PARAMS)
    # PARAMS['exclude_classes'] = ['notcataloged','notcatalogued', 'II. IDs, families uncertain', 'Unidentified']
    # PARAMS['regularization'] = {'l1':3e-4}
    # PARAMS['METRICS'] = ['accuracy','precision','recall']
    # PARAMS['target_size'] = (512,512) #(128,128)#(256,256)# #(768,768)#
    # PARAMS['augmentations'] = [{'flip':1.0}]
