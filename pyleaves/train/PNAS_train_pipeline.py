# @Author: Jacob A Rose
# @Date:   Thu, July 2nd 2020, 3:39 am
# @Email:  jacobrose@brown.edu
# @Filename: PNAS_train_pipeline.py


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict
from stuf import stuf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
from pprint import pprint
if __name__=='main':
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import tensorflow as tf
if __name__=='main':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        print('setting memory growth failed, continuing anyway.')
    random.seed(84)
    np.random.seed(58)
    tf.random.set_seed(34)
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from pyleaves.models import resnet, vgg16
from pyleaves.datasets import leaves_dataset, fossil_dataset, pnas_dataset, base_dataset
import neptune
import arrow
from pyleaves.utils import ensure_dir_exists
from more_itertools import unzip

##########################################################################
##########################################################################

def image_reshape(x):
    return [
        tf.image.resize(x, (7, 7)),
        tf.image.resize(x, (14, 14)),
        x
    ]


def load_img(image_path):#, img_size=(224,224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def resize(image, h=512, w=512):
    return tf.image.resize_with_pad(image, target_height=h, target_width=w)


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

def rotate(x, y):
    """Rotation augmentation

    Args:
        x,     tf.Tensor: Image
        y,     tf.Tensor: arbitrary tensor, passes through unchanged

    Returns:
        Augmented image, y
    """
    # Rotate 0, 90, 180, 270 degrees

    # angles = tf.random.uniform(shape=[],minval=0, maxval=3.14159, seed=2)

    # return tfa.image.rotate(x, angles, interpolation='NEAREST'), y
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32,seed=2)), y


def flip(x, y):
    """Flip augmentation

    Args:
        x,     tf.Tensor: Image to flip
        y,     tf.Tensor: arbitrary tensor, passes through unchanged
    Returns:
        Augmented image, y
    """
    x = tf.image.random_flip_left_right(x, seed=3)
    x = tf.image.random_flip_up_down(x, seed=4)

    return x, y

def color(x, y):
    """Color augmentation

    Args:
        x,     tf.Tensor: Image
        y,     tf.Tensor: arbitrary tensor, passes through unchanged

    Returns:
        Augmented image, y
    """
    x = tf.image.random_hue(x, 0.08, seed=5)
    x = tf.image.random_saturation(x, 0.6, 1.6, seed=6)
    x = tf.image.random_brightness(x, 0.05, seed=7)
    x = tf.image.random_contrast(x, 0.7, 1.3, seed=8)
    return x, y

def _cond_apply(x, y, func, prob):
    """Conditionally apply func to x and y with probability prob."""
    return tf.cond((tf.random.uniform([], 0, 1) >= (1.0 - prob)), lambda: func(x,y), lambda: (x,y))

def augment_sample(x, y, prob=1.0):
    x, y = _cond_apply(x, y, flip, prob)
    x, y = _cond_apply(x, y, rotate, prob)
    x, y = _cond_apply(x, y, color, prob)
    return x, y

preprocess_input(tf.zeros([4, 224, 224, 3]))
def apply_preprocess(x, y, num_classes=10):
    return preprocess_input(x), tf.one_hot(y, depth=num_classes)

# def create_mnist_dataset(batch_size):
#     (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
#     train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
#     train_images = train_images/127.5  - 1
#     dataset = tf.data.Dataset.from_tensor_slices(train_images)
#     dataset = dataset.map(image_reshape)
#     dataset = dataset.cache()
#     dataset = dataset.shuffle(len(train_images))
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.prefetch(1)
#     return dataset


def prep_dataset(dataset,
                 batch_size=32,
                 buffer_size=100,
                 shuffle=False,
                 target_size=(512,512),
                 num_channels=3,
                 color_mode='grayscale',
                 num_classes=10,
                 augmentations=[],
                 aug_prob=1.0):
    dataset = dataset.map(lambda x,y: (resize(x, *target_size),y), num_parallel_calls=-1)

    dataset = dataset.map(lambda x,y: apply_preprocess(x,y,num_classes),num_parallel_calls=-1)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.repeat()

    if 'flip' in augmentations:
        dataset = dataset.map(lambda x,y: _cond_apply(x, y, flip, prob=aug_prob), num_parallel_calls=-1)
    if 'rotate' in augmentations:
        dataset = dataset.map(lambda x,y: _cond_apply(x, y, rotate, prob=aug_prob), num_parallel_calls=-1)
    if 'color' in augmentations:
        dataset = dataset.map(lambda x,y: _cond_apply(x, y, color, prob=aug_prob), num_parallel_calls=-1)
        # dataset = dataset.map(lambda x,y: augment_sample(x, y, prob=aug_prob), num_parallel_calls=-1)

    if color_mode=='grayscale':
        if num_channels==3:
            dataset = dataset.map(lambda x,y: rgb2gray_3channel(x, y), num_parallel_calls=-1)
        elif num_channels==1:
            dataset = dataset.map(lambda x,y: rgb2gray_1channel(x, y), num_parallel_calls=-1)

    # dataset = dataset.cache()
    # dataset = dataset.repeat()
#         dataset = dataset.map(augment_sample, num_parallel_calls=-1)
    # if shuffle:
    #     dataset = dataset.shuffle(buffer_size)
    # dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


def create_Imagenette_dataset(batch_size,
                              target_size=(512,512),
                              augment_train=True,
                              aug_prob=1.0):

    data, info = tfds.load('Imagenette', as_supervised=True, with_info=True)
    train_data = prep_dataset(data['train'],
                              batch_size=batch_size,
                              buffer_size=batch_size*10,
                              shuffle=True,
                              target_size=target_size,
                              augment=augment_train,
                              aug_prob=aug_prob)
    val_data = prep_dataset(data['validation'],
                            batch_size=batch_size,
                            target_size=target_size)

    return train_data, val_data, info



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

def load_data(dataset_name='PNAS', splits={'train':0.7,'validation':0.3}, threshold=50):
    datasets = {
            'PNAS': pnas_dataset.PNASDataset(),
            'Leaves': leaves_dataset.LeavesDataset(),
            'Fossil': fossil_dataset.FossilDataset()
            }
    data_files = datasets[dataset_name]

    data_files.exclude_rare_classes(threshold=threshold)
    encoder = base_dataset.LabelEncoder(data_files.classes)
    data_files, _ = data_files.enforce_class_whitelist(class_names=encoder.classes)

    x = list(data_files.data['path'].values)
    y = np.array(encoder.encode(data_files.data['family']))

    shuffled_data = list(zip(x,y))
    random.shuffle(shuffled_data)
    partitioned_data = partition_data(data=shuffled_data,
                                      partitions=OrderedDict(splits))
    split_data = {k:v for k,v in partitioned_data.items() if len(v)>0}

    for subset, subset_data in split_data.items():
        split_data[subset] = [list(i) for i in unzip(subset_data)]

    paths = tf.data.Dataset.from_tensor_slices(split_data['train'][0])
    labels = tf.data.Dataset.from_tensor_slices(split_data['train'][1])
    train_data = tf.data.Dataset.zip((paths, labels))
    train_data = train_data.shuffle(int(data_files.num_samples*splits['train']))
    train_data = train_data.cache()
    train_data = train_data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=tf.uint8),y), num_parallel_calls=-1)

    paths = tf.data.Dataset.from_tensor_slices(split_data['validation'][0])
    labels = tf.data.Dataset.from_tensor_slices(split_data['validation'][1])
    validation_data = tf.data.Dataset.zip((paths, labels))
    validation_data = validation_data.cache()
    validation_data = validation_data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=tf.uint8),y), num_parallel_calls=-1)

    return {'train':train_data,
            'validation':validation_data}, data_files


def create_dataset(dataset_name='PNAS',
                   threshold=50,
                   batch_size=32,
                   buffer_size=200,
                   target_size=(512,512),
                   num_channels=1,
                   color_mode='grayscale',
                   splits={'train':0.7,'validation':0.3},
                   augmentations=[],
                   aug_prob=1.0):

    dataset, data_files = load_data(dataset_name=dataset_name, splits=splits, threshold=threshold)
    train_data = prep_dataset(dataset['train'],
                              batch_size=batch_size,
                              buffer_size=buffer_size,#int(data_files.num_samples*splits['train']),
                              shuffle=True,
                              target_size=target_size,
                              num_channels=num_channels,
                              color_mode=color_mode,
                              num_classes=data_files.num_classes,
                              augmentations=augmentations,
                              aug_prob=aug_prob)
    val_data = prep_dataset(dataset['validation'],
                            batch_size=batch_size,
                            target_size=target_size,
                            num_channels=num_channels,
                            color_mode=color_mode,
                            num_classes=data_files.num_classes)
    return train_data, val_data, data_files


# def create_pnas_dataset(batch_size):
#     data_files = pnas_dataset.PNASDataset()
#     dataset = tf.data.Dataset.from_tensor_slices(data_files.data['path'])
#     dataset = dataset.map(load_img, num_parallel_calls=-1)
#     dataset = dataset.map(resize, num_parallel_calls=-1)
#     dataset = dataset.cache()
#     dataset = dataset.shuffle(20)#len(data_files.data))
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.prefetch(1)
#     return dataset

##########################################################################
##########################################################################


def build_base_vgg16_RGB(PARAMS):
#     if PARAMS['optimizer']=='Adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS['lr'])

    base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                             include_top=False,
                                             input_tensor=Input(shape=(*PARAMS['target_size'],3)))

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

from functools import partial

def build_model(PARAMS):
    '''
    model_params = {
                'num_classes':PARAMS['num_classes'],
                'frozen_layers':PARAMS['frozen_layers'],
                'input_shape':(*PARAMS['target_size'],PARAMS['num_channels']),
                'base_learning_rate':PARAMS['lr'],
                'regularization':PARAMS['regularization'],
                'loss':'categorical_crossentropy'.
                'METRICS':['accuracy']
                }
    '''

    if PARAMS['model_name']=='vgg16':
        if PARAMS['num_channels']==1:
            model_builder = vgg16.VGG16GrayScale(PARAMS)
            build_base = model_builder.build_base
        else:
            build_base = partial(build_base_vgg16_RGB, PARAMS=PARAMS)

    elif PARAMS['model_name'].startswith('resnet'):
        model_builder = resnet.ResNet(PARAMS)
        build_base = model_builder.build_base

    base = build_base()
    model = build_head(base, num_classes=PARAMS['num_classes'])

    if PARAMS['optimizer']=='Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS['lr'])

    if PARAMS['loss']=='categorical_crossentropy':
        loss = 'categorical_crossentropy'

    METRICS = []
    if 'accuracy' in PARAMS['METRICS']:
        METRICS.append('accuracy')
    if 'precision' in PARAMS['METRICS']:
        METRICS.append(tf.keras.metrics.Precision())
    if 'recall' in PARAMS['METRICS']:
        METRICS.append(tf.keras.metrics.Recall())


    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=METRICS)

    return model



def log_data(logs):
    for k, v in logs.items():
        neptune.log_metric(k, v)


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

# def train_imagenette(PARAMS):
#
#     neptune.append_tag(PARAMS['dataset_name'])
#     neptune.append_tag(PARAMS['model_name'])
#
#     K.clear_session()
#     tf.random.set_seed(34)
#     target_size = PARAMS['target_size']
#     BATCH_SIZE = PARAMS['BATCH_SIZE']
#
#     train_dataset, validation_dataset, info = create_Imagenette_dataset(BATCH_SIZE, target_size=target_size, augment_train=PARAMS['augment_train'])
#     num_classes = info.features['label'].num_classes
#
#     encoder = base_dataset.LabelEncoder(info.features['label'].names)
#
#     train_dataset = train_dataset.map(lambda x,y: apply_preprocess(x,y,num_classes),num_parallel_calls=-1)
#     validation_dataset = validation_dataset.map(lambda x,y: apply_preprocess(x,y,num_classes),num_parallel_calls=-1)
#
#     PARAMS['num_classes'] = num_classes
#     steps_per_epoch = info.splits['train'].num_examples//BATCH_SIZE
#     validation_steps = info.splits['validation'].num_examples//BATCH_SIZE
#
#     neptune.set_property('num_classes',num_classes)
#     neptune.set_property('steps_per_epoch',steps_per_epoch)
#     neptune.set_property('validation_steps',validation_steps)
#
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS['learning_rate'])
#     loss = 'categorical_crossentropy'
#     METRICS = ['accuracy']
#
#     base = tf.keras.applications.vgg16.VGG16(weights='imagenet',
#                                              include_top=False,
#                                              input_tensor=Input(shape=(*target_size,3)))
#
#     # TODO try freezing weights for input_shape != (224,224)
#
#     model = build_head(base, num_classes=num_classes)
#
#     model.compile(optimizer=optimizer,
#                   loss=loss,
#                   metrics=METRICS)
#
#     callbacks = [neptune_logger,
#                  ImageLoggerCallback(data=train_dataset, freq=10, max_images=-1, name='train', encoder=encoder),
#                  ImageLoggerCallback(data=validation_dataset, freq=10, max_images=-1, name='val', encoder=encoder),
#                  EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
#
#     model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
#     pprint(PARAMS)
#     history = model.fit(train_dataset,
#                         epochs=10,
#                         callbacks=callbacks,
#                         validation_data=validation_dataset,
#                         shuffle=True,
#                         initial_epoch=0,
#                         steps_per_epoch=steps_per_epoch,
#                         validation_steps=validation_steps)


def train_pnas(PARAMS):
    ensure_dir_exists(PARAMS['log_dir'])
    ensure_dir_exists(PARAMS['model_dir'])
    neptune.append_tag(PARAMS['dataset_name'])
    neptune.append_tag(PARAMS['model_name'])
    neptune.append_tag(str(PARAMS['target_size']))
    neptune.append_tag(PARAMS['num_channels'])
    neptune.append_tag(PARAMS['color_mode'])
    K.clear_session()
    tf.random.set_seed(34)


    train_dataset, validation_dataset, data_files = create_dataset(dataset_name=PARAMS['dataset_name'],
                                                                   threshold=PARAMS['threshold'],
                                                                   batch_size=PARAMS['BATCH_SIZE'],
                                                                   buffer_size=PARAMS['buffer_size'],
                                                                   target_size=PARAMS['target_size'],
                                                                   num_channels=PARAMS['num_channels'],
                                                                   color_mode=PARAMS['color_mode'],
                                                                   splits=PARAMS['splits'],
                                                                   augmentations=PARAMS['augmentations'],
                                                                   aug_prob=PARAMS['aug_prob'])


    PARAMS['num_classes'] = data_files.num_classes
    PARAMS['splits_size'] = {'train':{},
                       'validation':{}}
    PARAMS['splits_size']['train'] = int(data_files.num_samples*PARAMS['splits']['train'])
    PARAMS['splits_size']['validation'] = int(data_files.num_samples*PARAMS['splits']['validation'])

    steps_per_epoch = PARAMS['splits_size']['train']//PARAMS['BATCH_SIZE']
    validation_steps = PARAMS['splits_size']['validation']//PARAMS['BATCH_SIZE']

    neptune.set_property('num_classes',PARAMS['num_classes'])
    neptune.set_property('steps_per_epoch',steps_per_epoch)
    neptune.set_property('validation_steps',validation_steps)


    encoder = base_dataset.LabelEncoder(data_files.classes)

#     METRICS = ['accuracy']
    callbacks = [neptune_logger,
                 ImageLoggerCallback(data=train_dataset, freq=1000, max_images=-1, name='train', encoder=encoder),
                 ImageLoggerCallback(data=validation_dataset, freq=1000, max_images=-1, name='val', encoder=encoder),
                 EarlyStopping(monitor='val_loss', patience=25, verbose=1)]

    PARAMS['base_learning_rate'] = PARAMS['lr']
    PARAMS['input_shape'] = (*PARAMS['target_size'],PARAMS['num_channels'])
    model = build_model(PARAMS)


#     if PARAMS['optimizer']=='Adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS['lr'])

    model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
    pprint(PARAMS)
    history = model.fit(train_dataset,
                        epochs=PARAMS['num_epochs'],
                        callbacks=callbacks,
                        validation_data=validation_dataset,
                        shuffle=True,
                        initial_epoch=0,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)


    for k,v in PARAMS.items():
        neptune.set_property(str(k),str(v))

    return history

if __name__=='__main__':
    PARAMS = {'neptune_project_name':'jacobarose/sandbox',
              'experiment_dir':'/media/data/jacob/sandbox_logs',
              'experiment_start_time':arrow.utcnow().format('YYYY-MM-DD_HH-mm-ss'),
              'optimizer':'Adam',
              'loss':'categorical_crossentropy',
              'lr':1e-5,
              'color_mode':'grayscale',
              'num_channels':3,
              'BATCH_SIZE':8,
              'buffer_size':400,
              'num_epochs':150,
              'dataset_name':'PNAS',
              'threshold':2,
              'frozen_layers':None,
              'model_name':'vgg16',#'resnet_50_v2',
              # 'augment_train':True,
              'aug_prob':1.0,
              'splits':{'train':0.5,'validation':0.5}}#stuf({'train':0.5,'validation':0.5})}

    PARAMS = stuf(PARAMS)

    PARAMS['experiment_name'] = '_'.join([PARAMS['dataset_name'], PARAMS['model_name']])
    PARAMS['regularization'] = {'l1':3e-4}
    PARAMS['METRICS'] = ['accuracy','precision','recall']
    PARAMS['target_size'] = (768,768)#(128,128)#(256,256)#(512,512)#
    PARAMS['augmentations'] = ['flip']

    PARAMS['log_dir'] = os.path.join(PARAMS['experiment_dir'], PARAMS['experiment_name'], 'log_dir', PARAMS['loss'], PARAMS['experiment_start_time'])
    PARAMS['model_dir'] = os.path.join(PARAMS['log_dir'],'model_dir')

    neptune.init(project_qualified_name=PARAMS['neptune_project_name'])
    with neptune.create_experiment(name=PARAMS['experiment_name']+'-'+str(PARAMS['splits']), params=PARAMS):
        train_pnas(PARAMS)


# In[ ]:
