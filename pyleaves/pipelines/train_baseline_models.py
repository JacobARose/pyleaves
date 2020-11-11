#!/usr/bin/env python
# coding: utf-8

# # Train baseline models and save them

# ## Libraries & utils

# In[2]:

import sys
import os
import subprocess
import warnings
from IPython.display import display
warnings.filterwarnings('ignore')
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from pyleaves.utils import set_tf_config
set_tf_config(num_gpus=1)
import tensorflow as tf
import hashlib

from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm, spearmanr, wasserstein_distance
from skimage.metrics import structural_similarity
from omegaconf import OmegaConf
from pyleaves.utils import save_class_labels
from pyleaves.utils.WandB_artifact_utils import load_Leaves_Minus_PNAS_dataset, load_dataset_from_artifact
import wandb
from wandb.keras import WandbCallback
    
def random_resample_dataset(data: pd.DataFrame, y_col='family', target_class_population: int=None, set_class_floor: int=None, set_class_ceil: int=None):
    
    y = data[y_col].values
    counter = Counter(y)
  
    min_pop = min(list(counter.values()))
    max_pop = max(list(counter.values()))
    mean_pop = np.mean(list(counter.values()))
    num_samples = data.shape[0]


    if target_class_population:
        target_class_population = target_class_population or max_pop
        resampled = []
        print(f'[INFO] Current class population min={min_pop}|max={max_pop}|mean={mean_pop:.1f}')
        print(f'[INFO] Resampling data to a uniform class population of {target_class_population} samples/class')
        for family_name, fam in data.groupby(y_col):
            resampled.append(fam.sample(target_class_population, replace=True))
        data = pd.concat(resampled)
        print(f'[INFO] Random resampling complete. Previous num_samples={y.shape[0]}, new num_samples={data.shape[0]}')


    if set_class_ceil:
        undersampled = []
        for _, fam in data.groupby(y_col):
            if fam.shape[0] > set_class_ceil:
                undersampled.append(fam.sample(set_class_ceil, replace=True))
            else:
                undersampled.append(fam)
        data = pd.concat(undersampled)
        print(f'[INFO] Random undersampling complete. Previous num_samples={y.shape[0]}, new num_samples={data.shape[0]}')
        print(f'[INFO] Random undersampling complete. Previous max_pop={max_pop}, new max_pop={set_class_ceil}')

    if set_class_floor:
        oversampled = []
        for _, fam in data.groupby(y_col):
            if fam.shape[0] < set_class_floor:
                oversampled.append(fam.sample(set_class_floor, replace=True))
            else:
                oversampled.append(fam)
        data = pd.concat(oversampled)
        print(f'[INFO] Random oversampling complete. Previous num_samples={num_samples}, new num_samples={data.shape[0]}')
        print(f'[INFO] Random oversampling complete. Previous min_pop={min_pop}, new min_pop={set_class_floor}')

    return oversampled

os.environ['WANDB_NOTEBOOK_NAME'] = 'baseline_models' 

from tensorflow.keras.callbacks import EarlyStopping
from pyleaves.utils.img_aug_utils import apply_cutmixup

from sklearn.model_selection import train_test_split
from pyleaves.pipelines.WandB_Leaves_vs_PNAS import *

import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from functools import partial
from pyleaves.utils.tf_utils import BalancedAccuracyMetric


def get_metrics(metrics_list, num_classes=None):
    METRICS = []
    if 'f1' in metrics_list:
        METRICS.append(tfa.metrics.F1Score(num_classes=num_classes,
                                        average='weighted',
                                        name='weighted_f1'))
    if 'accuracy' in metrics_list:
        METRICS.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))
    if 'top-3_accuracy' in metrics_list:
        METRICS.append(tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top-3_accuracy'))
    if 'top-5_accuracy' in metrics_list:
        METRICS.append(tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top-5_accuracy'))
    if 'balanced_accuracy' in metrics_list:
        METRICS.append(BalancedAccuracyMetric(num_classes))
    if 'precision' in metrics_list:
        METRICS.append(tf.keras.metrics.Precision())
    if 'recall' in metrics_list:
        METRICS.append(tf.keras.metrics.Recall())
    return METRICS


def generate_weighted_accuracy():
    nb_classes = 3
    epsilon = 1e-5

    def metric(y_true, y_pred, weights=None):

        __name__ = 'Weighted Accuracy'

        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        nominator = tf.cast([tf.logical_and(y_pred == y_true, y_true == i) for i in range(nb_classes)], tf.float32)
        nominator = tf.reduce_sum(nominator, axis=-1)

        denominator = tf.cast([y_true == i for i in range(nb_classes)], tf.float32)
        denominator = tf.reduce_sum(denominator, axis=-1) + epsilon

        acc_per_class = nominator / denominator
        avg_acc = tf.reduce_sum(acc_per_class) / nb_classes

        return avg_acc

    return metric

class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """ Cosine Annealing Learning Rate Scheduling with Warmup"""
 
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0,
                 logging_freq=1):
        """
                 Initialization parameters 
                 :param learning_rate_base: base learning rate
                 :param total_steps: the total number of batch steps epoch * num_samples / batch_size
                 :param global_step_init: initial
                 :param warmup_learning_rate: Warmup learning rate default 0.0
                 :param warmup_steps: The number of warmup steps defaults to 0
        :param hold_base_rate_steps:
        :param verbose:
        """
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
                 # Whether to print the learning rate at the end of each training
        self.verbose = verbose
        self.logging_freq = logging_freq
                 # Record each accurate learning rate of all batches, which can be used for printing and display
        self.learning_rates = []
 
    def on_batch_end(self, batch, logs=None):
                 # 1. The current number of steps before the batch starts +1
        self.global_step = self.global_step + 1
                 # 2. Get the last learning rate of the optimizer and record it
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
 
    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
                 # 1. Pass the parameters and the number of records and the last learning rate
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
                 # 2. Set the learning rate of the optimizer this time
        K.set_value(self.model.optimizer.lr, lr)
#         if self.verbose > 0 and (batch % self.logging_freq == 0):
#              print('\nNumber of batches %05d: Set the learning rate %s.' % (self.global_step + 1, lr))

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0:
            lr = self.model.optimizer.lr
            print('\n Number of batches %05d: Set the learning rate %s.' % (self.global_step + 1, lr.numpy()))



def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
         Calculation of warmup cosine annealing learning rate for each batch
         :param global_step: the number of steps currently reached
         :param learning_rate_base: base learning rate after warmup
         :param total_steps: total number of batches required
         :param warmup_learning_rate: the learning rate at which warmup started
         :param warmup_steps:warmup learning rate steps
         :param hold_base_rate_steps: reserved total step number and warmup step number interval
    :return:
    """
 
    if total_steps < warmup_steps:
        raise ValueError('The total number of steps must be greater than warmup')
 
    # 1. Cosine annealing learning rate calculation
    # Calculate from the end of warmup
    # 0.5 * 0.01 * (1 + cos(pi*(1-5-0)/(10 - 5 - 0))
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
                                                            np.pi *
                                                    (
                                                        global_step - warmup_steps - hold_base_rate_steps
                                                    ) / float(total_steps - warmup_steps - hold_base_rate_steps)))

    # 2, the learning rate calculation after warmup
    # If the reservation is greater than 0, judge whether the current number of steps> warmup steps + reserved steps, if yes, return to the learning rate calculated above, if not, use the basic learning rate after warmup
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                     learning_rate, learning_rate_base)
        # 3. The number of warmup steps is greater than 0
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
             raise ValueError('The learning rate after warmup must be greater than the starting learning rate of warmup')
        # 1. Calculate a difference/warmup_steps between 0.01 and 0.000006 to get the increase before warmup ends
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        # 2. Calculate the learning rate of the next step global_step of warmup
        warmup_rate = slope * global_step + warmup_learning_rate
        # 3. If global_step is judged to be less than warmup_steps, return the learning rate of this warmup at that time, otherwise directly return to the calculation of cosine annealing
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                 learning_rate)

    # 4. If the last number of steps reached is greater than the total number of steps, return to 0, otherwise return to the current calculated learning rate (may be warmup learning rate or the result of cosine decay)
    return np.where(global_step > total_steps, 0.0, learning_rate)

from tensorflow.python.keras.layers import Dropout, Input, Conv2D, MaxPooling2D

def build_model(model_params, config: DictConfig, dropout_rate: float, channels: int, model=None, rebuild_head=True):
    if model is None:
        print('Building model')
        headless_model     = tf.keras.applications.ResNet50V2(**model_params)
    else:
        headless_model = model.layers[1]

    if rebuild_head or model is None:
        headless_model = tf.keras.Model(headless_model.input, headless_model.layers[-2].output)
        model_input    = tf.keras.Input(shape=(*config.target_size, channels))
        model          = headless_model(model_input, training=False)
        model          = tf.keras.layers.GlobalAveragePooling2D()(model)
        if config.num_dropout_layers>0:
            model = tf.keras.layers.Dropout(dropout_rate)(model)

        for i in range(len(config.head_layer_units)):
            model     = tf.keras.layers.Dense(config.head_layer_units[i], kernel_regularizer=tf.keras.regularizers.l2(config.kernel_l2))(model)
            if config.num_dropout_layers > i+1:
                model = tf.keras.layers.Dropout(dropout_rate)(model)
            model     = tf.keras.layers.ReLU()(model)              
        model_output = tf.keras.layers.Dense(config.num_classes, kernel_regularizer=tf.keras.regularizers.l2(config.kernel_l2))(model)
        model = tf.keras.Model(model_input, model_output)


    model_name     = 'ResNet50_pretrained'
    headless_model.trainable = True #
    if config.frozen_layers:
        for l in headless_model.layers[config.frozen_layers[0]:config.frozen_layers[-1]]:
            l.trainable = False

    if config.frozen_top_layers:
        for l in model.layers[config.frozen_top_layers[0]:config.frozen_top_layers[-1]]:
            l.trainable = False

    if config.freeze_bnorm_layers:
        for l in headless_model.layers[0:-1]:
            if 'bn' in l.name:
                l.trainable = False
    #region
    # model_input    = tf.keras.Input(shape=(*config.target_size, channels))
    # model          = headless_model(model_input)#, training=False)
    # model          = tf.keras.layers.GlobalAveragePooling2D()(model)
    # if config.num_dropout_layers>0:
    #     model = tf.keras.layers.Dropout(dropout_rate)(model)

    # for i in range(len(config.head_layer_units)):
    #     model     = tf.keras.layers.Dense(config.head_layer_units[i], kernel_regularizer=tf.keras.regularizers.l2(config.kernel_l2))(model)
    #     if config.num_dropout_layers > i+1:
    #         model = tf.keras.layers.Dropout(dropout_rate)(model)
    #     model     = tf.keras.layers.ReLU()(model)              
    # model_output = tf.keras.layers.Dense(config.num_classes, kernel_regularizer=tf.keras.regularizers.l2(config.kernel_l2))(model)
    # model = tf.keras.Model(model_input, model_output)
    #endregion
    metrics = get_metrics(config.metrics, num_classes=config.num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.warmup_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=metrics)
    # models.append(model)
    model.summary()
    
    return model, model_name

#region
# config = OmegaConf.create({'seed':49, #237,
#                            'target_size':(768,768), #(1024,1024),#(512,512),
#                            'resize_mode':'smart_resize',
#                            'resize_buffer_size':128,
#                            'channels':3,
#                            'batch_size':10,
#                            'augmentations':{'flip':1.0},#,'rotate':1.0},
#                            'num_parallel_calls':-1,
#                            'fit_class_weights':False,
#                            'num_epochs':150,
#                            'base_lr':8e-4,
#                            'warmup_learning_rate':1e-04,
#                            'lr_attack':6,
#                            'lr_sustain':1,
#                            'kernel_l2':1e-3,
#                            'dropout_rate':0.4,
#                            'num_dropout_layers':3,
#                            'head_layer_units':[1024,512], #[512,256],
#                            'dataset_name': 'PNAS', #'Leaves-PNAS',#
#                            'target_class_population':False, #100,
#                            'threshold': 100,
#                            'test_size': 0.5,
#                            'validation_split':0.1,
#                            'use_tfrecords':True,
#                            'samples_per_shard':300,
#                            'frozen_layers':False, #(0,-1)
#                            'metrics':['f1','accuracy','top-3_accuracy','balanced_accuracy']}#,'precision','recall']}
#                          )

# config = OmegaConf.create({'seed':49, #237,
#                            'target_size':(768,768), #(1024,1024),#(512,512),
#                            'resize_mode':'smart_resize_image',
#                            'resize_buffer_size':64, #128,
#                            'channels':3,
#                            'batch_size':10,
#                            'augmentations':{'flip':1.0,'rotate':1.0},
#                            'num_parallel_calls':-1,
#                            'fit_class_weights':False,
#                            'num_epochs':150,
#                            'base_lr':6e-4,
#                            'warmup_learning_rate':2e-04,
#                            'lr_attack':8,
#                            'lr_sustain':1,
#                            'kernel_l2':1e-3,
#                            'dropout_rate':0.3,
#                            'num_dropout_layers':1,
#                            'head_layer_units':[512,256], #[512,256],
#                            'dataset_name': 'Leaves-PNAS',# 'PNAS', #
#                            'target_class_population':False, #100,
#                            'threshold': 100,
#                            'test_size': 0.3,
#                            'validation_split':0.0,
#                            'use_tfrecords':True,
#                            'samples_per_shard':300,
#                            'frozen_layers':False, #(0,-1)
#                            'metrics':['f1','accuracy','top-3_accuracy','balanced_accuracy']}#,'precision','recall']}
#                          )
#endregion

from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from pyleaves.pipelines.WandB_Leaves_vs_PNAS import *
from tensorflow.keras.applications.resnet_v2 import preprocess_input


def hash_config(config: DictConfig) -> str:
    """Convert DictConfig to unique hash string using sha1
    """
    hash_key = hashlib.sha1()
    hash_key.update(config.pretty().encode('utf-8'))
    print(hash_key.digest())
    return str(hash_key.digest())

def load_data_splits(config, run=None):
    if config.dataset_name == "Leaves-PNAS":
        print('Loading Leaves-PNAS dataset for train/val, and loading PNAS_test for test')
        train_df, val_df = load_dataset_from_artifact(dataset_name=config.dataset_name, threshold=config.threshold, test_size=config.test_size, version='latest', artifact_name=None, run=run)
        pnas_train_df, test_df = load_dataset_from_artifact(dataset_name='PNAS', threshold=100, test_size=0.5, version='latest', run=run)

        if config.combined_train:
            train_df = pd.concat([train_df, pnas_train_df])

    else:
        print(f'Loading {config.dataset_name} dataset for train and test, with train set split into {1-config.validation_split}:{config.validation_split} train:val subsets.')
        train_df, test_df = load_dataset_from_artifact(dataset_name=config.dataset_name, threshold=config.threshold, test_size=config.test_size, version='latest', run=run)
        train_df, val_df = train_test_split(train_df, test_size=config.validation_split, random_state=config.seed, shuffle=True, stratify=train_df.family)

    if config.target_class_population:
        train_df = random_resample_dataset(data=train_df, y_col=config.label_type, target_class_population=config.target_class_population, set_class_floor=config.set_class_floor, set_class_ceil=config.set_class_ceil)
    
    return train_df, val_df, test_df


def get_config(cli_args=None, **kwargs):

    base_config = OmegaConf.create({'model_name':'resnet50v2',
                                    'model_weights':None, #'imagenet',
                                    'frozen_layers':None, #(0,-1)
                                    'frozen_top_layers':None, #(0,-3),
                                    'freeze_bnorm_layers':True,
                                    'label_type':'family',
                                    'seed':49, #237,
                                    'target_size':(768,768), #(1024,1024),#(512,512),
                                    'resize_mode':'smart_resize_image',
                                    'resize_buffer_size':64, #128,
                                    'channels':3,
                                    'batch_size':10,
                                    'augmentations':{'flip':1.0,'rotate':1.0},
                                    'num_parallel_calls':-1,
                                    'fit_class_weights':False,
                                    'num_epochs':150,
                                    'base_lr':1e-3,
                                    'warmup_learning_rate':3e-04,
                                    'lr_attack':8,
                                    'lr_sustain':1,
                                    'kernel_l2':1e-3,
                                    'dropout_rate':0.3,
                                    'num_dropout_layers':1,
                                    'head_layer_units':[1024,512], #[512,256],
                                    'target_class_population':False, #100,
                                    'threshold':None,
                                    'set_class_floor':None,
                                    'set_class_ceil':None,
                                    'use_tfrecords':True,
                                    'samples_per_shard':300,
                                    'combined_train':False,
                                    'metrics':['f1','accuracy','top-3_accuracy','balanced_accuracy'],
                                    'WarmUpCosineDecayScheduler':True,
                                    'early_stopping_patience':5,
                                    'run_id':None,
                                    'group':None,
                                    'tags':[f'{k}:{v}' for k,v in kwargs.items()]}#,'precision','recall']}
                                 )

    config = OmegaConf.merge(base_config, OmegaConf.create(kwargs))
    if cli_args is not None:
        config = OmegaConf.merge(config, cli_args)
        
    if config.WarmUpCosineDecayScheduler:
        if config.num_epochs <= config.lr_attack + config.lr_sustain:
            config.lr_attack = config.num_epochs - 1 - config.lr_sustain
            if not config.lr_attack+config.lr_sustain > config.num_epochs:
                config.lr_sustain=0

    if 'dataset_name' not in config:
        config['dataset_name'] = 'Leaves-PNAS'
    config.tags.append(config['dataset_name'])
    if config['dataset_name'] == 'Leaves-PNAS':
        config.dataset_name = config['dataset_name']
        config.threshold = 4
        config.test_size = 0.3
        config.validation_split = 0.0
    elif config['dataset_name'] == 'PNAS':
        config.dataset_name = config['dataset_name']
        config.threshold = 100
        config.test_size = 0.5
        config.validation_split = 0.1
        config.augmentations = {'flip':1.0}
    elif config['dataset_name'] == 'Fossil':
        config.dataset_name = config['dataset_name']
        config.threshold = config.threshold or 100
        config.test_size = 0.3
        config.validation_split = 0.1
    elif config['dataset_name'] == 'Leaves':
        config.dataset_name = config['dataset_name']
        config.threshold = config.threshold or 100
        config.test_size = 0.3
        config.validation_split = 0.1
        
    print(OmegaConf.to_yaml(config))

    config.tfrecord_dir = f'/media/data/jacob/tfrecords/{config.dataset_name}/{config.label_type}'
    if config.dataset_name=="Leaves-PNAS":
        if config.combined_train:
            config.tfrecord_dir += '/with_PNAS_train'
        else:
            config.tfrecord_dir += '/without_PNAS_train'

    if config.target_class_population:
        config.tfrecord_dir += f'_resampled-{config.target_class_population}'
    if config.set_class_floor:
        config.tfrecord_dir += f'_oversampled-{config.set_class_floor}'
    if config.set_class_ceil:
        config.tfrecord_dir += f'_undersampled-{config.set_class_ceil}'

    os.makedirs(config.tfrecord_dir, exist_ok=True)    
    trial_id = hash_config(config)
    config.trial_id = trial_id

    config.model_path = f'{config.model_name}-weights_{config.model_weights}-{config.dataset_name}_{config.target_size[0]}'
    config.class_label_path = f'{config.dataset_name}-class_labels.csv'
    return config


from tensorflow.keras.applications.resnet_v2 import preprocess_input

def load_trainvaltest_data(config, run=None):
    train_df, val_df, test_df = load_data_splits(config, run=run)
    train_data_info = data_df_2_tf_data(train_df,
                                        x_col='archive_path',
                                        y_col=config.label_type,
                                        training=True,
                                        preprocess_input=preprocess_input,
                                        seed=config.seed,
                                        target_size=config.target_size,
                                        resize_mode=config.resize_mode,
                                        resize_buffer_size=config.resize_buffer_size,
                                        batch_size=config.batch_size,
                                        augmentations=config.augmentations,
                                        num_parallel_calls=config.num_parallel_calls,
                                        cache=False,
                                        shuffle_first=True,
                                        fit_class_weights=config.fit_class_weights,
                                        subset_key='train',
                                        use_tfrecords=config.use_tfrecords,
                                        samples_per_shard=config.samples_per_shard,
                                        tfrecord_dir=config.tfrecord_dir)

    val_data_info = data_df_2_tf_data(val_df,
                                      x_col='archive_path',
                                      y_col=config.label_type,
                                      training=False,
                                      preprocess_input=preprocess_input,
                                      seed=config.seed,
                                      target_size=config.target_size,
                                      resize_mode=config.resize_mode,
                                      resize_buffer_size=config.resize_buffer_size,
                                      batch_size=config.batch_size,
                                      num_parallel_calls=config.num_parallel_calls,
                                      cache=True,
                                      shuffle_first=True,
                                      class_encodings=train_data_info['encoder'],
                                      subset_key='val',
                                      use_tfrecords=config.use_tfrecords,
                                      samples_per_shard=config.samples_per_shard,
                                      tfrecord_dir=config.tfrecord_dir)

    test_data_info = data_df_2_tf_data(test_df,
                                       x_col='archive_path',
                                       y_col=config.label_type,
                                       training=False,
                                       preprocess_input=preprocess_input,
                                       seed=config.seed,
                                       target_size=config.target_size,
                                       resize_mode=config.resize_mode,
                                       resize_buffer_size=config.resize_buffer_size,
                                       batch_size=config.batch_size,
                                       num_parallel_calls=config.num_parallel_calls,
                                       cache=True,
                                       shuffle_first=True,
                                       class_encodings=train_data_info['encoder'],
                                       subset_key='test',
                                       use_tfrecords=config.use_tfrecords,
                                       samples_per_shard=config.samples_per_shard,
                                       tfrecord_dir=config.tfrecord_dir)

    return train_data_info, val_data_info, test_data_info


def get_callbacks(config, initial_epoch=0, train_data=None, val_data=None, test_data=None, class_labels=None):

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                    patience=4, min_lr=1e-5, verbose=True)]


    if config.WarmUpCosineDecayScheduler:

        num_samples_train = config.num_samples.train
        total_steps = int(config.num_epochs * num_samples_train / config.batch_size) # total iteration batch steps
        warmup_steps = int(config.lr_attack * num_samples_train / config.batch_size) # total number of warmup batches
        warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=config.base_lr,
                                                total_steps=total_steps,
                                                global_step_init=initial_epoch,
                                                warmup_learning_rate=config.warmup_learning_rate,
                                                warmup_steps=warmup_steps,
                                                hold_base_rate_steps=config.lr_sustain,
                                                verbose=1
                                                )
        callbacks.append(warm_up_lr)

    early_stop = EarlyStopping(monitor='val_loss',
                  patience=config.early_stopping_patience,
                  min_delta=1e-5, 
                  verbose=1,
                  restore_best_weights=True)
    callbacks.append(early_stop)

    visualize_train_data = [(batch[0].numpy(), batch[1].numpy()) for batch in next(iter(train_data.take(1)))][0]
    visualize_test_data = ((batch[0].numpy(), batch[1].numpy()) for batch in next(iter(test_data.take(10))))

    wandb_cb = WandbCallback(save_model=True,
                            monitor='val_loss',
                            training_data=visualize_train_data,
                            data_type='image',
                            labels=class_labels,
                            predictions=64,
                            generator = visualize_test_data)
    callbacks.append(wandb_cb)
    return callbacks







def fit_one_cycle(config, model=None, run=None, initial_epoch=None, rebuild_head=True):

    # WANDB_CREDENTIALS = {"entity":"jrose",
    #                      "project":"Leaves_vs_PNAS",
    #                      "dir":"/media/data_cifs_lrs/projects/prj_fossils/users/jacob/WandB_artifacts"}
    # run = wandb.init(**WANDB_CREDENTIALS, id=config.run_id, tags=config.tags, resume="allow", reinit=True)
    # config.run_id = run.id
    if run is None:
        run = init_wandb_run(config)
    ###########################################   
    #region
    # train_df, val_df, test_df = load_data_splits(config, run=run)
    # train_data_info = data_df_2_tf_data(train_df,
    #                                     x_col='archive_path',
    #                                     y_col='family',
    #                                     training=True,
    #                                     preprocess_input=preprocess_input,
    #                                     seed=config.seed,
    #                                     target_size=config.target_size,
    #                                     resize_mode=config.resize_mode,
    #                                     resize_buffer_size=config.resize_buffer_size,
    #                                     batch_size=config.batch_size,
    #                                     augmentations=config.augmentations,
    #                                     num_parallel_calls=config.num_parallel_calls,
    #                                     cache=False,
    #                                     shuffle_first=True,
    #                                     fit_class_weights=config.fit_class_weights,
    #                                     subset_key='train',
    #                                     use_tfrecords=config.use_tfrecords,
    #                                     samples_per_shard=config.samples_per_shard,
    #                                     tfrecord_dir=config.tfrecord_dir)

    # val_data_info = data_df_2_tf_data(val_df,
    #                                   x_col='archive_path',
    #                                   y_col='family',
    #                                   training=False,
    #                                   preprocess_input=preprocess_input,
    #                                   seed=config.seed,
    #                                   target_size=config.target_size,
    #                                   resize_mode=config.resize_mode,
    #                                   resize_buffer_size=config.resize_buffer_size,
    #                                   batch_size=config.batch_size,
    #                                   num_parallel_calls=config.num_parallel_calls,
    #                                   cache=True,
    #                                   shuffle_first=True,
    #                                   class_encodings=train_data_info['encoder'],
    #                                   subset_key='val',
    #                                   use_tfrecords=config.use_tfrecords,
    #                                   samples_per_shard=config.samples_per_shard,
    #                                   tfrecord_dir=config.tfrecord_dir)

    # test_data_info = data_df_2_tf_data(test_df,
    #                                    x_col='archive_path',
    #                                    y_col='family',
    #                                    training=False,
    #                                    preprocess_input=preprocess_input,
    #                                    seed=config.seed,
    #                                    target_size=config.target_size,
    #                                    resize_mode=config.resize_mode,
    #                                    resize_buffer_size=config.resize_buffer_size,
    #                                    batch_size=config.batch_size,
    #                                    num_parallel_calls=config.num_parallel_calls,
    #                                    cache=True,
    #                                    shuffle_first=True,
    #                                    class_encodings=train_data_info['encoder'],
    #                                    subset_key='test',
    #                                    use_tfrecords=config.use_tfrecords,
    #                                    samples_per_shard=config.samples_per_shard,
    #                                    tfrecord_dir=config.tfrecord_dir)
    #endregion

    ###########################################
    # DATA #
    ###########################################
    train_data_info, val_data_info, test_data_info = load_trainvaltest_data(config, run=run)
    encoder = train_data_info['encoder']
    class_weights = train_data_info['class_weights']
    class_labels = list(train_data_info['encoder'])
    train_iter = train_data_info['data']
    val_iter = val_data_info['data']
    test_iter = test_data_info['data']

    train_data = train_iter.unbatch().map(lambda x,y,_: (x,y))
    val_data = val_iter.map(lambda x,y,_: (x,y))#.repeat()
    test_data = test_iter.map(lambda x,y,_: (x,y))
    if 'cutmixup' in config.augmentations:
        print('Applying cutmixup image augmentations to train data')
        train_data = apply_cutmixup(dataset=train_data, aug_batch_size=config.batch_size, num_classes=config.num_classes, target_size=config.target_size, batch_size=config.batch_size)
    else:
        train_data = train_data.batch(config.batch_size)

    config.num_samples = {}
    config.num_samples.train = train_data_info['num_samples']
    config.num_samples.val = val_data_info['num_samples']
    config.num_samples.test = test_data_info['num_samples']
    config.num_classes = train_data_info['num_classes']
    steps_per_epoch=int(np.ceil(config.num_samples.train/config.batch_size))
    validation_steps=int(np.ceil(config.num_samples.val/config.batch_size))
    # test_steps=int(np.ceil(config.num_samples.test/config.batch_size))

    ###########################################
    # MODEL #
    ###########################################
    model_params = {'input_shape' : (*config.target_size, config.channels), 'include_top': False, 'weights':config.model_weights}
    model, _ = build_model(model_params, config=config, dropout_rate=config.dropout_rate, channels=config.channels, model=model, rebuild_head=rebuild_head)

    histories = []
    if initial_epoch is None:
        initial_epoch = wandb.run.step or 0
    # if wandb.run.resumed:
    #     print(f'Restoring model from checkpoint at epoch {initial_epoch}')
    #     model = tf.keras.models.load_model(wandb.restore("model-best.h5").name)
    run.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)
    # if model.history is None:
    #     initial_epoch = initial_epoch or 0
    callbacks = get_callbacks(config,
                              initial_epoch=initial_epoch,
                              train_data=train_data, val_data=val_data, test_data=test_data,
                              class_labels=class_labels)
    history = model.fit(train_data,
                        epochs=config.num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_data,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch,
                        class_weight=class_weights,
                        callbacks=callbacks)

    histories.append(history)
    model.save(config.model_path)
    save_class_labels(class_labels=encoder, label_path=config.class_label_path)


    artifact = wandb.Artifact(type='model', name=config.model_path)
    if os.path.isfile(config.model_path):
        artifact.add_file(config.model_path, name=config.model_path)
    elif os.path.isdir(config.model_path):
        artifact.add_dir(config.model_path, name=config.model_path)

    artifact.add_file(config.class_label_path, name=str(Path(config.class_label_path).name))
    run.log_artifact(artifact)
    print('INITIATING MODEL EVALUATION ON TEST SET')
    test_data_info['data'] = test_data
    perform_evaluation_stage(model, test_data_info, class_encoder=encoder, batch_size=config.batch_size, subset='test')
    # try:
    #     run.join()
    # finally:
    #     print('Finished')
    
    return model


# model = load_model_artifact(config_1.load_model_artifact)



def init_wandb_run(config, group=None, resume="allow", reinit=True):
    WANDB_CREDENTIALS = {"entity":"jrose",
                         "project":"Leaves_vs_PNAS",
                         "dir":"/media/data_cifs_lrs/projects/prj_fossils/users/jacob/WandB_artifacts"}

    run = wandb.init(**WANDB_CREDENTIALS, group=group, id=config.run_id, tags=config.tags, resume=resume, reinit=reinit, allow_val_change=True)
    config.run_id = run.id
    return run

def random_initialization_trial(cli_args=None):
    model_weights = None
    K.clear_session()
    print(f'Beginning training from scratch')
    config = get_config(model_weights=model_weights, frozen_layers=None, num_epochs=100, cli_args=cli_args)

    run = init_wandb_run(config)
    model = fit_one_cycle(config, run=run)
    model.save(config.model_path+'_final')

    run.join()
    return model


def finetune_trial(cli_args=None):
    cli_args = cli_args or {}
    model_weights = 'imagenet'

    if 'frozen_layer_sequence' in cli_args:
        frozen_layer_sequence = cli_args.pop('frozen_layer_sequence')
    else:
        frozen_layer_sequence = [-1, -4, -12]

    if 'num_epochs_sequence' in cli_args:
        num_epochs_sequence = cli_args.pop('num_epochs_sequence')
    else:
        num_epochs_sequence = [75,75,75]

    K.clear_session()
    initial_epoch = 0
    print(f'Beginning stage 1 of finetune trial')
    default_kwargs = OmegaConf.create(dict(model_weights=model_weights, frozen_layers=(0,frozen_layer_sequence[0]), 
                                      num_epochs=num_epochs_sequence[0], WarmUpCosineDecayScheduler=False))
    kwargs = OmegaConf.merge(default_kwargs, cli_args)
    config_1 = get_config(**kwargs, cli_args=cli_args)

    # if config_1.load_model_artifact:
    #     model = load_model_artifact(config_1.load_model_artifact)

    run = init_wandb_run(config_1, group=config_1.group)
    model = fit_one_cycle(config_1, run=run, initial_epoch=initial_epoch)

    print(f'Beginning stage 2 of finetune trial')
    initial_epoch += config_1.num_epochs
    config_2 = get_config(warmup_learning_rate=config_1.warmup_learning_rate/2, model_weights=model_weights, 
                          frozen_layers=(0,frozen_layer_sequence[1]), head_layer_units=config_1.head_layer_units,
                          num_epochs=initial_epoch+num_epochs_sequence[1], cli_args=cli_args)
    # run = init_wandb_run(config_2, group=config_2.group)
    model = fit_one_cycle(config_2, model=model, run=run, initial_epoch=initial_epoch, rebuild_head=False)

    print(f'Beginning stage 3 of finetune trial')
    initial_epoch += config_2.num_epochs
    config_3 = get_config(warmup_learning_rate=config_2.warmup_learning_rate/2, model_weights=model_weights,
                          frozen_layers=(0,frozen_layer_sequence[2]), head_layer_units=config_2.head_layer_units, 
                          num_epochs=initial_epoch+num_epochs_sequence[2], cli_args=cli_args)
    # run = init_wandb_run(config_3, group=config_3.group)
    model = fit_one_cycle(config_3, model=model, run=run, initial_epoch=initial_epoch, rebuild_head=False)

    model.save(config_3.model_path+'_final')
    run.join()
    return model


def Leaves_finetune_trials(cli_args=None):
    cli_args = cli_args or {}
    model_weights = 'imagenet'

    if 'frozen_layer_sequence' in cli_args:
        frozen_layer_sequence = cli_args.pop('frozen_layer_sequence')
    else:
        frozen_layer_sequence = [-36, -12]

    if 'num_epochs_sequence' in cli_args:
        num_epochs_sequence = cli_args.pop('num_epochs_sequence')
    else:
        num_epochs_sequence = [120,100]

    K.clear_session()
    initial_epoch = 0
    print(f'Beginning stage 1 of finetune trial on Leaves-PNAS')
    default_kwargs = OmegaConf.create(dict(dataset_name="Leaves-PNAS", model_weights=model_weights, frozen_layers=(0,frozen_layer_sequence[0]), 
                                      num_epochs=num_epochs_sequence[0], WarmUpCosineDecayScheduler=False))
    kwargs = OmegaConf.merge(default_kwargs, cli_args)
    config_1 = get_config(**kwargs, cli_args=cli_args)

    run = init_wandb_run(config_1, group=config_1.group)
    model = fit_one_cycle(config_1, run=run, initial_epoch=initial_epoch)

    print(f'Beginning stage 2 of finetune trial on PNAS')
    initial_epoch += config_1.num_epochs
    config_2 = get_config(dataset_name="PNAS", warmup_learning_rate=config_1.warmup_learning_rate/2, model_weights=model_weights, 
                          frozen_layers=(0,frozen_layer_sequence[1]), head_layer_units=config_1.head_layer_units,
                          num_epochs=initial_epoch+run.step, cli_args=cli_args)
    model = fit_one_cycle(config_2, model=model, run=run, initial_epoch=initial_epoch)

    model.save(config_2.model_path+'_final')
    run.join()
    return model



if __name__=='__main__':
    # for model_weights in ['imagenet', None]:
    #     K.clear_session()
    #     config = get_config(dataset_name='Leaves-PNAS', model_weights=model_weights, frozen_layers=None, head_layer_units=[1024,512])
    #     model = fit_one_cycle(config)
    import sys
    cli_args = OmegaConf.from_cli()
    if '--random_initialization_trial' in sys.argv:
        random_initialization_trial(cli_args)

    if '--finetune_imagenet' in sys.argv:
        finetune_trial(cli_args)

    if '--finetune_Leaves' in sys.argv:
        Leaves_finetune_trials(cli_args)


# %%
