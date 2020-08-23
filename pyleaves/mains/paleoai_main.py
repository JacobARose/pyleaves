# @Author: Jacob A Rose
# @Date:   Tue August 18th 2020, 11:53 pm
# @Email:  jacobrose@brown.edu
# @Filename: paleoai_main.py


'''
Script built off of configurable_train_pipeline.py


python '/home/jacob/projects/pyleaves/pyleaves/mains/paleoai_main.py'

'''
import arrow
from box import Box
from collections import OrderedDict
from functools import partial
from stuf import stuf
import copy
from datetime import datetime, timedelta
import hydra
from more_itertools import unzip
import neptune
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
from pprint import pprint
from pathlib import Path
from typing import List
import pyleaves
from pyleaves.datasets.base_dataset import BaseDataset
from pyleaves.models import resnet, vgg16
from pyleaves.utils.callback_utils import BackupAndRestore
from pyleaves.utils import setGPU, set_tf_config
from pyleaves.datasets import leaves_dataset, fossil_dataset, pnas_dataset, base_dataset
from pyleaves.utils import ensure_dir_exists, img_aug_utils as iau
from tfrecord_utils.encoders import TFRecordCoder
from paleoai_data.utils.kfold_cross_validation import DataFold
from paleoai_data.utils.kfold_cross_validation import generate_KFoldDataset, export_folds_to_csv, KFoldLoader #, prep_dataset


from paleoai_data.utils.multiprocessing_utils import RunAsCUDASubprocess
##########################################################################
##########################################################################
CONFIG_DIR = str(Path(pyleaves.RESOURCES_DIR,'..','..','configs','hydra'))
date_format = '%Y-%m-%d_%H-%M-%S'


import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_logical_devices('GPU')







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
    cfg['stage_0']['model_dir'] = cfg.model_dir #os.path.join(cfg.log_dir,'model_dir')
    cfg.stage_0.update(tfrecord_dir = os.path.join(cfg.log_dir,'tfrecord_dir'))
    cfg.update(tfrecord_dir = cfg.stage_0.tfrecord_dir)
    cfg.saved_model_path = str(Path(cfg.model_dir) / Path('saved_model'))
    cfg.checkpoints_path = str(Path(cfg.model_dir) / Path('checkpoints'))
    cfg['stage_0']['checkpoints_path'] = cfg.checkpoints_path
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



def log_config(cfg: DictConfig, neptune, verbose: bool=False):
    if verbose: print(cfg.pretty())

    cfg_0 = cfg.stage_0
    ensure_dir_exists(cfg['log_dir'])
    ensure_dir_exists(cfg['model_dir'])
    neptune.append_tag(cfg_0.dataset.dataset_name)
    neptune.append_tag(cfg_0.model.model_name)
    neptune.append_tag(str(cfg_0.dataset.target_size))
    neptune.append_tag(cfg_0.dataset.num_channels)
    neptune.append_tag(cfg_0.dataset.color_mode)


def log_dataset(cfg: DictConfig, train_dataset: BaseDataset, test_dataset: BaseDataset, neptune):
    cfg['dataset']['num_classes'] = train_dataset.num_classes
    cfg['dataset']['splits_size'] = {'train':{},
                          'test':{}}
    cfg['dataset']['splits_size']['train'] = int(train_dataset.num_samples)
    cfg['dataset']['splits_size']['test'] = int(test_dataset.num_samples)

    cfg['steps_per_epoch'] = cfg['dataset']['splits_size']['train']//cfg['training']['batch_size']
    cfg['validation_steps'] = cfg['dataset']['splits_size']['test']//cfg['training']['batch_size']

    neptune.set_property('num_classes',cfg['num_classes'])
    neptune.set_property('steps_per_epoch',cfg['steps_per_epoch'])
    neptune.set_property('validation_steps',cfg['validation_steps'])


def get_model_config(cfg: DictConfig):
    cfg['model']['base_learning_rate'] = cfg['lr']
    cfg['model']['input_shape'] = (*cfg.dataset['target_size'],cfg.dataset['num_channels'])
    cfg['model']['model_dir'] = cfg['model_dir']
    cfg['model']['num_classes'] = cfg['dataset']['num_classes']
    model_config = OmegaConf.merge(cfg.model, cfg.training)
    return model_config


@RunAsCUDASubprocess()
def train_single_fold(fold: DataFold, cfg : DictConfig, neptune,  verbose: bool=True) -> None:
    import tensorflow as tf
    set_tf_config()
    preprocess_input(tf.zeros([4, 224, 224, 3]))
    from tensorflow.keras import backend as K
    K.clear_session()

    
    cfg.tfrecord_dir = os.path.join(cfg.tfrecord_dir,fold.fold_name)
    ensure_dir_exists(cfg.tfrecord_dir)
    if verbose:
        print('='*20)
        print(f'RUNNING: fold {fold.fold_id}')
        print(cfg.tfrecord_dir)
        print('='*20)
    
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
    log_dataset(cfg=cfg, train_dataset=train_dataset, test_dataset=test_dataset, neptune=neptune)

    # cfg['model']['base_learning_rate'] = cfg['lr']
    # cfg['model']['input_shape'] = (*cfg.dataset['target_size'],cfg.dataset['num_channels'])
    # cfg['model']['model_dir'] = cfg['model_dir']
    # cfg['model']['num_classes'] = cfg['dataset']['num_classes']
    # model_config = OmegaConf.merge(cfg.model, cfg.training)
    model_config = get_model_config(cfg=cfg)

    gpu_device = setGPU(only_return=True)
    with tf.Graph.as_default():
        with tf.device(gpu_device.name): #.strip('/physical_device:')):
            model = build_model(model_config)

    model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
    pprint(cfg)

    from pyleaves.utils.neptune_utils import ImageLoggerCallback

    backup_callback = BackupAndRestore(cfg['checkpoints_path'])
    backup_callback.set_model(model)
    neptune_logger = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: log_data(logs, neptune),
                                                   on_epoch_end=lambda epoch, logs: log_data(logs, neptune))
    callbacks = [neptune_logger,
                backup_callback,
                tf.keras.callbacks.CSVLogger(cfg.log_dir, separator=',', append=False),
                EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True),
                ImageLoggerCallback(data=train_data, freq=1000, max_images=-1, name='train', encoder=encoder, neptune_logger=neptune),
                ImageLoggerCallback(data=test_data, freq=1000, max_images=-1, name='val', encoder=encoder, neptune_logger=neptune)]

    # import pdb; pdb.set_trace()
    history = model.fit(train_data,
                        epochs=cfg.training['num_epochs'],
                        callbacks=callbacks,
                        validation_data=test_data,
                        shuffle=True,
                        steps_per_epoch=cfg['steps_per_epoch'],
                        validation_steps=cfg['validation_steps'])
    return history.history


# from keras.wrappers.scikit_learn import KerasClassifier
# from tune_sklearn import TuneGridSearchCV
# from joblib import Parallel, delayed


    

# def train_paleoai_dataset(cfg : DictConfig, fold_ids: List[int]=[0], n_jobs: int=1, verbose: bool=False) -> None:

#     histories = []
#     def log_history(history: dict):
#         try:
#             histories.append(history)
#         except:
#             histories.append(None)


#     cfg_0 = cfg.stage_0
#     # cfg_1 = cfg.stage_1
#     log_config(cfg=cfg, verbose=verbose)
#     kfold_loader = KFoldLoader(root_dir=cfg_0.dataset.fold_dir)
#     kfold_iter = kfold_loader.iter_folds(repeats=1)
#     # histories = Parallel(n_jobs=n_jobs)(delayed(train_single_fold)(fold=fold, cfg=copy.deepcopy(cfg_0), gpu_device=gpus[i]) for i, fold in enumerate(kfold_iter) if i < n_jobs)

#     print(f'Beginning training of models with fold_ids: {fold_ids}')
#     with Pool(processes=n_jobs) as pool:
#         for i, fold in enumerate(kfold_iter):
#             if i in fold_ids:
#                 history = pool.apply_async(train_single_fold, 
#                                            args=(fold, copy.deepcopy(cfg_0)),
#                                            callback = log_history)#, gpu_device=gpus[0])
#                 fold_ids.pop(i)
#             if len(fold_ids)==0:
#                 break

#     return history


# def train_paleoai_dataset(cfg : DictConfig, fold_ids: List[int]=[0], n_jobs: int=1, verbose: bool=False) -> None:

#     cfg_0 = cfg.stage_0
#     # cfg_1 = cfg.stage_1

#     log_config(cfg=cfg, verbose=verbose)

#     kfold_loader = KFoldLoader(root_dir=cfg_0.dataset.fold_dir)
#     kfold_iter = kfold_loader.iter_folds(repeats=1)
#     # histories = Parallel(n_jobs=n_jobs)(delayed(train_single_fold)(fold=fold, cfg=copy.deepcopy(cfg_0), gpu_device=gpus[i]) for i, fold in enumerate(kfold_iter) if i < n_jobs)
#     history = Box({'history':[]})
#     print(f'Beginning training of models with fold_ids: {fold_ids}')
#     for i, fold in enumerate(kfold_iter):
#         if i in fold_ids:
#             history = train_single_fold(fold=fold, cfg=copy.deepcopy(cfg_0))#, gpu_device=gpus[0])
#             fold_ids.pop(i)
#         if len(fold_ids)==0:
#             break

#     return history



# @hydra.main(config_path=Path(CONFIG_DIR,'Leaves-PNAS.yaml'))
# def train(cfg : DictConfig) -> None:

#     OmegaConf.set_struct(cfg, False)
#     cfg = restore_or_initialize_experiment(cfg, restore_last=True, prefix='log_dir__', verbose=2)

#     neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
#     params=OmegaConf.to_container(cfg)
#     with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name), params=params):
#         # train_pyleaves_dataset(cfg)

#         train_paleoai_dataset(cfg=cfg, n_jobs=1, verbose=True)



# if __name__=='__main__':

#     train()
