'''
Created on Thu Mar 5 22:51:53 2020

script: pyleaves/pyleaves/train/transfer_trainer.py

Script for defining a class to manage a multi-stage training process for transfer learning.



##TBD Currently this script is justt a copy of BaseTrainer


@author: JacobARose
'''

import dataset
import datetime
from functools import partial
import numpy as np
import os
import pandas as pd

gpu_ids = '0,1'
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
from pyleaves.utils import ensure_dir_exists, get_visible_devices

get_visible_devices('GPU')


import pyleaves
from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, generate_encoding_map
from pyleaves import leavesdb
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves.analysis.img_utils import TFRecordCoder, plot_image_grid, imagenet_mean_subtraction, ImageAugmentor, get_keras_preprocessing_function, rgb2gray_3channel, rgb2gray_1channel
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata
from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords

from pyleaves.config import DatasetConfig, TrainConfig, ModelConfig, ExperimentConfig
from pyleaves.train.base_trainer import BaseTrainer
from pyleaves.models.resnet import ResNet, ResNetGrayScale
from pyleaves.models.vgg16 import VGG16, VGG16GrayScale
from stuf import stuf




        
        
        
        
    
class TransferTrainer:
    
    def __init__(self, experiment_configs=[], model_builder=None, src_db=pyleaves.DATABASE_PATH):
        self.model_builder = model_builder
        self.num_domains = len(experiment_configs) # Number of pipeline stages

        self.domains = {'source':BaseTrainer(experiment_configs[0], src_db=src_db)}
        encoder = self.domains['source'].encoder
        self.domains.update({'target':BaseTrainer(experiment_configs[1], label_encoder=encoder, src_db=src_db)})
        self.configs = {'source':experiment_configs[0],
                        'target':experiment_configs[1]}
        self.histories = {}
        
        
    def init_model_builder(self, domain='source',subset='train',num_classes=None):
#         import pdb; pdb.set_trace()
        model_config = self.get_model_config(domain=domain, subset=subset, num_classes=num_classes)
        self.model_name = model_config.model_name
        if self.model_name == 'vgg16':
            self.add_model_manager(VGG16GrayScale(model_config))
        elif self.model_name.startswith('resnet'):
            self.add_model_manager(ResNet(model_config))
        self.model = self.model_manager.build_model()
        
    def get_data_loader(self, domain='source', subset='train'):
        '''
        Get the proper trainer instance for specified domain ('source' or 'target'), then from that return the data loader for the specified subset ('train', 'val', or 'test').
        '''
        trainer = self.domains[domain]
        return trainer.get_data_loader(subset=subset)
    
    def get_model_config(self, domain='source', subset='train', num_classes=None):
        trainer = self.domains[domain]
        return trainer.get_model_config(subset=subset, num_classes=num_classes)

    def get_fit_params(self, domain='source'):
        trainer = self.domains[domain]
        return trainer.get_fit_params()
    
    def add_model_manager(self, model_manager):
        '''Simply for adding a subclass of BaseModel for trainer to keep track of, in order to
        extend model building/saving/loading/importing/exporting functionality to trainer.'''
        self.model_manager = model_manager
    
    def save_weights(self, filepath):
        self.model_manager.save_weights(filepath=filepath)
        
    def load_weights(self, filepath):
        self.model_manager.load_weights(filepath=filepath)        
        self.model = self.model_manager.model
        
    def save_model(self, filepath):
        self.model_manager.save(filepath=filepath)
        
    def load_model(self, filepath):
        self.model_manager.load(filepath=filepath)        
        self.model = self.model_manager.model
        
        
        
    

# from pyleaves.train.callbacks import get_callbacks
# import json
    
# model_name = 'vgg16'
# dataset_names = ['PNAS','Fossil']
# base_learning_rate=1e-4
# batch_size=128
# num_epochs=200
# regularizer={'l2':0.001}
# target_size =(224,224)    
# num_channels = 1
# color_type='grayscale'
# low_class_count_thresh=20
# tfrecord_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data'
# model_dir = r'/media/data_cifs/jacob/Fossil_Project/models'


# dataset_config_source_domain = DatasetConfig(dataset_name=dataset_names[0],
#                                 label_col='family',
#                                 target_size=target_size,
#                                 num_channels=num_channels,
#                                 grayscale=(color_type=='grayscale'),
#                                 low_class_count_thresh=low_class_count_thresh,
#                                 data_splits={'val_size':0.2,'test_size':0.0},
#                                 tfrecord_root_dir=tfrecord_dir,
#                                 num_shards=10)

# dataset_config_target_domain = DatasetConfig(dataset_name=dataset_names[1],
#                                 label_col='family',
#                                 target_size=target_size,
#                                 num_channels=num_channels,
#                                 grayscale=(color_type=='grayscale'),
#                                 low_class_count_thresh=low_class_count_thresh,
#                                 data_splits={'val_size':0.2,'test_size':0.2},
#                                 tfrecord_root_dir=tfrecord_dir,
#                                 num_shards=10)


# train_config = TrainConfig(model_name=model_name,
#                            model_dir=model_dir,
#                            batch_size=batch_size,
#                            frozen_layers=None,
#                            base_learning_rate=base_learning_rate,
#                            buffer_size=500,
#                            num_epochs=num_epochs,
#                            preprocessing=True,
#                            augment_images=True,
#                            augmentations=['rotate','flip'],
#                            regularization=regularizer,
#                            seed=5,
#                            verbose=True)

# current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project',
#                                           'experiments',
#                                           'domain_transfer',
#                                           '-'.join([model_name,color_type]),
#                                           '-'.join(dataset_names),
#                                           f'lr-{base_learning_rate}-bsz_{batch_size}',
#                                           current_time)


# configs = [ExperimentConfig(dataset_config_source_domain, train_config),
#            ExperimentConfig(dataset_config_target_domain, train_config)]



# trainer = TransferTrainer(configs)


# source_trainer = trainer.domains['source']
# # source_trainer.extract()
# # source_trainer.transform()
# # source_trainer.load()

# target_trainer = trainer.domains['target']
# # target_trainer.extract()
# # target_trainer.transform()
# # target_trainer.load()


# source_counts = source_trainer.get_class_counts()
# target_counts = target_trainer.get_class_counts()

# #Build Model

# model_config = trainer.get_model_config(domain='source',subset='train')

# if model_config.model_name is 'vgg16':
#     trainer.add_model_manager(VGG16GrayScale(model_config))
    
# elif model_config.model_name.startswith('resnet'):
#     trainer.add_model_manager(ResNet(model_config))

# trainer.model = trainer.model_manager.build_model()


# #Get source domain data

# train_data = trainer.get_data_loader(domain='source', subset='train')
# val_data = trainer.get_data_loader(domain='source', subset= 'val')

# #Get parameters for fitting and callbacks
# fit_params = trainer.get_fit_params(domain='source')#, subset='train')
# callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'source_domain_weights_best.h5'), 
#                               logs_dir=os.path.join(experiment_dir,'tensorboard_logs'), 
#                               restore_best_weights=True)


# print('model_config:\n',json.dumps(model_config,indent=4))

# # TRAIN ON SOURCE DOMAIN

# history = trainer.model.fit(train_data,
#                  steps_per_epoch = fit_params['steps_per_epoch'],
#                  epochs=1,#fit_params['epochs'],
#                  validation_data=val_data,
#                  validation_steps=fit_params['validation_steps'],
#                  callbacks=callbacks
#                  )


# # trainer.save_weights(filepath=trainer.model_manager.weights_filepath)
# trainer.save_model(filepath=os.path.join(trainer.model_manager.model_dir,model_config.model_name,'source_model.h5'))



# trainer.load_model(filepath=os.path.join(trainer.model_manager.model_dir,model_config.model_name,'source_model.h5'))


# target_train_data = trainer.get_data_loader(domain='target', subset='train')
# target_val_data = trainer.get_data_loader(domain='target', subset= 'val')
# target_test_data = trainer.get_data_loader(domain='target', subset='test')


# # model_config = trainer.get_model_config(domain='target',subset='train')
# fit_params = trainer.get_fit_params(domain='target')
# callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'target_domain_weights_best.h5'), 
#                               logs_dir=os.path.join(experiment_dir,'tensorboard_logs'), 
#                               restore_best_weights=True)

# num_test_samples = trainer.domains['target'].metadata_splits['test']['num_samples']
# num_steps = num_test_samples//batch_size

# zero_shot_test_results = trainer.model.evaluate(target_test_data, steps=num_steps)

# # FINETUNE ON TARGET DOMAIN

# history = trainer.model.fit(target_train_data,
#                  steps_per_epoch = fit_params['steps_per_epoch'],
#                  epochs=fit_params['epochs'],
#                  validation_data=target_val_data,
#                  validation_steps=fit_params['validation_steps'],
#                  callbacks=callbacks
#                  )











# train_config.save_config(os.path.expanduser('~/experiments/configs/experiment_0/train_config.json'))
# dataset_config.save_config(os.path.expanduser('~/experiments/configs/experiment_0/dataset_config.json'))
# config.save_config(os.path.expanduser('~/experiments/configs/experiment_0/experiment_config.json'))


# loaded_train_config = train_config.load_config(os.path.expanduser('~/experiments/configs/experiment_0/train_config.json'))
# loaded_dataset_config = dataset_config.load_config(os.path.expanduser('~/experiments/configs/experiment_0/dataset_config.json'))
# loaded_config = config.load_config(os.path.expanduser('~/experiments/configs/experiment_0/experiment_config.json'))


# a=set(train_config).symmetric_difference(loaded_train_config)

# list(loaded_train_config - train_config)
# list(loaded_train_config - dataset_config)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#########################################################################################################
#########################################################################################################
#########################################################################################################
    
    
class BaseTrainer_v1(BaseTrainer):
    '''
    Trainer class that uses a tf.compat.v1.Session() as well as using tf.data.Datasets with initializable syntax.
    
    Use regular BaseTrainer class for eager execution
    '''
    
    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        
        self.sess = tf.compat.v1.Session()    
    
    def get_data_loader(self, subset='train'):
        
        data = super().get_data_loader(subset=subset)
        
        print(f'returning non-eager {subset} tf.data.Dataset iterator')
        data_iterator = data.make_initializable_iterator()
        self.sess.run(data_iterator.initializer)
        
        return data_iterator
    
    
# class MLFlowTrainer(BaseTrainer):
    
#     '''
#     Subclass of BaseTrainer that uses mlflow.log_artifacts to log the exact tfrecord files used in experiment.
    
#     '''
#     def __init__(self, *args, **kwargs):
        
    

    
    
# class KerasTrainer(BaseTrain):
    
#     def __init__(self, experiment_config)

    

    

    
if __name__ == '__main__':

    dataset_config = DatasetConfig(dataset_name='PNAS',
                                   label_col='family',
                                   target_size=(224,224),
                                   channels=3,
                                   low_class_count_thresh=3,
                                   data_splits={'val_size':0.2,'test_size':0.2},
                                   tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
                                   num_shards=10)

    train_config = TrainConfig(model_name='vgg16',
                     batch_size=64,
                     frozen_layers=(0,-4),
                     base_learning_rate=1e-4,
                     buffer_size=1000,
                     num_epochs=100,
                     preprocessing='imagenet',
                     augment_images=True,
                     seed=3)
    
    
    
    experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                         train_config=train_config)

    trainer = BaseTrainer(experiment_config=experiment_config)

    ##LOAD AND PLOT 1 BATCH OF IMAGES AND LABELS FROM FOSSIL DATASET
    experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',trainer.config.model_name,trainer.config.dataset_name)

    train_data = trainer.get_data_loader(subset='train')
    val_data = trainer.get_data_loader(subset='val')
    test_data = trainer.get_data_loader(subset='test')
    

    for imgs, labels in train_data.take(1):
        labels = [trainer.label_encodings[np.argmax(label)] for label in labels.numpy()]
        imgs = (imgs.numpy()+1)/2
        plot_image_grid(imgs, labels, 4, 8)

    
