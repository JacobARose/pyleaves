'''
tfds_trainer.py

Script for implementing construction of a trainer object for managing experiment data and metadata for TFDS datasets.

# TODO Create a trainer that can correctly parse through the different types of labels on TFDS
# e.g. family, disease, species, 
# or that have some useful information outside of the label column
# e.g. Like having species in the name of the dataset with 'disease_type' for a label
# Purpose: So we can smoothly combine these datasets at run time in the same way we do with leavesdb

@author: JacobARose
'''


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name', default='tf_flowers', type=str, help='Name of dataset of images to use for creating TFRecords')
parser.add_argument('-m', '--model_name', default='vgg16', type=str, help='Name of model to train')
parser.add_argument('-gpu', '--gpu_id', default=0, type=int, help='integer number of gpu to train on')

parser.add_argument('-ch', '--num_channels', default=3, type=int, help='Number of input channels, either 1 for grayscale, or 3 for rgb')    
parser.add_argument('-bsz', '--batch_size', default=64, type=int, help='Batch size. What else do you need to know?')
parser.add_argument('-lr', '--base_learning_rate', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('-epochs', '--num_epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('-f',default='')
args = parser.parse_args()    

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from pyleaves.utils import ensure_dir_exists, set_visible_gpus
gpu_ids = [args.gpu_id]
set_visible_gpus(gpu_ids)
AUTOTUNE = tf.data.experimental.AUTOTUNE
from pyleaves.train.base_train import BaseTrainer







class TFDSTrainer(BaseTrainer):
    '''
    Trainer for querying data from tensorflow-datasets for use in pyleaves, while maintaining as close to an identical API to our custom datasets loaded from leavesdb
    
    Still need to figure out good way to implement:
        1) filter_low_count_labels functionality for datasets loaded from TFDS. Need to do it without loading full train+val+test data into memory
    '''
    
    def __init__(self, experiment_config):
        
        print(experiment_config.dataset_name)
        self.list_builders = tfds.list_builders()
        assert experiment_config.dataset_name in self.list_builders
        super().__init__(experiment_config)
        

    def extract(self):
        # Load a given dataset by name, along with the DatasetInfo
               
        dataset_name = self.config.dataset_name
        
        self.dataset_builder = tfds.builder(dataset_name)
        self.dataset_builder.download_and_prepare()
        
    def transform(self):
        #Get the available splits in original TFDS dataset configuration
        dataset_builder = self.dataset_builder
        #List the str names of the subsplits included in the official version of the dataset on TFDS
        dataset_splits = list(dataset_builder.info.splits.keys())
        self.num_classes = dataset_builder.info.features['label'].num_classes
        
        val_size = self.config.data_splits['val_size']
        test_size = self.config.data_splits['test_size']        
        
        #Calculate absolute percent for each subsplit based on fractional split sizes
        #e.g. for {'val_size':0.2,'test_size':0.2}: (train / val / test) percent-> (64% / 16% / 20%)
        test_percent = int(test_size*100)
        val_percent = int(np.floor(val_size*(100-test_percent)))
        train_percent = 100 - (val_percent+test_percent)
        self.splits_specification = {
                                     'train': '+'.join([s+f'[:{train_percent}%]' for s in dataset_splits]),
                                     'val'  : '+'.join([s+f'[{train_percent}%:{val_percent+train_percent}%]' for s in dataset_splits]),
                                     'test' : '+'.join([s+f'[-{test_percent}%:]' for s in dataset_splits]) 
                                     }
        
        self.metadata_splits = {}
        total_samples = 0
        for sub_split in dataset_splits:
            total_samples += dataset_builder.info.splits[sub_split].num_examples

        self.metadata_splits = {
                                'train':{
                                         'num_samples':total_samples*train_percent*0.01,
                                         'num_classes':self.num_classes
                                        },
                                'val':{
                                         'num_samples':total_samples*val_percent*0.01,
                                         'num_classes':self.num_classes
                                       },
                                'test':{
                                         'num_samples':total_samples*test_percent*0.01,
                                         'num_classes':self.num_classes
                                        }
                               }
        print('total_samples',total_samples)        
        
        
        
    def load(self):
        dataset_builder = self.dataset_builder
        splits_specification = self.splits_specification
        
        self.dataloader_splits = {
        'train': dataset_builder.as_dataset(split=splits_specification['train']),
        'val': dataset_builder.as_dataset(split=splits_specification['val']),
        'test': dataset_builder.as_dataset(split=splits_specification['test'])
        }
        
#         self.metadata_splits = 
        
        
    def get_image_resize_function(self):
        height, width = self.config.target_size
        resizer = partial(tf.image.resize_image_with_pad,
                       target_height=height,
                       target_width=width)
        def resizer_func(x, y):
            return tf.cast(resizer(x), tf.uint8), y
        
        return resizer_func
    
    def decode_tfds_example(self, x, y):
        x = tf.image.convert_image_dtype(x, dtype=tf.uint8)
        y = tf.cast(y, tf.int32)
        y = tf.one_hot(y, depth=self.num_classes)
        return x, y
        
    def get_data_loader(self, subset='train', skip_preprocessing=False):
        assert subset in self.dataloader_splits.keys()
        
        data = self.dataloader_splits[subset]
        config = self.config
    
        resizer = self.get_image_resize_function()
        decode_example = self.decode_tfds_example
        
        if skip_preprocessing:
            print('skipping_preprocessing')
            def preprocessing(data_sample):
                return data_sample['image'], data_sample['label']
            data = data.map(preprocessing, num_parallel_calls=AUTOTUNE)            
        else:
            data = data \
                  .map(self.preprocessing, num_parallel_calls=AUTOTUNE) # \
#                   .map(resizer, num_parallel_calls=AUTOTUNE)
        data = data.map(decode_example, num_parallel_calls=AUTOTUNE) \
                   .map(resizer, num_parallel_calls=AUTOTUNE)
#         if self.preprocessing == 'imagenet':
#             data = data.map(imagenet_mean_subtraction, num_parallel_calls=AUTOTUNE)
        
        if subset == 'train':
            data = data.shuffle(buffer_size=config.buffer_size, seed=config.seed)
            
            if config.augment_images == True:
                data = data.map(self.augmentors.rotate, num_parallel_calls=AUTOTUNE) \
                           .map(self.augmentors.flip, num_parallel_calls=AUTOTUNE) #\
#                            .map(self.augmentors.color, num_parallel_calls=AUTOTUNE)
#         if not skip_preprocessing:
        data = data.batch(config.batch_size, drop_remainder=False) \
                   .repeat() \
                   .prefetch(AUTOTUNE)
        return data




    

    
    
    




    
if __name__ == '__main__':
    


    
    import datetime
    import dataset
    from functools import partial
    import mlflow
    import mlflow.tensorflow
    import numpy as np
    import os
    import pandas as pd


    from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, generate_encoding_map
    from pyleaves import leavesdb
    from pyleaves.models.keras_models import build_model
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
    from pyleaves.analysis.img_utils import TFRecordCoder, plot_image_grid, imagenet_mean_subtraction, ImageAugmentor, get_keras_preprocessing_function
    from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata, reset_eager_session
    from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords

    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig

    from stuf import stuf
    import tensorflow_datasets as tfds    
    
    
    tracking_dir = r'/media/data/jacob/Fossil_Project/experiments/mlflow'
    ensure_dir_exists(tracking_dir)
    mlflow.set_tracking_uri(tracking_dir)
#     mlflow.set_tracking_uri(r'sqlite:///'+tracking_dir+'/experiment.db')
    print(mlflow.tracking.get_tracking_uri())
    mlflow.set_experiment('tfds-baselines')
    print(mlflow.get_artifact_uri())    
    
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',args.model_name,args.dataset_name, current_time)

    reset_eager_session()
    
    if args.dataset_name in ['None',*[str(i) for i in range(9)]]:
        plant_dataset_names = ['cassava', 'citrus_leaves', 'deep_weeds', 'i_naturalist2017', 'oxford_flowers102', 'plant_leaves', 'plant_village', 'plantae_k', 'tf_flowers']
        idx = int(args.dataset_name)
        args.dataset_name = plant_dataset_names[idx]

    model_names = ['vgg16', 'xception', 'resnet_50_v2','resnet_101_v2', 'shallow']
    
    print('model_name=',args.model_name)
    if args.model_name in ['vgg16', 'resnet_50_v2','resnet_101_v2']:
        target_size=(224,224)
    elif args.model_name=='xception':
        target_size=(299,299)
    else:
        target_size=(224,224)
    print('target_size=',target_size)
    dataset_config = DatasetConfig(dataset_name=args.dataset_name,
                                   label_col='family',
                                   target_size=target_size,
                                   num_channels=args.num_channels,
                                   low_class_count_thresh=3,
                                   data_splits={'val_size':0.2,'test_size':0.2},
                                   num_shards=10,
                                   input_format=dict)

    train_config = TrainConfig(model_name=args.model_name,
                                               batch_size=args.batch_size,
                                               frozen_layers=(0,-4),
                                               base_learning_rate=args.base_learning_rate,
                                               buffer_size=1000,
                                               num_epochs=args.num_epochs,
                                               preprocessing=True,
                                               augment_images=True,
                                               augmentations=['rotate','flip'],
                                               seed=4) #3)

    experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                         train_config=train_config)            

    trainer = TFDSTrainer(experiment_config=experiment_config)

    ##LOAD AND PLOT 1 BATCH OF IMAGES AND LABELS FROM FOSSIL DATASET
#     experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',trainer.config.model_name,trainer.config.dataset_name)

    train_data = trainer.get_data_loader(subset='train')#, skip_preprocessing=True)
    val_data = trainer.get_data_loader(subset='val')
    test_data = trainer.get_data_loader(subset='test')
    
    model_params = trainer.get_model_params('train')
    fit_params = trainer.get_fit_params()
    
    with mlflow.start_run(run_name=f'tfds-{args.model_name}-{args.dataset_name}-lr_{args.base_learning_rate}_baseline', nested=True):
        mlflow.tensorflow.autolog()

        callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'weights_best.h5'), logs_dir=os.path.join(experiment_dir,'logdir'), restore_best_weights=False)

        print('model_params',model_params)
    
        model = build_model(**model_params)  #name='shallow', num_classes=10000, frozen_layers=(0,-4), input_shape=(224,224,3), base_learning_rate=0.0001)        
        
        
        history = model.fit(train_data,
                     steps_per_epoch = fit_params['steps_per_epoch'],
                     epochs=fit_params['epochs'],
                     validation_data=val_data,
                     validation_steps=fit_params['validation_steps'],
                     callbacks=callbacks
                     )
        
        mlflow.log_params(trainer.config)    
    
    


#     for imgs, labels in train_data.take(1):
#         labels = [np.argmax(label) for label in labels.numpy()]    
# #         imgs = (imgs.numpy()+1)/2
#         plot_image_grid(imgs, labels, 4, 8)

    
    

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# subsets = ['train','val','test']
    
# for subset in subsets:
#     data = tf.data.Dataset.from_tensor_slices(trainer.tfrecord_files[subset]) \
#             .apply(lambda x: tf.data.TFRecordDataset(x)) \
#             .map(trainer.coder.decode_example,num_parallel_calls=AUTOTUNE) \
#             .shuffle(buffer_size=500, seed=trainer.config.seed) \
#             .batch(trainer.config.batch_size, drop_remainder=False) \
#             .prefetch(AUTOTUNE)
#     try:
#         for i, (imgs, labels) in enumerate(data):
#             print(i, imgs.shape, labels.shape)
#     finally:
#         pass
    

# from tqdm import tqdm
# invalid_images = []
# for k, v in trainer.data_splits.items():
#     print(k)
    
#     for path in tqdm(v['path']):
#         if not os.path.isfile(path[0]):
#             print(f'FILE NOT FOUND: {path[0]}')
#             invalid_images.append(path[0])