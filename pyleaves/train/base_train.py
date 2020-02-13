'''
base_train.py

Script for implementing basic train logic.

@author: JacobARose
'''

import dataset
import numpy as np
import os
import pandas as pd
from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, generate_encoding_map
from pyleaves import leavesdb
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves.analysis.img_utils import TFRecordCoder, plot_image_grid
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata
from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords
from pyleaves.utils import ensure_dir_exists, set_visible_gpus
from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig

from stuf import stuf

import tensorflow as tf
# tf.enable_eager_execution()
# gpu_ids = [0]
# set_visible_gpus(gpu_ids)

    
    
class BaseTrainer:
    
    def __init__(self, experiment_config):
        
        self.config = experiment_config
        self.name = ''
        self.tfrecord_root_dir = self.config.dirs['tfrecord_root_dir']       
        
        self.extract()
        self.transform()
        self.load()
        
    def extract(self):
        self.db_df = self.db_query(dataset_name=self.config.dataset_name)
        return self.db_df
    
    def transform(self):
        self.x, self.y = self.db_filter(self.db_df)
        self.data_splits, self.metadata_splits = self.split_data(self.x, self.y)
        self.num_classes = self.metadata_splits['train']['num_classes']
        self.label_encodings = self.get_label_encodings(self.db_df, label_col=self.config.label_col)
        return self.data_splits
    
    def load(self):
        self.dataset_builder = DatasetBuilder(root_dir=self.tfrecord_root_dir,
                                              num_classes=self.num_classes,
                                              batch_size=self.config.batch_size,
                                              seed=self.config.seed)
        self.coder, self.tfrecord_files = self.stage_tfrecords()
        return self.tfrecord_files
        
        
    def db_query(self, dataset_name='Fossil', label_col='family'):
        '''
        Query all filenames and labels associated with dataset_name
        
        Return:
            self.db_df, pd.DataFrame:
                DataFrame containing columns ['path','label']
        '''
        self.local_db = leavesdb.init_local_db()
        self.db = dataset.connect(f'sqlite:///{self.local_db}', row_type=stuf)
        self.db_df = pd.DataFrame(leavesdb.db_query.load_data(self.db, y_col=label_col, dataset=dataset_name))
        
        return self.db_df
    
    def db_filter(self, db_df):
        '''
        Function to apply preprocessing to output of db_query, prior to conversion of images to TFRecord. 
        
        '''
        threshold = self.config.low_class_count_thresh
        db_df = filter_low_count_labels(db_df, threshold=threshold)
        db_df = encode_labels(db_df)
        self.x = db_df['path'].values.reshape((-1,1))
        self.y = db_df['label'].values
        
        return self.x, self.y
        
    def split_data(self, x, y):
        '''
        Function to split data ino k-splits. Currently, default is to simply split into train/val/test sets
        '''
        val_size = self.config.data_splits['val_size']
        test_size = self.config.data_splits['test_size']
        
        self.data_splits = train_val_test_split(x, y, val_size, test_size)
            
        self.metadata_splits = get_data_splits_metadata(self.data_splits, self.db_df, verbose=True)
        return self.data_splits, self.metadata_splits
    
    def get_label_encodings(self, db_df, label_col='family'):
        self.label_encodings = leavesdb.db_query.generate_encoding_map(db_df, text_label_col=label_col, int_label_col='label')
        self.num_classes = len(self.label_encodings)
        return self.label_encodings
    
    
    def stage_tfrecords(self):
        '''
        Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
        '''
        self.root_dir = self.tfrecord_root_dir
        dataset_name = self.config.dataset_name
        print('os.listdir(self.root_dir) = ', os.listdir(self.root_dir))
        if dataset_name in os.listdir(self.root_dir):
            dataset_root_dir = os.path.join(self.root_dir, dataset_name)
            if len(os.listdir(dataset_root_dir)) == 0:
                return create_tfrecords(self.config)
            
            for subset_dir in os.listdir(dataset_root_dir):
                if len(os.listdir(os.path.join(dataset_root_dir,subset_dir))) == 0:
                    return create_tfrecords(self.config)
                
            tfrecords = self.dataset_builder.collect_subsets(dataset_root_dir)
            
            if np.all([len(records_list) > 0 for _, records_list in tfrecords.items()]):    
                for records_subset, records_list in tfrecords.items():
                    print(f'found {len(records_list)} records in {records_subset}')
                return TFRecordCoder(self.data_splits['train'], self.root_dir,num_classes=self.num_classes), tfrecords
        
        print('Creating records')
        return create_tfrecords(self.config)
    
    def get_data_loader(self, subset='train'):
        assert subset in self.tfrecord_files.keys()
    
        return self.coder.read_tfrecords(self.tfrecord_files[subset],
                                  buffer_size=self.config.buffer_size,
                                  seed=self.config.seed,
                                  batch_size=self.config.batch_size)
    
    def get_model_params(self, subset='train'):
        metadata = self.metadata_splits[subset]
        config = self.config
        
        params = {'name':config.model_name,
                  'num_classes':metadata['num_classes'],
                  'frozen_layers':config.frozen_layers,
                  'input_shape':(*config.target_size,config.channels),
                  'base_learning_rate':config.base_learning_rate
                 }
        return params
    
    def get_fit_params(self):
        params = {'steps_per_epoch' : self.metadata_splits['train']['num_samples']//self.config.batch_size,
                  'validation_steps' : self.metadata_splits['val']['num_samples']//self.config.batch_size,
                  'epochs' : self.config.num_epochs
                 }
        return params
    
    
    
# class KerasTrainer(BaseTrain):
    
#     def __init__(self, experiment_config)

    
    
    

    
if __name__ == '__main__':

    dataset_config = DatasetConfig(dataset_name='Leaves', #'Fossil',
                                   label_col='family',
                                   target_size=(224,224),
                                   low_class_count_thresh=2, #0,
                                   data_splits={'val_size':0.2,'test_size':0.2},
                                   tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
                                   num_shards=10)

    train_config = TrainConfig(batch_size=32,
                               seed=4)

    experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                         train_config=train_config)

    trainer = BaseTrainer(experiment_config=experiment_config)
    # db_df = trainer.db_query(dataset_name='Fossil')
    # db_df.head()

    ##LOAD AND PLOT 1 BATCH OF IMAGES AND LABELS FROM FOSSIL DATASET

    train_data = trainer.coder.read_tfrecords(trainer.tfrecord_files['train'], buffer_size=500, seed=trainer.config.seed, batch_size=trainer.config.batch_size)

    for imgs, labels in train_data.take(1):
        labels = [trainer.label_encodings[label] for label in labels.numpy()]
        plot_image_grid(imgs, labels, 4, 8)

    
    

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