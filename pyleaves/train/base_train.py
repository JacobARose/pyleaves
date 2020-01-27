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
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata
from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords
from pyleaves.utils import ensure_dir_exists
from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig

from stuf import stuf

import tensorflow as tf
tf.enable_eager_execution()
gpu = 0




gpus = tf.config.experimental.get_visible_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
    logical_gpus = tf.config.experimental.get_visible_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")




    
    
class BaseTrainer:
    
    def __init__(self, experiment_config):
        
        self.config = experiment_config
        self.name = ''
    
        self.db_df = self.db_query(dataset_name=self.config.dataset_name)
        self.x, self.y = self.db_filter(self.db_df)
        self.data_splits, self.metadata_splits = self.split_data(self.x, self.y)
        self.label_encodings = self.get_label_encodings(self.db_df, y_col='family')
        
        
        self.root_dir = self.config.dirs['tfrecord_root_dir']       
        
        print('pre dataset_builder')
        self.dataset_builder = DatasetBuilder(root_dir=self.root_dir,
                                              num_classes=self.num_classes,
                                              batch_size=self.config.batch_size,
                                              seed=self.config.seed)
        print('post dataset_builder')
        self.tfrecord_files = self.generate_tfrecords()
        
        
    def db_query(self, dataset_name='Fossil', y_col='family'):
        '''
        Query all filenames and labels associated with dataset_name
        
        Return:
            self.db_df, pd.DataFrame:
                DataFrame containing columns ['path','label']
        '''
        self.local_db = leavesdb.init_local_db()
        self.db = dataset.connect(f'sqlite:///{self.local_db}', row_type=stuf)
        self.db_df = pd.DataFrame(leavesdb.db_query.load_data(self.db, y_col=y_col, dataset=dataset_name))
        
#         self.db_df = self.db_df.rename(columns={y_col:'label'})
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
    
    def get_label_encodings(self, db_df, y_col='family'):
        self.label_encodings = leavesdb.db_query.generate_encoding_map(db_df, text_label_col=y_col, int_label_col='label')
        self.num_classes = len(self.label_encodings)
        return self.label_encodings
    
    
    def generate_tfrecords(self):
        '''
        Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
        '''
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
                return tfrecords
        
        print('Creating records')
        return create_tfrecords(self.config)
    

    
    
    
dataset_config = DatasetConfig(dataset_name='Fossil',
                               target_size=(224,224),
                               low_class_count_thresh=10,
                               data_splits={'val_size':0.2,'test_size':0.2},
                               tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
                               num_shards=10)

train_config = TrainConfig(batch_size=32,
                           seed=3)

experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                     train_config=train_config)
    
trainer = BaseTrainer(experiment_config=experiment_config)
# db_df = trainer.db_query(dataset_name='Fossil')
# db_df.head()








