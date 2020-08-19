'''

DEPRECATED (3/31/2020): All functionality moved to pyleaves/base/base_trainer.py

base_trainer.py

Script for implementing basic train logic.

@author: JacobARose
'''

import dataset
from stuf import stuf
from functools import partial
import numpy as np
import os
import pandas as pd

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

from pyleaves.utils import ensure_dir_exists, set_visible_gpus, validate_filepath
# gpu_ids = [3]
# set_visible_gpus(gpu_ids)
AUTOTUNE = tf.data.experimental.AUTOTUNE


from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, generate_encoding_map, LabelEncoder, get_class_counts
import pyleaves
from pyleaves import leavesdb
from pyleaves.leavesdb import db_utils, db_query
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves.analysis.img_utils import TFRecordCoder, plot_image_grid, imagenet_mean_subtraction, ImageAugmentor, get_keras_preprocessing_function, rgb2gray_3channel, rgb2gray_1channel
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata

from pyleaves.utils.csv_utils import load_csv_data
from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords

from pyleaves.config import DatasetConfig, TrainConfig, ModelConfig, ExperimentConfig
from pyleaves.models.base_model import get_model_default_param
from pyleaves.models.resnet import ResNet, ResNetGrayScale
from pyleaves.models.vgg16 import VGG16, VGG16GrayScale
from stuf import stuf



class SQLManager:
    '''
    ETL pipeline for preparing data from Leavesdb SQLite database and staging TFRecords for feeding into data loaders.

    Meant to be subclassed for use with BaseTrainer and future Trainer classes.
    '''
    def __init__(self, experiment_config, label_encoder=None):

        self.config = experiment_config
        self.configs = {'experiment_config':self.config}
        self.name = ''
        print('In SQLManager.__init__')
        self.init_params(label_encoder=label_encoder)

    def init_params(self, label_encoder):
#         import pdb; pdb.set_trace()
        self.tfrecord_root_dir = self.config['tfrecord_root_dir']
        self.model_dir = self.config['model_dir']
        self.data_db_path = self.config['data_db_path']
        self.db = None
        if label_encoder is None:
            self.encoder = LabelEncoder()
        else:
            self.encoder = label_encoder

        if 'label_encodings_filepath' in self.config:
            assert validate_filepath(self.config['label_encodings_filepath'],file_type='json')
            self.label_encodings_filepath = self.config['label_encodings_filepath']
        else:
            self.label_encodings_filepath = os.path.join(self.model_dir,f'{self.name}-label_encodings.json')
        self.config['label_encodings_filepath'] = self.label_encodings_filepath

    def extract(self, dataset_names=''):
        '''
        Query all filenames and labels associated with dataset_name

        Argmuents:
            dataset_names, list(str):
                list of individual dataset names to load into one dataframe

        Return:
            data, pd.DataFrame:
                DataFrame containing columns ['path','label','dataset']
        '''
        dataset_names = dataset_name.split('+')
        self.db_df = self.db_query(dataset_names=dataset_names)
        self.target_size = self.config.target_size
        self.num_channels = self.config.num_channels
        return self.db_df

    def transform(self, verbose=False):
        self.x, self.y = self.db_filter(self.db_df, verbose=verbose)
        self.data_splits, self.metadata_splits = self.split_data(self.x, self.y)
        self.num_classes = self.metadata_splits['train']['num_classes']
        self.config.num_classes = self.num_classes
        self.label_encodings = self.encoder.get_encodings()
        return self.data_splits

    def load(self):
        self.dataset_builder = DatasetBuilder(root_dir=self.tfrecord_root_dir,
                                              num_classes=self.num_classes)

        self.coder, self.tfrecord_files = self.stage_tfrecords()
        return self.tfrecord_files


#     '''
#     TODO: Refactor starting from db_query() to accept arbitrary number of datasets to be queried and concatenated together ###


#     '''
    def open_db_connection(self):
        '''
        Returns an open connection to db, starts it if it doesn't yet exist
        '''
        if not self.db:
            self.local_db = leavesdb.init_local_db(src_db=self.data_db_path)
            self.db = dataset.connect(f'sqlite:///{self.local_db}', row_type=stuf)
        return self.db

    def load_data(self,
                  db,
                  datasets=['Fossil','Leaves'],
                  x_col='path',
                  y_col='family',
                  keep_cols=['dataset']
                  ):

        data_df = pd.DataFrame(db['dataset'].all())
        data = []
        columns = [x_col, y_col, *keep_cols]
        for name in datasets:
            data += [ data_df[data_df.loc[:,'dataset'] == name] ]
        data = pd.concat(data)
        data = data.loc[:,columns]

        return data


    def db_query(self, dataset_names=['Fossil'], label_col='family'):
        '''
        Query all filenames and labels associated with dataset_name

        Argmuents:
            dataset_names, list(str):
                list of individual dataset names to load into one dataframe

        Return:
            data, pd.DataFrame:
                DataFrame containing columns ['path','label','dataset']
        '''
        db = self.open_db_connection()
        data = self.load_data(db,datasets=dataset_names,x_col='path',y_col=label_col)
        return data

    def db_filter(self, db_df, label_col='family', verbose=False):
        '''
        Function to apply preprocessing to output of db_query, prior to conversion of images to TFRecord.

        '''
        threshold = self.config.low_class_count_thresh
        db_df = filter_low_count_labels(db_df, threshold=threshold, verbose=verbose)

        if len(self.encoder)==0:
            self.encoder.merge_labels(labels=list(db_df[label_col]))
        self.encoder.save_labels(self.config['label_encodings_filepath'])

        db_df = self.encoder.filter(db_df, label_col=label_col)

        self.x = db_df['path'].values.reshape((-1,1))
        self.y = np.array(self.encoder.transform(db_df[label_col]))


        return self.x, self.y

    def split_data(self, x, y, verbose=False):
        '''
        Function to split data ino k-splits. Currently, default is to simply split into train/val/test sets
        '''
        val_size = self.config.data_splits['val_size']
        test_size = self.config.data_splits['test_size']

        self.data_splits = train_val_test_split(x, y, val_size=val_size, test_size=test_size)

        self.metadata_splits = get_data_splits_metadata(self.data_splits, self.db_df, encoder=self.encoder, verbose=verbose)
        return self.data_splits, self.metadata_splits

    def get_class_counts(self):
        class_count_splits = {}
        for subset, subset_data in self.data_splits.items():
            print(subset)
            if type(subset_data['path'])==np.ndarray:
                subset_data['path'] = subset_data['path'].flatten().tolist()
            labels, label_counts = get_class_counts(pd.DataFrame.from_dict(subset_data))
            class_count_splits[subset] = {l:c for l,c in zip(labels, label_counts)}
        return class_count_splits

    def stage_tfrecords(self, verbose=False):
        '''
        Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
        '''
        self.root_dir = self.tfrecord_root_dir
        dataset_name = self.config.dataset_name
        val_size = self.config.data_splits['val_size']
        test_size = self.config.data_splits['test_size']

        #Store records in subdirectories labeled with relevant metadata
        record_subdirs = [dataset_name,
                          f'num_channels-3_thresh-{self.config.low_class_count_thresh}',
                          f'val_size={val_size}-test_size={test_size}']

        tfrecords = self.dataset_builder.recursive_search(self.root_dir,
                                                          subdirs=record_subdirs,
                                                          verbose=verbose)
        if tfrecords is None:
            return create_tfrecords(self.config,
                                    record_subdirs,
                                    data_splits=self.data_splits,
                                    metadata_splits=self.metadata_splits)
        else:
            coder = TFRecordCoder(self.data_splits['train'],
                                 self.root_dir,
                                 record_subdirs=record_subdirs,
                                 target_size=self.target_size,
                                 num_channels=self.num_channels,
                                 num_classes=self.num_classes)
            return coder, tfrecords



#####################################################################################
#####################################################################################




class SQLMultiManager:
    '''
    ETL pipeline for preparing data from Leavesdb SQLite database and staging TFRecords for feeding into data loaders.

    Meant to be subclassed for use with BaseTrainer and future Trainer classes.
    '''
    def __init__(self, exp_configs={}, label_encoder=None):

        self.configs = exp_configs
        self.name = ''
        print('In SQLManager.__init__')
        self.init_params(label_encoder=label_encoder)

    def init_params(self, label_encoder):
#         import pdb; pdb.set_trace()
        self.tfrecord_root_dir = self.config['tfrecord_root_dir']
        self.model_dir = self.config['model_dir']
        self.data_db_path = self.config['data_db_path']
        self.db = None
        self.data = {}
        if label_encoder is None:
            self.encoder = LabelEncoder()
        else:
            self.encoder = label_encoder

        if 'label_encodings_filepath' in self.config:
            assert validate_filepath(self.config['label_encodings_filepath'],file_type='json')
            self.label_encodings_filepath = self.config['label_encodings_filepath']
        else:
            self.label_encodings_filepath = os.path.join(self.model_dir,f'{self.name}-label_encodings.json')
        self.config['label_encodings_filepath'] = self.label_encodings_filepath

    def extract(self):
        for key, config in self.configs.items():
            self.data[key] = self.db_query(dataset_names=config.dataset_names)

        self.target_size = config.target_size
        self.num_channels = config.num_channels
        return self.data

    def transform(self, verbose=False):

        self.x, self.y = self.db_filter(self.db_df, verbose=verbose)
        self.data_splits, self.metadata_splits = self.split_data(self.x, self.y)
        self.num_classes = self.metadata_splits['train']['num_classes']
        self.config.num_classes = self.num_classes
        self.label_encodings = self.encoder.get_encodings()
        return self.data_splits

    def load(self):
        self.dataset_builder = DatasetBuilder(root_dir=self.tfrecord_root_dir,
                                              num_classes=self.num_classes)

        self.coder, self.tfrecord_files = self.stage_tfrecords()
        return self.tfrecord_files


#     '''
#     TODO: Refactor starting from db_query() to accept arbitrary number of datasets to be queried and concatenated together ###


#     '''
    def open_db_connection(self):
        '''
        Returns an open connection to db, starts it if it doesn't yet exist
        '''
        if not self.db:
            self.local_db = leavesdb.init_local_db(src_db=self.data_db_path)
            self.db = dataset.connect(f'sqlite:///{self.local_db}', row_type=stuf)
        return self.db

    def load_data(self,
                  db,
                  datasets=['Fossil','Leaves'],
                  x_col='path',
                  y_col='family',
                  keep_cols=['dataset']
                  ):

        data_df = pd.DataFrame(db['dataset'].all())
        data = []
        columns = [x_col, y_col, *keep_cols]
        for name in datasets:
            data += [
                        data_df[data_df.loc[:,'dataset'] == name]
                    ]
        data = pd.concat(data)
        data = data.loc[:,columns]

        return data


    def db_query(self, dataset_names=['Fossil'], x_col='path', y_col='family', keep_cols=['dataset']):
        '''
        Query all filenames and labels associated with dataset_names

        Return:
            self.db_df, pd.DataFrame:
                DataFrame containing columns ['path','label']
        '''
#         import pdb; pdb.set_trace()
        db = self.open_db_connection()
        data = []
        data = self.load_data(db,
                  datasets=dataset_names,
                  x_col=x_col,
                  y_col=y_col,
                  keep_cols=keep_cols
                  )
        return data

    def db_filter(self, db_df, label_col='family', verbose=False):
        '''
        Function to apply preprocessing to output of db_query, prior to conversion of images to TFRecord.

        '''
        threshold = self.config.low_class_count_thresh
        db_df = filter_low_count_labels(db_df, threshold=threshold, verbose=verbose)

        if len(self.encoder)==0:
            self.encoder.merge_labels(labels=list(db_df[label_col]))
        self.encoder.save_labels(self.config['label_encodings_filepath'])

        db_df = self.encoder.filter(db_df, label_col=label_col)

        self.x = db_df['path'].values.reshape((-1,1))
        self.y = np.array(self.encoder.transform(db_df[label_col]))


        return self.x, self.y

    def split_data(self, x, y, verbose=False):
        '''
        Function to split data ino k-splits. Currently, default is to simply split into train/val/test sets
        '''
        val_size = self.config.data_splits['val_size']
        test_size = self.config.data_splits['test_size']

        self.data_splits = train_val_test_split(x, y, val_size=val_size, test_size=test_size)

        self.metadata_splits = get_data_splits_metadata(self.data_splits, self.db_df, encoder=self.encoder, verbose=verbose)
        return self.data_splits, self.metadata_splits

    def get_class_counts(self):
        class_count_splits = {}
        for subset, subset_data in self.data_splits.items():
            print(subset)
            if type(subset_data['path'])==np.ndarray:
                subset_data['path'] = subset_data['path'].flatten().tolist()
            labels, label_counts = get_class_counts(pd.DataFrame.from_dict(subset_data))
            class_count_splits[subset] = {l:c for l,c in zip(labels, label_counts)}
        return class_count_splits

    def stage_tfrecords(self, verbose=False):
        '''
        Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
        '''
        self.root_dir = self.tfrecord_root_dir
        dataset_name = self.config.dataset_name
        val_size = self.config.data_splits['val_size']
        test_size = self.config.data_splits['test_size']

        #Store records in subdirectories labeled with relevant metadata
        record_subdirs = [dataset_name,
                          f'num_channels-3_thresh-{self.config.low_class_count_thresh}',
                          f'val_size={val_size}-test_size={test_size}']

        tfrecords = self.dataset_builder.recursive_search(self.root_dir,
                                                          subdirs=record_subdirs,
                                                          verbose=verbose)
        if tfrecords is None:
            return create_tfrecords(self.config,
                                    record_subdirs,
                                    data_splits=self.data_splits,
                                    metadata_splits=self.metadata_splits)
        else:
            coder = TFRecordCoder(self.data_splits['train'],
                                 self.root_dir,
                                 record_subdirs=record_subdirs,
                                 target_size=self.target_size,
                                 num_channels=self.num_channels,
                                 num_classes=self.num_classes)
            return coder, tfrecords













###########################################################################
###########################################################################



# class SQLManager:
#     '''
#     ETL pipeline for preparing data from Leavesdb SQLite database and staging TFRecords for feeding into data loaders.

#     Meant to be subclassed for use with BaseTrainer and future Trainer classes.
#     '''
#     def __init__(self, experiment_config, label_encoder=None):

#         self.config = experiment_config
#         self.configs = {'experiment_config':self.config}
#         self.name = ''
#         print('In SQLManager.__init__')
#         self.init_params(label_encoder=label_encoder)

#     def init_params(self, label_encoder):
# #         import pdb; pdb.set_trace()
#         self.tfrecord_root_dir = self.config['tfrecord_root_dir']
#         self.model_dir = self.config['model_dir']
#         self.data_db_path = self.config['data_db_path']

#         if label_encoder is None:
#             self.encoder = LabelEncoder()
#         else:
#             self.encoder = label_encoder

#         if 'label_encodings_filepath' in self.config:
#             assert validate_filepath(self.config['label_encodings_filepath'],file_type='json')
#             self.label_encodings_filepath = self.config['label_encodings_filepath']
#         else:
#             self.label_encodings_filepath = os.path.join(self.model_dir,f'{self.name}-label_encodings.json')
#         self.config['label_encodings_filepath'] = self.label_encodings_filepath

#     def extract(self):
#         self.db_df = self.db_query(dataset_name=self.config.dataset_name)
#         self.target_size = self.config.target_size
#         self.num_channels = self.config.num_channels
#         return self.db_df

#     def transform(self, verbose=False):
#         self.x, self.y = self.db_filter(self.db_df, verbose=verbose)
#         self.data_splits, self.metadata_splits = self.split_data(self.x, self.y)
#         self.num_classes = self.metadata_splits['train']['num_classes']
#         self.config.num_classes = self.num_classes
#         self.label_encodings = self.encoder.get_encodings()
#         return self.data_splits

#     def load(self):
#         self.dataset_builder = DatasetBuilder(root_dir=self.tfrecord_root_dir,
#                                               num_classes=self.num_classes)

#         self.coder, self.tfrecord_files = self.stage_tfrecords()
#         return self.tfrecord_files


# #     '''
# #     TODO: Refactor starting from db_query() to accept arbitrary number of datasets to be queried and concatenated together ###


# #     '''

#     def db_query(self, dataset_name='Fossil', label_col='family'):
#         '''
#         Query all filenames and labels associated with dataset_name

#         Return:
#             self.db_df, pd.DataFrame:
#                 DataFrame containing columns ['path','label']
#         '''
# #         import pdb; pdb.set_trace()
#         self.local_db = leavesdb.init_local_db(src_db=self.data_db_path)
#         self.db = dataset.connect(f'sqlite:///{self.local_db}', row_type=stuf)
#         self.db_df = pd.DataFrame(leavesdb.db_query.load_data(self.db, y_col=label_col, dataset=dataset_name))

#         return self.db_df

#     def db_filter(self, db_df, label_col='family', verbose=False):
#         '''
#         Function to apply preprocessing to output of db_query, prior to conversion of images to TFRecord.

#         '''
#         threshold = self.config.low_class_count_thresh
#         db_df = filter_low_count_labels(db_df, threshold=threshold, verbose=verbose)

#         if len(self.encoder)==0:
#             self.encoder.merge_labels(labels=list(db_df[label_col]))
#         self.encoder.save_labels(self.config['label_encodings_filepath'])

#         db_df = self.encoder.filter(db_df, label_col=label_col)

#         self.x = db_df['path'].values.reshape((-1,1))
#         self.y = np.array(self.encoder.transform(db_df[label_col]))


#         return self.x, self.y

#     def split_data(self, x, y, verbose=False):
#         '''
#         Function to split data ino k-splits. Currently, default is to simply split into train/val/test sets
#         '''
#         val_size = self.config.data_splits['val_size']
#         test_size = self.config.data_splits['test_size']

#         self.data_splits = train_val_test_split(x, y, val_size=val_size, test_size=test_size)

#         self.metadata_splits = get_data_splits_metadata(self.data_splits, self.db_df, encoder=self.encoder, verbose=verbose)
#         return self.data_splits, self.metadata_splits

#     def get_class_counts(self):
#         class_count_splits = {}
#         for subset, subset_data in self.data_splits.items():
#             print(subset)
#             if type(subset_data['path'])==np.ndarray:
#                 subset_data['path'] = subset_data['path'].flatten().tolist()
#             labels, label_counts = get_class_counts(pd.DataFrame.from_dict(subset_data))
#             class_count_splits[subset] = {l:c for l,c in zip(labels, label_counts)}
#         return class_count_splits

#     def stage_tfrecords(self, verbose=False):
#         '''
#         Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
#         '''
#         self.root_dir = self.tfrecord_root_dir
#         dataset_name = self.config.dataset_name
#         val_size = self.config.data_splits['val_size']
#         test_size = self.config.data_splits['test_size']

#         #Store records in subdirectories labeled with relevant metadata
#         record_subdirs = [dataset_name,
#                           f'num_channels-3_thresh-{self.config.low_class_count_thresh}',
#                           f'val_size={val_size}-test_size={test_size}']

#         tfrecords = self.dataset_builder.recursive_search(self.root_dir,
#                                                           subdirs=record_subdirs,
#                                                           verbose=verbose)
#         if tfrecords is None:
#             return create_tfrecords(self.config,
#                                     record_subdirs,
#                                     data_splits=self.data_splits,
#                                     metadata_splits=self.metadata_splits)
#         else:
#             coder = TFRecordCoder(self.data_splits['train'],
#                                  self.root_dir,
#                                  record_subdirs=record_subdirs,
#                                  target_size=self.target_size,
#                                  num_channels=self.num_channels,
#                                  num_classes=self.num_classes)
#             return coder, tfrecords


# def load_csv_data(filepath):
#     data = pd.read_csv(filepath, encoding='latin1')
#     data = data.rename(columns={'Unnamed: 0':'id'})
#     return data

# class CSVManager(SQLManager):
#     '''
#     ETL pipeline for preparing data from CSV files and staging TFRecords for feeding into data loaders.

#     For increasing replicability of data experiments
#     '''
#     def __init__(self, experiment_config, label_encoder=None):
#         super().__init__(experiment_config, label_encoder=label_encoder)

#     def init_params(self, label_encoder):
# #         import pdb; pdb.set_trace()
#         self.tfrecord_root_dir = self.config['tfrecord_root_dir']
#         self.model_dir = self.config['model_dir']
#         self.experiment_root_dir = self.config.experiment_root_dir
#         self.domain = list(self.config.domain_data_configs.keys())
#         if label_encoder is None:
#             self.encoder = LabelEncoder()
#         else:
#             self.encoder = label_encoder


#     def extract(self):
#         self.subset_files = {d.domain: d['data'] for d in self.config.domain_data_configs.values()}

#         self.subset_data = {d.domain: {subset:[] for subset in d.subsets} for d in self.config.domain_data_configs.values()}
#         for domain, subsets in self.subset_data.items():
#             for subset in subsets:
#                 self.subset_data[domain][subset] = load_csv_data(self.subset_files[domain][subset])
#         return self.subset_data

#     def transform(self, verbose=False):
#         self.x, self.y = self.db_filter(self.db_df, verbose=verbose)

#         self.metadata_splits = {}
#         for domain, subsets in self.subset_data.items():
#             self.metadata_splits[domain] = get_data_splits_metadata(data_splits=self.subset_data[domain],
#                                                                     encoder=self.encoder,
#                                                                     verbose=verbose)

#         self.num_classes = self.metadata_splits[self.domain[0]]['train']['num_classes']
#         self.config.num_classes = self.num_classes
#         self.label_encodings = self.encoder.get_encodings()
#         return self.data_splits

#     def load(self):
#         self.dataset_builder = DatasetBuilder(root_dir=self.tfrecord_root_dir,
#                                               num_classes=self.num_classes)

#         self.coder, self.tfrecord_files = self.stage_tfrecords()
#         return self.tfrecord_files


#     def db_filter(self, data_df, label_col='y', verbose=False):
#         '''
#         Function to apply preprocessing to output of db_query, prior to conversion of images to TFRecord.

#         '''
# #         threshold = self.config.low_class_count_thresh
# #         db_df = filter_low_count_labels(db_df, threshold=threshold,y_col=label_col, verbose=verbose)
#         if len(self.encoder)==0:
#             self.encoder.merge_labels(labels=list(data_df[label_col]))
#         self.encoder.save_labels(self.config.domain_data_configs[self.domain[0]]['label_mappings'])

#         data_df = self.encoder.filter(data_df, label_col=label_col)

#         self.x = data_df['x'].values.reshape((-1,1))
#         self.y = np.array(self.encoder.transform(data_df[label_col]))

#         return self.x, self.y


#     def get_class_counts(self):
#         class_count_splits = {}
#         for subset, subset_data in self.data_splits.items():
#             print(subset)
#             if type(subset_data['path'])==np.ndarray:
#                 subset_data['path'] = subset_data['path'].flatten().tolist()
#             labels, label_counts = get_class_counts(pd.DataFrame.from_dict(subset_data))
#             class_count_splits[subset] = {l:c for l,c in zip(labels, label_counts)}
#         return class_count_splits

#     def stage_tfrecords(self, verbose=False):
#         '''
#         Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
#         '''
#         self.root_dir = self.tfrecord_root_dir
#         dataset_name = self.config.dataset_name
#         val_size = self.config.data_splits['val_size']
#         test_size = self.config.data_splits['test_size']

#         #Store records in subdirectories labeled with relevant metadata
#         record_subdirs = [dataset_name,
#                           f'num_channels-3_thresh-{self.config.low_class_count_thresh}',
#                           f'val_size={val_size}-test_size={test_size}']

#         tfrecords = self.dataset_builder.recursive_search(self.root_dir,
#                                                           subdirs=record_subdirs,
#                                                           verbose=verbose)
#         if tfrecords is None:
#             return create_tfrecords(self.config,
#                                     record_subdirs,
#                                     data_splits=self.data_splits,
#                                     metadata_splits=self.metadata_splits)
#         else:
#             coder = TFRecordCoder(self.data_splits['train'],
#                                  self.root_dir,
#                                  record_subdirs=record_subdirs,
#                                  target_size=self.target_size,
#                                  num_channels=self.num_channels,
#                                  num_classes=self.num_classes)
#             return coder, tfrecords




######################################################################################################
######################################################################################################
######################################################################################################










class BaseTrainer(SQLManager):

    def __init__(self, experiment_config, label_encoder=None): #, **kwargs):
        print('In BaseTrainer.__init__')
        super().__init__(experiment_config, label_encoder=label_encoder)

#         self.init_params()

        self.extract(self.dataset_names)
        self.transform()
        self.load()

    def init_params(self, label_encoder):

#         print('In BaseTrainer.init_params')
        super().init_params(label_encoder=label_encoder)
        self.augmentors = ImageAugmentor(self.config.augmentations, seed=self.config.seed)
        self.grayscale = self.config.grayscale

        if self.config.target_size=='default':
            self.config.input_shape = get_model_default_param(config=self.config, param='input_shape')
            self.config.target_size = self.config.input_shape[:-1]
            self.config.num_channels = self.config.input_shape[-1]

        if self.config.preprocessing:
            self.preprocessing = get_keras_preprocessing_function(self.config.model_name,
                                                                  self.config.input_format,
                                                                  x_col = self.config.x_col,
                                                                  y_col = self.config.y_col)

    def get_data_loader(self, subset='train'):
        assert subset in self.tfrecord_files.keys()

        tfrecord_paths = self.tfrecord_files[subset]
        config = self.config

        data = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
                    .apply(lambda x: tf.data.TFRecordDataset(x)) \
                    .map(self.coder.decode_example, num_parallel_calls=AUTOTUNE) \
                    .map(self.preprocessing, num_parallel_calls=AUTOTUNE)

#         import pdb; pdb.set_trace()
        if self.grayscale == True:
            if self.num_channels==3:
                data = data.map(rgb2gray_3channel, num_parallel_calls=AUTOTUNE)
            elif self.num_channels==1:
                data = data.map(rgb2gray_1channel, num_parallel_calls=AUTOTUNE)
#         if self.preprocessing == 'imagenet':
#             data = data.map(imagenet_mean_subtraction, num_parallel_calls=AUTOTUNE)

        if subset == 'train':
            if config.augment_images == True:
                data = data.map(self.augmentors.rotate, num_parallel_calls=AUTOTUNE) \
                           .map(self.augmentors.flip, num_parallel_calls=AUTOTUNE)

            data = data.shuffle(buffer_size=config.buffer_size, seed=config.seed)

        data = data.batch(config.batch_size, drop_remainder=False) \
                   .repeat() \
                   .prefetch(AUTOTUNE)

        return data

    def get_model_config(self, subset='train', num_classes=None):
        if num_classes is None:
            metadata = self.metadata_splits[subset]
            num_classes = metadata['num_classes']
        config = self.config

        model_config = ModelConfig(
                                   model_name=config.model_name,
                                   num_classes=num_classes,
                                   frozen_layers=config.frozen_layers,
                                   input_shape=(*config.target_size,config.num_channels),
                                   base_learning_rate=config.base_learning_rate,
                                   regularization=config.regularization
                                   )
        self.configs['model_config'] = model_config
        return model_config

    def init_model_builder(self, subset='train', num_classes=None):
        model_config = self.get_model_config(subset=subset, num_classes=num_classes)
        self.model_name = model_config.model_name
        if self.model_name == 'vgg16':
            self.add_model_manager(VGG16GrayScale(model_config))
        elif self.model_name.startswith('resnet'):
            self.add_model_manager(ResNet(model_config))
        self.model = self.model_manager.build_model()

    def add_model_manager(self, model_manager):
        '''Simply for adding a subclass of BaseModel for trainer to keep track of, in order to
        extend model building/saving/loading/importing/exporting functionality to trainer.'''
        self.model_manager = model_manager


    def get_fit_params(self):
        params = {'steps_per_epoch' : self.metadata_splits['train']['num_samples']//self.config.batch_size,
                  'validation_steps' : self.metadata_splits['val']['num_samples']//self.config.batch_size,
                  'epochs' : self.config.num_epochs
                 }
        self.configs['fit_params'] = params
        return params


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
