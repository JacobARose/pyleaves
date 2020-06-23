'''
base_trainer.py

Script for implementing basic train logic.

@author: JacobARose
'''

import dataset
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
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves.analysis.img_utils import TFRecordCoder, plot_image_grid, imagenet_mean_subtraction, ImageAugmentor, get_keras_preprocessing_function, rgb2gray_3channel, rgb2gray_1channel
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata

from pyleaves.utils.csv_utils import load_csv_data
from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords


from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history
from pyleaves.train.base_trainer import BaseTrainer, SQLManager
from pyleaves.train.transfer_trainer import MLFlowTrainer
from pyleaves.config import DatasetConfig, TrainConfig, ModelConfig, ExperimentConfig
from pyleaves.models.base_model import get_model_default_param
from pyleaves.models.resnet import ResNet, ResNetGrayScale
from pyleaves.models.vgg16 import VGG16, VGG16GrayScale
from stuf import stuf


        
def load_csv_data(filepath):
    data = pd.read_csv(filepath, encoding='latin1')
    data = data.rename(columns={'Unnamed: 0':'id'})    
    return data

class CSVManager(SQLManager):
    '''
    ETL pipeline for preparing data from CSV files and staging TFRecords for feeding into data loaders.
    
    For increasing replicability of data experiments
    '''
    def __init__(self, experiment_config, label_encoder=None):
        super().__init__(experiment_config, label_encoder=label_encoder)

    def init_params(self, label_encoder):
#         import pdb; pdb.set_trace()
#         self.config['tfrecord_root_dir'] = self.config.dirs['tfrecord_root_dir']
        self.tfrecord_root_dir = self.config['tfrecord_root_dir']
        self.model_dir = self.config['model_dir']
        self.experiment_root_dir = self.config.experiment_root_dir
        self.domain = self.config.domain
#         self.target_size = self.config.data_configs[self.domain]['target_size']
#         self.num_channels = self.config.data_configs[self.domain]['num_channels']
        self.config.grayscale = self.config['grayscale']
        self.config.color_type = self.config['color_type']
#         self.config.target_size = self.target_size
#         self.config.num_channels = self.num_channels
        self.config.input_format = 'tuple'#'dict'
        if label_encoder is None:
            self.encoder = LabelEncoder()
        else:
            self.encoder = label_encoder        
        

    def extract(self):
#         self.subset_files = {d.domain: d['data'] for d in self.config.data_configs.values()}
        self.subset_files = self.config.data

#         self.subset_data = {d.domain: {subset:[] for subset in d.subsets} for d in self.config.data_configs.values()}
        self.subset_data = {subset:[] for subset in self.config.subsets}
#         for domain, subsets in self.subset_data.items():
#             for subset in subsets:
#                 self.subset_data[domain][subset] = load_csv_data(self.subset_files[domain][subset])
#         self.data_splits = self.subset_data[self.domain] #TODO DEPRECATE
        for subset in self.subset_data.keys():
            self.subset_data[subset] = load_csv_data(self.subset_files[subset])
        self.data_splits = self.subset_data

        return self.subset_data
    
    def transform(self, verbose=False):
        
#         self.metadata_splits = {}
#         for domain, subsets in self.subset_data.items():
#             for subset in subsets:
#                 self.subset_data[domain][subset] = self.db_filter(self.subset_data[domain][subset], int_label_col='y', verbose=verbose)
#             self.metadata_splits[domain] = get_data_splits_metadata(data_splits=self.subset_data[domain],
#                                                                     encoder=self.encoder,
#                                                                     int_label_col='y',
#                                                                     verbose=verbose)
        
#         self.num_classes = self.metadata_splits[self.domain]['train']['num_classes']
        self.metadata_splits = {}
        for subset in self.subset_data.keys():
            self.subset_data[subset] = self.db_filter(self.subset_data[subset], int_label_col='y', verbose=verbose)
            self.metadata_splits = get_data_splits_metadata(data_splits=self.subset_data,
                                                                    encoder=self.encoder,
                                                                    int_label_col='y',
                                                                    verbose=verbose)
        
        self.num_classes = self.metadata_splits['train']['num_classes']


        self.config.num_classes = self.num_classes
        self.label_encodings = self.encoder.get_encodings()
        return self.subset_data
    
    def load(self):
        self.dataset_builder = DatasetBuilder(root_dir=self.tfrecord_root_dir,
                                              num_classes=self.num_classes)
        
        self.coder, self.tfrecord_files = self.stage_tfrecords()
        self.data_root_dir = self.coder.root_dir
        return self.tfrecord_files
        
    
    def db_filter(self, data_df, text_label_col='family', int_label_col=None, verbose=False):
        '''
        Function to apply preprocessing to output of db_query, prior to conversion of images to TFRecord. 
        
        '''
#         threshold = self.config.low_class_count_thresh
#         db_df = filter_low_count_labels(db_df, threshold=threshold,y_col=label_col, verbose=verbose)
        
        if (len(self.encoder)==0) and (text_label_col in data_df.columns):
            self.encoder.merge_labels(labels=list(data_df[text_label_col]))
        self.encoder.save_labels(self.config.label_mappings)
        
        if (text_label_col in data_df.columns) or (int_label_col in data_df.columns):
            data_df = self.encoder.filter(data_df, text_label_col=text_label_col, int_label_col=int_label_col)
        
        x = data_df['x'].values.reshape((-1,1))
        if (text_label_col in data_df.columns):
            y = np.array(self.encoder.transform(data_df[text_label_col]))
        else:
            y = np.array(data_df['y'])
            
        print('db_filter output shapes: ', x.shape, y.shape)
        return {'x':x, 'y':y}


    def get_class_counts(self):
        class_count_splits = {}
        for subset, subset_data in self.data_splits.items():
            print(subset)
            if type(subset_data['path'])==np.ndarray:
                subset_data['path'] = subset_data['path'].flatten().tolist()
            labels, label_counts = get_class_counts(pd.DataFrame.from_dict(subset_data))
            class_count_splits[subset] = {l:c for l,c in zip(labels, label_counts)}
        return class_count_splits
    
    def stage_tfrecords(self, config=None, verbose=False):
        '''
        Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
        '''
        self.root_dir = self.tfrecord_root_dir
        domain = self.domain
        dataset_name = self.config.dataset_name
        val_size = self.config.data_splits['val_size']
        test_size = self.config.data_splits['test_size']
        
        thresh = self.config.low_class_count_thresh
        
        #Store records in subdirectories labeled with relevant metadata
        record_subdirs = [dataset_name,
                          f'num_channels-3_thresh-{thresh}',
                          f'val_size={val_size}-test_size={test_size}']
        
        tfrecords = self.dataset_builder.recursive_search(self.root_dir,
                                                          subdirs=record_subdirs,
                                                          verbose=verbose)
#         self.config.data_configs[domain].tfrecord_root_dir = self.config.tfrecord_root_dir
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
        
        
        
        
######################################################################################################        
######################################################################################################
######################################################################################################
        
        
class CSVTrainer(BaseTrainer, MLFlowTrainer, CSVManager):
    
    def __init__(self, experiment_config, model_builder=None, label_encoder=None):
        
        self.config = experiment_config
        self.model_builder = model_builder
        self.num_domains = 1
        self.histories = {}
        if True: #self.config.target_size=='default':
            self.grayscale=True
            self.config.color_type = self.config.dataset_config.color_type
            self.config.input_shape = get_model_default_param(config=self.config, param='input_shape')
            self.config.target_size = self.config.input_shape[:-1]
            self.target_size  = self.config.target_size
            self.config.num_channels = self.config.input_shape[-1]
            self.num_channels = self.config.num_channels
            
#         import pdb; pdb.set_trace()
        super().__init__(experiment_config, label_encoder=label_encoder)
        
        
    def init_model_builder(self, subset='train',num_classes=None):
#         import pdb; pdb.set_trace()
        super().init_model_builder(subset=subset, num_classes=num_classes)
        
        '''SPAGHETTI CODE, TODO: REPLACE'''
        self.metrics_names = self.model.metrics_names
        self.coder.num_channels=self.num_channels
        self.coder.target_size = self.target_size
#         self.config.num_channels=self.num_channels
        
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
#     def get_data_loader(self, domain='source', subset='train'):
#         '''
#         Get the proper trainer instance for specified domain ('source' or 'target'), then from that return the data loader for the specified subset ('train', 'val', or 'test').
#         '''
#         trainer = self.domains[domain]
#         return trainer.get_data_loader(subset=subset)
    
#     def get_model_config(self, domain='source', subset='train', num_classes=None):
#         trainer = self.domains[domain]
#         return trainer.get_model_config(subset=subset, num_classes=num_classes)
    
    
#     def get_model_config(self, domain, subset='train', num_classes=None):
#         if num_classes is None:
#             metadata = self.metadata_splits[subset]
#             num_classes = metadata['num_classes']
#         config = self.config
        
#         model_config = ModelConfig(
#                                    model_name=config.model_name,
#                                    num_classes=num_classes,
#                                    frozen_layers=config.frozen_layers,
#                                    input_shape=(*config.target_size,config.num_channels),
#                                    base_learning_rate=config.base_learning_rate,
#                                    regularization=config.regularization
#                                    )
#         self.configs['model_config'] = model_config
#         return model_config
    
#     def get_fit_params(self):
#         params = {'steps_per_epoch' : self.metadata_splits[self.domain]['train']['num_samples']//self.config.batch_size,
#                   'validation_steps' : self.metadata_splits[self.domain]['val']['num_samples']//self.config.batch_size,
#                   'epochs' : self.config.num_epochs
#                  }
#         self.configs['fit_params'] = params
#         return params


        
        

        
        
        
        

    
    
# class BaseTrainer(SQLManager):
    
#     def __init__(self, experiment_config, label_encoder=None): #, **kwargs):
#         print('In BaseTrainer.__init__')
#         super().__init__(experiment_config, label_encoder=label_encoder)

# #         self.init_params()
        
#         self.extract()
#         self.transform()
#         self.load()
        
#     def init_params(self, label_encoder):
# #         import pdb; pdb.set_trace()
# #         print('In BaseTrainer.init_params')
#         super().init_params(label_encoder=label_encoder)
#         self.augmentors = ImageAugmentor(self.config.augmentations, seed=self.config.seed)
#         self.grayscale = self.config.grayscale
        
#         if self.config.target_size=='default':
#             self.config.input_shape = get_model_default_param(config=self.config, param='input_shape')
#             self.config.target_size = self.config.input_shape[:-1]
#             self.config.num_channels = self.config.input_shape[-1]

#         if self.config.preprocessing:
#             self.preprocessing = get_keras_preprocessing_function(self.config.model_name, self.config.input_format)
    
#     def get_data_loader(self, subset='train'):
#         assert subset in self.tfrecord_files.keys()
        
#         tfrecord_paths = self.tfrecord_files[subset]
#         config = self.config
    
#         data = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
#                     .apply(lambda x: tf.data.TFRecordDataset(x)) \
#                     .map(self.coder.decode_example, num_parallel_calls=AUTOTUNE) \
#                     .map(self.preprocessing, num_parallel_calls=AUTOTUNE)

#         if self.grayscale == True:
#             if self.num_channels==3:
#                 data = data.map(rgb2gray_3channel, num_parallel_calls=AUTOTUNE)
#             elif self.num_channels==1:
#                 data = data.map(rgb2gray_1channel, num_parallel_calls=AUTOTUNE)
# #         if self.preprocessing == 'imagenet':
# #             data = data.map(imagenet_mean_subtraction, num_parallel_calls=AUTOTUNE)
        
#         if subset == 'train':            
#             if config.augment_images == True:
#                 data = data.map(self.augmentors.rotate, num_parallel_calls=AUTOTUNE) \
#                            .map(self.augmentors.flip, num_parallel_calls=AUTOTUNE)
                
#             data = data.shuffle(buffer_size=config.buffer_size, seed=config.seed)
    
#         data = data.batch(config.batch_size, drop_remainder=False) \
#                    .repeat() \
#                    .prefetch(AUTOTUNE)
        
#         return data
    
#     def get_model_config(self, subset='train', num_classes=None):
#         if num_classes is None:
#             metadata = self.metadata_splits[subset]
#             num_classes = metadata['num_classes']
#         config = self.config
        
#         model_config = ModelConfig(
#                                    model_name=config.model_name,
#                                    num_classes=num_classes,
#                                    frozen_layers=config.frozen_layers,
#                                    input_shape=(*config.target_size,config.num_channels),
#                                    base_learning_rate=config.base_learning_rate,
#                                    regularization=config.regularization
#                                    )
#         self.configs['model_config'] = model_config
#         return model_config

#     def init_model_builder(self, subset='train', num_classes=None):
#         model_config = self.get_model_config(subset=subset, num_classes=num_classes)
#         self.model_name = model_config.model_name
#         if self.model_name == 'vgg16':
#             self.add_model_manager(VGG16GrayScale(model_config))
#         elif self.model_name.startswith('resnet'):
#             self.add_model_manager(ResNet(model_config))
#         self.model = self.model_manager.build_model()
    
#     def add_model_manager(self, model_manager):
#         '''Simply for adding a subclass of BaseModel for trainer to keep track of, in order to
#         extend model building/saving/loading/importing/exporting functionality to trainer.'''
#         self.model_manager = model_manager    
   

#     def get_fit_params(self):
#         params = {'steps_per_epoch' : self.metadata_splits['train']['num_samples']//self.config.batch_size,
#                   'validation_steps' : self.metadata_splits['val']['num_samples']//self.config.batch_size,
#                   'epochs' : self.config.num_epochs
#                  }
#         self.configs['fit_params'] = params
#         return params
    
    
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
    
    
# class BaseTrainer_v1(BaseTrainer):
#     '''
#     Trainer class that uses a tf.compat.v1.Session() as well as using tf.data.Datasets with initializable syntax.
    
#     Use regular BaseTrainer class for eager execution
#     '''
    
#     def __init__(self, experiment_config):
#         super().__init__(experiment_config)
        
#         self.sess = tf.compat.v1.Session()    
    
#     def get_data_loader(self, subset='train'):
        
#         data = super().get_data_loader(subset=subset)
        
#         print(f'returning non-eager {subset} tf.data.Dataset iterator')
#         data_iterator = data.make_initializable_iterator()
#         self.sess.run(data_iterator.initializer)
        
#         return data_iterator
    
    
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

    
