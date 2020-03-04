'''
base_train.py

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

from pyleaves.utils import ensure_dir_exists, set_visible_gpus
# gpu_ids = [6]
# set_visible_gpus(gpu_ids)
AUTOTUNE = tf.data.experimental.AUTOTUNE



from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, generate_encoding_map
from pyleaves import leavesdb
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves.analysis.img_utils import TFRecordCoder, plot_image_grid, imagenet_mean_subtraction, ImageAugmentor, get_keras_preprocessing_function, rgb2gray_3channel, rgb2gray_1channel
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata
from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords

from pyleaves.config import DatasetConfig, TrainConfig, ModelConfig, ExperimentConfig

from stuf import stuf

    
    
class BaseTrainer:
    
    def __init__(self, experiment_config):
        
        self.config = experiment_config
        self.name = ''
        self.tfrecord_root_dir = self.config.dirs['tfrecord_root_dir']     
        if self.config.preprocessing:
            self.preprocessing = get_keras_preprocessing_function(self.config.model_name, self.config.input_format)
            
        self.augmentors = ImageAugmentor(self.config.augmentations, seed=self.config.seed)
        self.grayscale = self.config.grayscale
        self.visualize = False#True

        self.extract()
        self.transform()
        self.load()
        
    def extract(self):
        self.db_df = self.db_query(dataset_name=self.config.dataset_name)
        self.target_size = self.config.target_size
        self.num_channels = self.config.num_channels
        return self.db_df
    
    def transform(self):
        self.x, self.y = self.db_filter(self.db_df)
        self.data_splits, self.metadata_splits = self.split_data(self.x, self.y)
        self.num_classes = self.metadata_splits['train']['num_classes']
        self.label_encodings = self.get_label_encodings(self.db_df, label_col=self.config.label_col)
        return self.data_splits
    
    def load(self):
        self.dataset_builder = DatasetBuilder(root_dir=self.tfrecord_root_dir,
                                              num_classes=self.num_classes) #,
#                                               batch_size=self.config.batch_size,
#                                               seed=self.config.seed)
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
        db_df = filter_low_count_labels(db_df, threshold=threshold, verbose=self.config.verbose)
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
            
        self.metadata_splits = get_data_splits_metadata(self.data_splits, self.db_df, verbose=self.config.verbose)
        return self.data_splits, self.metadata_splits
    
    def get_label_encodings(self, db_df, label_col='family'):
        self.label_encodings = leavesdb.db_query.get_label_encodings(db_df,
                                                                     y_col=label_col, 
                                                                     low_count_thresh=self.config.low_class_count_thresh, 
                                                                     verbose=self.config.verbose)
        print('Getting label_encodings:\n Previous computed num_classes =',self.num_classes,'\n num_classes based on label_encodings =',len(self.label_encodings))
        return self.label_encodings
    
    
    def stage_tfrecords(self):
        '''
        Looks for tfrecords corresponding to DatasetConfig parameters, if nonexistent then proceeds to create tfrecords.
        '''
        self.root_dir = self.tfrecord_root_dir
        dataset_name = self.config.dataset_name
        
        tfrecords = self.dataset_builder.recursive_search(
                                                          self.root_dir,
                                                          subdirs=[dataset_name, f'num_channels-3_thresh-{self.config.low_class_count_thresh}']
                                                         )
#                                                           subdirs=[dataset_name, f'num_channels-{self.num_channels}_thresh-{self.config.low_class_count_thresh}'])
        if tfrecords is None:
            return create_tfrecords(self.config)
        else:
            coder = TFRecordCoder(self.data_splits['train'],
                                 self.root_dir,
                                 target_size=self.target_size, 
                                 num_channels=self.num_channels, 
                                 num_classes=self.num_classes)
            return coder, tfrecords
    
    
    def get_data_loader(self, subset='train'):
        assert subset in self.tfrecord_files.keys()
        
        tfrecord_paths = self.tfrecord_files[subset]
        config = self.config
    
        data = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
                    .apply(lambda x: tf.data.TFRecordDataset(x)) \
                    .map(self.coder.decode_example, num_parallel_calls=AUTOTUNE) \
                    .map(self.preprocessing, num_parallel_calls=AUTOTUNE)

#         if self.visualize:
#             def convert_to_uint(x,y):
#                 return tf.image.convert_image_dtype(x,dtype=tf.uint8),y
#             print('Converting data to uint8 for visualization')
#             data = data.map(convert_to_uint, num_parallel_calls=AUTOTUNE)

        if self.grayscale == True:
            if self.num_channels==3:
                data = data.map(rgb2gray_3channel, num_parallel_calls=AUTOTUNE)
            elif self.num_channels==1:
                data = data.map(rgb2gray_1channel, num_parallel_calls=AUTOTUNE)
#         if self.preprocessing == 'imagenet':
#             data = data.map(imagenet_mean_subtraction, num_parallel_calls=AUTOTUNE)
        
        if subset == 'train':
            data = data.shuffle(buffer_size=config.buffer_size, seed=config.seed)
            
            if config.augment_images == True:
                data = data.map(self.augmentors.rotate, num_parallel_calls=AUTOTUNE) \
                           .map(self.augmentors.flip, num_parallel_calls=AUTOTUNE) #\
#                            .map(self.augmentors.color, num_parallel_calls=AUTOTUNE)
            
        data = data.batch(config.batch_size, drop_remainder=False) \
                   .repeat() \
                   .prefetch(AUTOTUNE)
        
        return data
    
    def get_model_params(self, subset='train'):
        metadata = self.metadata_splits[subset]
        config = self.config
        
        model_params = ModelConfig(
                                   model_name=config.model_name,
                                   num_classes=metadata['num_classes'],
                                   frozen_layers=config.frozen_layers,
                                   input_shape=(*config.target_size,config.num_channels),
                                   base_learning_rate=config.base_learning_rate,
                                   regularization=config.regularization
                                   )
        return model_params
        
        
#         params = {'name':config.model_name,
#                   'num_classes':metadata['num_classes'],
#                   'frozen_layers':config.frozen_layers,
#                   'input_shape':(*config.target_size,config.num_channels),
#                   'base_learning_rate':config.base_learning_rate,
#                   'regularization':config.regularization
#                  }
        return params
    
    def get_fit_params(self):
        params = {'steps_per_epoch' : self.metadata_splits['train']['num_samples']//self.config.batch_size,
                  'validation_steps' : self.metadata_splits['val']['num_samples']//self.config.batch_size,
                  'epochs' : self.config.num_epochs
                 }
        return params
    
    
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

    
