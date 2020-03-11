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
from pyleaves.leavesdb.tf_utils.create_tfrecords import main as create_tfrecords

from pyleaves.config import DatasetConfig, TrainConfig, ModelConfig, ExperimentConfig
from pyleaves.models.resnet import ResNet, ResNetGrayScale
from pyleaves.models.vgg16 import VGG16, VGG16GrayScale
from stuf import stuf



class SQLManager:
    '''
    ETL pipeline for preparing data from Leavesdb SQLite database and staging TFRecords for feeding into data loaders.
    
    Meant to be subclassed for use with BaseTrainer and future Trainer classes.
    '''
    def __init__(self, experiment_config, label_encoder=None, src_db=pyleaves.DATABASE_PATH):

        self.config = experiment_config
        self.configs = {'experiment_config':self.config}
        self.name = ''
        self.tfrecord_root_dir = self.config['tfrecord_root_dir']
        self.model_dir = self.config['model_dir']
        self.data_db_path = self.config['data_db_path']
        self.init_dirs()
        if label_encoder is None:
            self.encoder = LabelEncoder()
        else:
            self.encoder = label_encoder
        self.src_db = src_db
        
    def init_dirs(self):
        if 'label_encodings_filepath' in self.config:
            assert validate_filepath(self.config['label_encodings_filepath'],file_type='json')
            self.label_encodings_filepath = self.config['label_encodings_filepath']
        else:
            self.label_encodings_filepath = os.path.join(self.model_dir,f'{self.name}-label_encodings.json')

        self.config['label_encodings_filepath'] = self.label_encodings_filepath        

    def extract(self):
        self.db_df = self.db_query(dataset_name=self.config.dataset_name)
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
        
        
    def db_query(self, dataset_name='Fossil', label_col='family'):
        '''
        Query all filenames and labels associated with dataset_name
        
        Return:
            self.db_df, pd.DataFrame:
                DataFrame containing columns ['path','label']
        '''
#         import pdb; pdb.set_trace()
        self.local_db = leavesdb.init_local_db(src_db=self.src_db)
        self.db = dataset.connect(f'sqlite:///{self.local_db}', row_type=stuf)
        self.db_df = pd.DataFrame(leavesdb.db_query.load_data(self.db, y_col=label_col, dataset=dataset_name))
        
        return self.db_df
    
    def db_filter(self, db_df, label_col='family', verbose=False):
        '''
        Function to apply preprocessing to output of db_query, prior to conversion of images to TFRecord. 
        
        '''
#         print('FILTERING: DB_DF.COLUMNS = ',db_df.columns)
        threshold = self.config.low_class_count_thresh
        db_df = filter_low_count_labels(db_df, threshold=threshold, verbose=verbose)
#         if os.path.isfile(self.config['label_encodings_filepath']):
#             filepath=self.config['label_encodings_filepath']
#         else:
#             filepath=None
        if len(self.encoder)==0:
            self.encoder.merge_labels(labels=list(db_df[label_col]))
        self.encoder.save_labels(self.config['label_encodings_filepath'])
        
        db_df = self.encoder.filter(db_df, label_col=label_col)
        
        self.x = db_df['path'].values.reshape((-1,1))
        self.y = np.array(self.encoder.transform(db_df[label_col]))
        
#         self.db_df['label']=self.y
#         db_df = encode_labels(db_df)
#         self.x = db_df['path'].values.reshape((-1,1))
#         self.y = db_df['label'].values
        
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
    
    def get_label_encodings(self, db_df, label_col='family', verbose=False):
        if 'label_encodings_filepath' in self.config:
            self.label_encodings.merge_labels()
            # Load encodings from file
#             self.label_encodings = leavesdb.db_query.load_label_encodings_from_file()
        if True:
            self.label_encodings = leavesdb.db_query.get_label_encodings(db_df,
                                                                         y_col=label_col, 
                                                                         low_count_thresh=self.config.low_class_count_thresh, 
                                                                         verbose=verbose) #self.config.verbose)
        if verbose: print('Getting label_encodings:\n Previous computed num_classes =',self.num_classes,'\n num_classes based on label_encodings =',len(self.label_encodings))
        return self.label_encodings
    
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


    
    
class BaseTrainer(SQLManager):
    
    def __init__(self, experiment_config, label_encoder=None, src_db=pyleaves.DATABASE_PATH):
        
        super().__init__(experiment_config, label_encoder=label_encoder, src_db=src_db)

        if self.config.preprocessing:
            self.preprocessing = get_keras_preprocessing_function(self.config.model_name, self.config.input_format)
            
        self.augmentors = ImageAugmentor(self.config.augmentations, seed=self.config.seed)
        self.grayscale = self.config.grayscale

        self.extract()
        self.transform()
        self.load()
        
    
    def get_data_loader(self, subset='train'):
        assert subset in self.tfrecord_files.keys()
        
        tfrecord_paths = self.tfrecord_files[subset]
        config = self.config
    
        data = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
                    .apply(lambda x: tf.data.TFRecordDataset(x)) \
                    .map(self.coder.decode_example, num_parallel_calls=AUTOTUNE) \
                    .map(self.preprocessing, num_parallel_calls=AUTOTUNE)

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
   

        
#     def update_model(self, model):
#         '''Simply for adding a tf.keras.models.Model for Trainer to manage'''
#         self.model = model
    
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

    
