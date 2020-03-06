import os
from pyleaves.utils import ensure_dir_exists


class Config:
    """DEPRECATED
    Config class with global project variables."""

    def __init__(self, dataset_name='Fossil', local_tfrecords=None, **kwargs):
        """Global config file for normalization experiments."""
        self.dataset_name = 'Fossil'
        self.project_directory = '/media/data/jacob/fossil_experiments/'
        self.tfrecords = os.path.join(
            self.project_directory,
            'tf_records')  # Alternative, slow I/O path for tfrecords.
        self.local_tfrecords = local_tfrecords or self.tfrecords  # Primary path for tfrecords.
        self.checkpoints = os.path.join(
            self.project_directory,
            'checkpoints')
        self.summaries = os.path.join(
            self.project_directory,
            'summaries')
        self.experiment_evaluations = os.path.join(
            self.project_directory,
            'experiment_evaluations')
        self.plots = os.path.join(
            self.project_directory,
            'plots')
        self.results = 'results'
        self.log_dir = os.path.join(self.project_directory, 'logs')

        # Create directories if they do not exist
        check_dirs = [
            self.tfrecords,
            self.local_tfrecords,
            self.checkpoints,
            self.summaries,
            self.experiment_evaluations,
            self.plots,
            self.log_dir,
        ]
        [ensure_dir_exists(x) for x in check_dirs]
        
        self.seed = 1085

    def __getitem__(self, name):
        """Get item from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains field."""
        return hasattr(self, name)



class BaseConfig(dict):
    
    def __init__(self,**kwargs):
        '''
        Base class for storing experiment configuration parameters
        '''
        for k, v in kwargs.items():
            self.update({k:v})
            self.__dict__.update({k:v})
        
    def init_directories(self, dirs):
        
        for dir_name, dir_path in dirs.items():
            ensure_dir_exists(dir_path)
        
        
class DatasetConfig(BaseConfig):
    
    def __init__(self,
                 dataset_name='PNAS',
                 label_col='family',
                 target_size=(224,224),
                 num_channels=3,
                 grayscale=False,
                 low_class_count_thresh=3,
                 data_splits={'val_size':0.2,'test_size':0.2},
                 tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
                 num_shards=10,
                 input_format=tuple):
        '''
        if grayscale==True and num_channels==3:
            Convert to grayscale 1 channel then duplicate to 3 channels for full [batch,h,w,3] shape
        '''
        
        self.dirs = {'tfrecord_root_dir':tfrecord_root_dir}
        self.init_directories(self.dirs)
        
        super().__init__(dataset_name=dataset_name,
                         label_col=label_col,
                         target_size=target_size,
                         num_channels=num_channels,
                         grayscale=grayscale,
                         low_class_count_thresh=low_class_count_thresh,
                         data_splits=data_splits,
                         tfrecord_root_dir=tfrecord_root_dir,
                         num_shards=num_shards,
                         input_format=input_format,
                         dirs=self.dirs)
        
class TrainConfig(BaseConfig):
    
    def __init__(self,
                 model_name='shallow',
                 model_dir='/media/data_cifs/jacob/Fossil_Project/models',
                 batch_size=32,
                 frozen_layers=(0,-4),
                 base_learning_rate=0.001,
                 buffer_size=1000,
                 num_epochs=50,
                 preprocessing=None,
                 augment_images=False,
                 augmentations=['rotate','flip','color'],
                 regularization=None,
                 seed=3,
                 verbose=True):
        self.dirs = {'model_dir':model_dir}
        self.init_directories(self.dirs)
        
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         batch_size=batch_size,
                         frozen_layers=frozen_layers,
                         base_learning_rate=base_learning_rate,
                         buffer_size=buffer_size,
                         num_epochs=num_epochs,
                         preprocessing=preprocessing,
                         augment_images=augment_images,
                         augmentations=augmentations,
                         regularization=regularization,
                         seed=seed,
                         verbose=verbose,
                         dirs=self.dirs)
        '''
        
        preprocessing : Can be any of [None, 'imagenet']
            If 'imagenet', subtract hard-coded imagenet mean from each of the RGB channels
        
        '''

        
class ModelConfig(BaseConfig):

    def __init__(self,
                 model_name='vgg16',
                 model_dir='/media/data_cifs/jacob/Fossil_Project/models',
                 num_classes=1000,
                 frozen_layers=(0,-4),
                 input_shape=(224,224,3),                 
                 base_learning_rate=0.0001,
                 grayscale=False,
                 regularization=None,
                 seed=3,
                 verbose=True):
        
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         num_classes=num_classes,
                         frozen_layers=frozen_layers,
                         input_shape=input_shape,
                         base_learning_rate=base_learning_rate,
                         regularization=regularization,
                         seed=seed,
                         verbose=verbose)
        '''
        Config for feeding to subclasses of BaseModel for building/loading/saving models
        '''
        
        
        
class ExperimentConfig(BaseConfig):
    
    def __init__(self, 
                 dataset_config=DatasetConfig(),
                 train_config=TrainConfig()):
        
        self.dataset_config = dataset_config
        self.train_config = train_config
            
        self.dirs = {**dataset_config.dirs,**train_config.dirs}
        
        train_config.dirs.update(dataset_config.pop('dirs'))
        
        super().__init__(**dataset_config, **train_config)
        
        

train_config=TrainConfig()
dataset_config=DatasetConfig()        
        
class MLFlowConfig(BaseConfig):
    '''
    Config for packaging parameters used in managing MLFlow servers/loggers
    '''
    def __init__(self,
                 experiment_name='default',
                 tracking_uri=r'/media/data/jacob/Fossil_Project/mlflow',
                 artifact_uri=r'/media/data/jacob/Fossil_Project/tfrecord_data'):
        self.experiment_name = experiment_name