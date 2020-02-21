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
                 channels=3,
                 low_class_count_thresh=3,
                 data_splits={'val_size':0.2,'test_size':0.2},
                 tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
                 num_shards=10):
        self.dirs = {'tfrecord_root_dir':tfrecord_root_dir}
        self.init_directories(self.dirs)
        
        super().__init__(dataset_name=dataset_name,
                         label_col=label_col,
                         target_size=target_size,
                         channels=channels,
                         low_class_count_thresh=low_class_count_thresh,
                         data_splits=data_splits,
                         tfrecord_root_dir=tfrecord_root_dir,
                         num_shards=num_shards,
                         dirs=self.dirs)
        
class TrainConfig(BaseConfig):
    
    def __init__(self,
                 model_name='shallow',
                 batch_size=32,
                 frozen_layers=(0,-4),
                 base_learning_rate=0.001,
                 buffer_size=1000,
                 num_epochs=50,
                 preprocessing=None,
                 augment_images=False,
                 seed=3):
        super().__init__(model_name=model_name,
                         batch_size=batch_size,
                         frozen_layers=frozen_layers,
                         base_learning_rate=base_learning_rate,
                         buffer_size=buffer_size,
                         num_epochs=num_epochs,
                         preprocessing=preprocessing,
                         augment_images=augment_images,
                         seed=seed)
        '''
        
        preprocessing : Can be any of [None, 'imagenet']
            If 'imagenet', subtract hard-coded imagenet mean from each of the RGB channels
        
        '''
        
        
class ExperimentConfig(BaseConfig):
    
    def __init__(self, 
                 dataset_config,
                 train_config):
        self.dataset_config = dataset_config
        self.train_config = train_config
        
        super().__init__(**dataset_config, **train_config)
        
        
        
        
        
class MLFlowConfig(BaseConfig):
    '''
    Config for packaging parameters used in managing MLFlow servers/loggers
    '''
    def __init__(self,
                 experiment_name='default',
                 tracking_uri=r'/media/data/jacob/Fossil_Project/mlflow',
                 artifact_uri=r'/media/data/jacob/Fossil_Project/tfrecord_data'):
        self.experiment_name = experiment_name