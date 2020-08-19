import json
import numpy as np
import funcy
import os
from pyleaves.utils import ensure_dir_exists
from pyleaves.utils.csv_utils import gather_experiment_data
from toolz import diff



class BaseConfig(dict):

    def __init__(self,*args, **kwargs):
        '''
        Base class for storing experiment configuration parameters
        '''

        if len(args):
            self.args = args
        for k, v in kwargs.items():
            if (type(v) is dict) and (k in self.keys()):
                if (type(self.__dict__[k]) is dict):
                    self.__dict__[k].update(v)
            else:
                self.update({k:v})
                self.__dict__.update({k:v})

    def init_directories(self, dirs):
        for dir_name, dir_path in dirs.items():
            ensure_dir_exists(dir_path)

    @classmethod
    def load_config(cls, filepath):
        assert os.path.isfile(filepath)
        assert filepath.endswith('json')
        with open(filepath, 'r') as file:
            data = json.load(file)

        return cls(**data)

    def save_config(self, filepath):
        base_dir = os.path.dirname(filepath)
        ensure_dir_exists(base_dir)
        with open(filepath, 'w') as file:
            json.dump(self, file)

    def __eq__(self, *seqs):
        return not any(diff(self, *seqs, default=object()))

    def __sub__(self, *args, **kwargs):
        return diff(self, *args, **kwargs, default=object())


    def __repr__(self):
        return json.dumps(self.__dict__,indent=4)








class DatasetConfig(BaseConfig):

    def __init__(self,
                 dataset_name='PNAS',
                 label_col='family',
                 target_size=(224,224),
                 num_channels=3,
                 grayscale=False,
                 low_class_count_thresh=3,
                 data_splits={'val_size':0.2,'test_size':0.2},
                 num_shards=10,
                 tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
                 data_db_path=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db',
                 input_format='tuple',
                 dirs={},
                 **kwargs):
        '''
        if grayscale==True and num_channels==3:
            Convert to grayscale 1 channel then duplicate to 3 channels for full [batch,h,w,3] shape
        '''

        self.dirs = {'tfrecord_root_dir':tfrecord_root_dir, **dirs}
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
                         data_db_path=data_db_path,
                         dirs=self.dirs,
                        **kwargs)



print()


class CSVDomainDataConfig(BaseConfig):

    def __init__(self,
                 experiment_name="2-domain_experiments",
                 run="Leaves_Fossil",
                 domain="", #"target",
                 dataset_name="", #"Fossil",
                 subsets=[],
                 num_channels=3,
                 target_size='default', #(299,299),
                 grayscale=False,
                 color_type='grayscale',
                 low_class_count_thresh=3,
                 data_splits={'val_size':0.2,'test_size':0.2},
                 num_shards=10,
                 label_mappings="",
                 meta="",
                 data={}):
        self.run = run
        self.dataset_name = dataset_name

        super().__init__(experiment_name = experiment_name,
                         run = run,
                         domain = domain,
                         dataset_name = dataset_name,
                         subsets = subsets,
                         num_channels=num_channels,
                         target_size=target_size,
                         grayscale=grayscale,
                         color_type=color_type,
                         low_class_count_thresh=low_class_count_thresh,
                         data_splits=data_splits,
                         num_shards=num_shards,
                         label_mappings = label_mappings,
                         meta = meta,
                         data = data)

CSVFrozenDomainDataConfig = CSVDomainDataConfig



class CSVFrozenRunDataConfig(BaseConfig):

    def __init__(self,
                 experiment_name="2-domain_experiments",
                 run="Leaves-Fossil",
                 experiment_root_dir=r"/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments",
                 tfrecord_root_dir=r"/media/data/jacob/Fossil_Project/tfrecord_data",
                 low_class_count_thresh=10,
                 data_configs={}):
        '''
        Container class for holding 1 or more CSVFrozenDomainDataConfig instances for one run in a larger experiment.

        e.g.
        LeavesFossilDomainTransferRunConfig = CSVFrozenRunDataConfig(experiment_name="2-domain_experiments",
                                                                     run="Leaves_Fossil",
                                                                     domain_data_configs={
                                                                     'source':CSVFrozenDomainDataConfig(domain='source',dataset_name='Leaves'),
                                                                     'target':CSVFrozenDomainDataConfig(domain='target',dataset_name='Fossil')
                                                                    })
        Leaves_1DomainRunConfig = CSVFrozenRunDataConfig(experiment_name="single-domain_experiments",
                                                         run="Leaves",
                                                         domain_data_configs={
                                                          'Leaves':CSVFrozenDomainDataConfig(domain='Leaves',dataset_name='Leaves')
                                                         })



        '''
        super().__init__(experiment_name = experiment_name,
                         run = run,
                         low_class_count_thresh=low_class_count_thresh,
                         data_configs = data_configs,
                         target_size='default'
                         )

        self.experiment_root_dir = experiment_root_dir
        self.tfrecord_root_dir = tfrecord_root_dir
        self.filepath = os.path.join(experiment_root_dir, run, 'frozen-data-config.json')

        self.domains = [key for key in self.data_configs.keys()]
        self.dataset_names = [d.dataset_name for d in self.data_configs.values()]
        self.color_type = [d.color_type for d in self.data_configs.values()][0]

        self.dirs = {'experiment_root_dir':experiment_root_dir,
                     'tfrecord_root_dir':tfrecord_root_dir}

    def init_config_file(self):
        if os.path.isfile(self.filepath):
            try:
                self.load_config(self.filepath)
                print(f"Loaded {self.run} data config from {self.filepath}")
            except:
                print('Loading failed, saving new config file from input parameters')
        self.save_config(self.filepath)
        print(f"Saved {self.run} data config to {self.filepath}")



def get_records_attribute(experiment_records, attribute_key='run'):
    '''Select all unique values of attribute corresponding to attribute_key'''
    record_attribute_values=np.unique([*funcy.pluck(attribute_key, experiment_records)])

    return record_attribute_values.tolist()




def get_all_experiment_attributes(experiment_records, by='attribute'):

    attributes = {}
    if by=='attribute':
        keys = list(experiment_records[0].keys())
        keys = [k for k in keys if not k.endswith('_data')]

        for k in keys:
            attributes.update({k:print_records_attribute(experiment_records, attribute_key=k)})
            print(json.dumps({k:attributes[k]},indent=2))
    elif by=='run':
        attributes=experiment_records
        for r in experiment_records:
            print(json.dumps(r, indent=4))

    return attributes





if False:

    experiment_root_dir = r'/media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments'

    experiment_records = gather_experiment_data(experiment_root_dir, return_type='records')



    recs = get_all_experiment_attributes(experiment_records,by='attribute')
    attributes = get_all_experiment_attributes(experiment_records,by='run')

    leavesFossilDomainTransferRunConfig = CSVFrozenRunDataConfig(experiment_name="2-domain_experiments",
                                                                run="Leaves_Fossil",
                                                                experiment_root_dir=r"/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments",
                                                                domain_data_configs={
                                                                'source':CSVFrozenDomainDataConfig(domain='source',dataset_name='Leaves'),
                                                                'target':CSVFrozenDomainDataConfig(domain='target',dataset_name='Fossil')
                                                                })
    leavesFossilDomainTransferRunConfig.init_config_file()

    #####################################################################################################
    #####################################################################################################


    # import pdb; pdb.set_trace()
    leavesFossilDomainTransferRunConfig = CSVFrozenRunDataConfig(experiment_name="2-domain_experiments",
                                                                run="Leaves_Fossil",
                                                                experiment_root_dir=r"/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments",
                                                                domain_data_configs={
                                                                'source':CSVFrozenDomainDataConfig(domain='source',dataset_name='Leaves'),
                                                                'target':CSVFrozenDomainDataConfig(domain='target',dataset_name='Fossil')
                                                                })
    leavesFossilDomainTransferRunConfig.init_config_file()



    leaves_1DomainRunConfig = CSVFrozenRunDataConfig(experiment_name="single-domain_experiments",
                                                             run="Leaves",
                                                             experiment_root_dir=r"/media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments",
                                                             domain_data_configs={
                                                              'Leaves':CSVFrozenDomainDataConfig(domain='Leaves',dataset_name='Leaves')
                                                             })

    leaves_1DomainRunConfig.init_config_file()





# class CSVFrozenDatasetConfig(BaseConfig):
#     '''
#     Config for loading data from CSV files distributed according to a hierarchy defined in pyleaves/utils/csv_utils.py
#     '''
# #     def __init__(self,
# #                  experiment_name="2-domain_experiments",
# #                  run="Leaves-Fossil",
# #                  domain="target",
# #                  dataset_name="Fossil",
# #                  subset="val",
# #                  label_mappings="/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments/Leaves-Fossil/target_Fossil/label_mappings.csv",
# #                  metadata="/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments/Leaves-Fossil/target_Fossil/meta.csv",
# #                  train_data="/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments/Leaves-Fossil/target_Fossil/train_data.csv",
# #                  val_data="/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments/Leaves-Fossil/target_Fossil/val_data.csv",
# #                  test_data="/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments/Leaves-Fossil/target_Fossil/test_data.csv"


#                  label_col='family',
#                  target_size=None,
#                  num_channels=None,
#                  grayscale=False,
#                  low_class_count_thresh=10,
#                  num_shards=10,
#                  experiment_root_dir=r'/media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments',
#                  tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
#                  input_format='tuple',
#                  dirs={}):
#         '''
#         if grayscale==True and num_channels==3:
#             Convert to grayscale 1 channel then duplicate to 3 channels for full [batch,h,w,3] shape
#         '''

#         gather_domain_data(experiment_root_dir, run, domain)

#         self.dirs = {'tfrecord_root_dir':tfrecord_root_dir, **dirs}
#         self.init_directories(self.dirs)

#         super().__init__(dataset_name=dataset_name,
#                          label_col=label_col,
#                          target_size=target_size,
#                          num_channels=num_channels,
#                          grayscale=grayscale,
#                          low_class_count_thresh=low_class_count_thresh,
#                          data_splits=data_splits,
#                          tfrecord_root_dir=tfrecord_root_dir,
#                          num_shards=num_shards,
#                          input_format=input_format,
#                          dirs=self.dirs)




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
                 x_col='x',
                 y_col='y',
                 augment_images=False,
                 augmentations=['rotate','flip','color'],
                 regularization=None,
                 seed=3,
                 verbose=True,
                 dirs={}):
        self.dirs = {'model_dir':model_dir, **dirs}
        self.init_directories(self.dirs)

        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         batch_size=batch_size,
                         frozen_layers=frozen_layers,
                         base_learning_rate=base_learning_rate,
                         buffer_size=buffer_size,
                         num_epochs=num_epochs,
                         preprocessing=preprocessing,
                         x_col=x_col,
                         y_col=y_col,
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
                 verbose=True,
                 dirs={}):

        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         num_classes=num_classes,
                         frozen_layers=frozen_layers,
                         input_shape=input_shape,
                         base_learning_rate=base_learning_rate,
                         regularization=regularization,
                         seed=seed,
                         verbose=verbose,
                         dirs=dirs)
        '''
        Config for feeding to subclasses of BaseModel for building/loading/saving models
        '''



class ExperimentConfig(BaseConfig):

    def __init__(self,
                 dataset_config=None,
                 train_config=None,
                 *args,
                 **kwargs):

        self.dataset_config = dataset_config or DatasetConfig()
        self.train_config = train_config or TrainConfig()

        self.dirs = {**dataset_config.dirs,**train_config.dirs}
        self.__dict__.update(**self.dirs)
        dataset_config.pop('dirs',{})
        train_config.pop('dirs',{})

        if 'domains' in dataset_config.__dict__.keys():
            self.domains = dataset_config.domains

        if 'dataset_names' in dataset_config.__dict__.keys():
            self.dataset_names = dataset_config.dataset_names


        super().__init__(**dataset_config, **train_config, **kwargs, **{'dirs':self.dirs})



# train_config=TrainConfig()
# dataset_config=DatasetConfig()


# loaded_train_config == train_config #dataset_config


# config = ExperimentConfig(dataset_config, train_config)


# train_config.save_config(os.path.expanduser('~/experiments/configs/experiment_0/train_config.json'))
# dataset_config.save_config(os.path.expanduser('~/experiments/configs/experiment_0/dataset_config.json'))
# config.save_config(os.path.expanduser('~/experiments/configs/experiment_0/experiment_config.json'))


# loaded_train_config = train_config.load_config(os.path.expanduser('~/experiments/configs/experiment_0/train_config.json'))
# loaded_dataset_config = dataset_config.load_config(os.path.expanduser('~/experiments/configs/experiment_0/dataset_config.json'))
# loaded_config = config.load_config(os.path.expanduser('~/experiments/configs/experiment_0/experiment_config.json'))


# a=set(train_config).symmetric_difference(loaded_train_config)

# list(loaded_train_config - train_config)
# list(loaded_train_config - dataset_config)







class MLFlowConfig(BaseConfig):
    '''
    Config for packaging parameters used in managing MLFlow servers/loggers
    '''
    def __init__(self,
                 experiment_name='default',
                 tracking_uri=r'/media/data/jacob/Fossil_Project/mlflow',
                 artifact_uri=r'/media/data/jacob/Fossil_Project/tfrecord_data'):
        self.experiment_name = experiment_name







#####################################################################



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
