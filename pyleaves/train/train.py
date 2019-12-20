"""
Created on Tue Dec 17 03:23:32 2019

script: pyleaves/pyleaves/train/train.py

@author: JacobARose
"""
import numpy as np
import os

import tensorflow as tf

gpus = tf.config.experimental.get_visible_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.get_visible_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

# tf.enable_eager_execution()


from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, one_hot_encode_labels #, one_hot_decode_labels
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves.leavesdb.db_query import get_label_encodings as _get_label_encodings, load_from_db
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split as _train_val_test_split

from pyleaves.models.keras_models import build_model #vgg16_base, xception_base, resnet_50_v2_base, resnet_101_v2_base, shallow
from pyleaves.train.callbacks import get_callbacks as _get_callbacks
from pyleaves.utils import ensure_dir_exists



class Experiment:

    def __init__(self,
                 model_name='shallow',
                 dataset_name='PNAS',
                 experiment_name='PNAS_shallow',
                 data_root_dir=r'/media/data/jacob',
                 experiment_root_dir=r'/media/data/jacob/experiments',
                 input_shape=(224,224,3),
                 low_class_count_thresh=0,
                 frozen_layers=(0,-4),
                 base_learning_rate=0.001,
                 batch_size=64,
                 val_size=0.3,
                 test_size=0.3,
                 seed=17):
        '''
        Arguments:
            model_name, str : model to build from definitions stored in pyleaves.model.keras_models.py
            dataset_name, str : dataset to load from TFRecords generated by pyleaves.leavesdb.tf_utils.create_tfrecords.py
            experiment_name, str : Chosen unique label for storing experiment weights, logs and results. Should contain dataset_model and any relevant metadata
            data_root_dir, str : Absolute root directory containing subdirs for each dataset's tfrecords
                   e.g. data_root_dir = /media/data/jacob/
                                                         |Fossil/
                                                                |train/
                                                                      |train-00000-of-00010.tfrecord
                                                                      |train-00001-of-00010.tfrecord
                                                                      ...
                                                                      |train-00009-of-00010.tfrecord
                                                                      |train-00010-of-00010.tfrecord
                                                                |val/
                                                                    |...
                                                                |test/
                                                                     |...
                                                         |PNAS/
            experiment_root_dir, str : root dir in which to store the experiment's results directory named after experiment_name
            input_shape, tuple : input shape for images with dimension order (h, w, c)
            low_class_count_thresh, int :
            frozen_layers, tuple :
            base_learning_rate, float :
            batch_size, int :
            val_size, float :
            test_size, float :
            seed, int :


        '''

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.root_dir = os.path.join(data_root_dir, dataset_name)
#         self.output_dir = os.path.join(output_dir,model_name)
        self.experiment_root_dir = experiment_root_dir

        self.input_shape = input_shape
        self.low_class_count_thresh = low_class_count_thresh
        self.frozen_layers = frozen_layers
        self.base_learning_rate = base_learning_rate

        self.batch_size = batch_size
        self.seed = seed
        self.val_size=val_size
        self.test_size=test_size
        self.verbose=True

        
        self.initialize_experiment_paths()
        self.load_metadata_from_db()
        self.format_loaded_metadata()
        self.data_subsets = self.train_val_test_split()

        self.label_encodings = self.get_label_encodings()
        
        self.dataset_builder = DatasetBuilder(root_dir=self.root_dir,
                                              num_classes=self.num_classes,
                                              batch_size=self.batch_size,
                                              seed=self.seed)
        self.tfrecord_splits = self.dataset_builder.collect_subsets(self.root_dir)


        self.callbacks = self.get_callbacks()

    def initialize_experiment_paths(self):
        exp_dir = os.path.join(self.experiment_root_dir,self.experiment_name)
        self.experiment_paths = {
            'experiment_dir':exp_dir,
            'weights_filepath':os.path.join(exp_dir,self.experiment_name+'_model-weights.hs'),
            'logs_dir':os.path.join(exp_dir,self.experiment_name+'-logs_dir')
        }
        
        
        
    def load_metadata_from_db(self):
        self.metadata, self.db = load_from_db(dataset_name=self.dataset_name)
        return self.metadata
    def format_loaded_metadata(self):
        metadata=self.metadata
        low_class_count_thresh=self.low_class_count_thresh
        verbose=self.verbose

        data_df = encode_labels(metadata)
        data_df = filter_low_count_labels(data_df, threshold=low_class_count_thresh, verbose = verbose)
        data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)
        paths = data_df['path'].values.reshape((-1,1))
        labels = data_df['label'].values
        self.x = paths
        self.y = labels
        return self.x, self.y

    def train_val_test_split(self):
        x=self.x
        y=self.y
        test_size=self.test_size
        val_size=self.val_size
        verbose=self.verbose

        self.data_subsets, self.metadata_subsets = _train_val_test_split(x, y, test_size=test_size, val_size=val_size, verbose=verbose)
        return self.data_subsets

    def get_label_encodings(self):
        dataset=self.dataset_name
        low_count_thresh=self.low_class_count_thresh

        self.label_encodings = _get_label_encodings(dataset=dataset, low_count_thresh=low_count_thresh)
        self.num_classes = len(self.label_encodings)
        return self.label_encodings


    def build_model(self):
        model_name=self.model_name
        num_classes=self.num_classes
        frozen_layers=self.frozen_layers
        input_shape=self.input_shape
        base_learning_rate=self.base_learning_rate

        return build_model(name=model_name,
                           num_classes=num_classes,
                           frozen_layers=frozen_layers,
                           input_shape=input_shape,
                           base_learning_rate=base_learning_rate)

    def __setattr__(self, name, val):
        self.__dict__[name] = val

    def set_model_params(self,
                         model_name=None,
                         num_classes=None,
                         frozen_layers=None,
                         input_shape=None,
                         base_learning_rate=None):

        if model_name: self.model_name=model_name
        if num_classes: self.num_classes=num_classes
        if frozen_layers: self.frozen_layers=frozen_layers
        if input_shape: self.input_shape=input_shape
        if base_learning_rate: self.base_learning_rate=base_learning_rate

    def get_callbacks(self):
        return _get_callbacks(weights_best=self.experiment_paths['weights_filepath'],
                              logs_dir=self.experiment_paths['logs_dir'],
                              restore_best_weights=False)



    def get_dataset(self,subset='train', batch_size=32, num_classes=None):
        if num_classes is None:
            num_classes=self.num_classes
        return self.dataset_builder.get_dataset(subset=subset, batch_size=batch_size, num_classes=num_classes)






# def run_experiment():
if True:
    batch_size=32
    num_epochs=100
    model_name='resnet_50_v2'#'vgg16'#'xception'#'shallow',
    dataset_name='Fossil'
    experiment_name='_'.join([dataset_name,model_name])
    data_root_dir='/media/data/jacob'
    experiment_root_dir='/media/data/jacob/experiments'
    input_shape=(224,224,3)
    low_class_count_thresh=3
    frozen_layers=(0,-4)
    base_learning_rate=0.001

    experiment = Experiment(model_name=model_name,
                            dataset_name=dataset_name,
                            experiment_name=experiment_name,
                            data_root_dir=data_root_dir,
                            experiment_root_dir=experiment_root_dir,
                            input_shape=input_shape,
                            low_class_count_thresh=low_class_count_thresh,
                            frozen_layers=frozen_layers,
                            base_learning_rate=base_learning_rate)

    model = experiment.build_model()

    callbacks = experiment.callbacks


    print('num_classes = ',experiment.num_classes)

    train_data = experiment.get_dataset(subset='train', batch_size=batch_size)
    val_data = experiment.get_dataset(subset='val', batch_size=batch_size)
    test_data = experiment.get_dataset(subset='test', batch_size=batch_size)

    steps_per_epoch = experiment.metadata_subsets['train']['num_samples']//batch_size
    validation_steps = experiment.metadata_subsets['val']['num_samples']//batch_size

    history = model.fit(train_data,
             steps_per_epoch = steps_per_epoch,
             epochs=num_epochs,
             validation_data=val_data,
             validation_steps=validation_steps,
             callbacks=callbacks
             )

# import matplotlib.pyplot as plt
# plt.figure(); plt.plot(history.history['loss'],'b.');plt.plot(history.history['val_loss'],'r--')
# plt.legend(['loss','val_loss'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', type=str, default ='/home/irodri15/Lab/leafs/data/processed/full_dataset_processed.csv', help='input file with names')
#     parser.add_argument('--output_folder', type=str, default='SAVING', help='how to save this training')
#     parser.add_argument('--gpu',default =1, help= 'what gpu to use, if "all" try to allocate on every gpu'  )
#     parser.add_argument('--gpu_fraction', type=float, default =0.9, help= 'how much memory of the gpu to use' )
#     parser.add_argument('--pre_trained_weights',type=str,default= None,help='Pre_trained weights ')
#     parser.add_argument('--resolution',default=768,help='resolution if "all" will use all the resolutions available')
#     parser.add_argument('--splits',type=int,default=10,help='how many splits use for evaluation')


#     args = parser.parse_args()
#     fraction = float(args.gpu_fraction)
#     gpu = int(args.gpu)
#     path= args.path
#     output = args.output_folder
#     output_folder =args.output_folder
#     weights = args.pre_trained_weights
#     splits = args.splits

#     configure(gpu,fraction)

#     Data=LeafData(path)
#     if resolution == 'all':
#         Data.multiple_resolution()
#     else:
#         Data.single_resolution(resolution)
#     X,y = Data.X, Data.Y

#     classes =len(np.unique(y))
