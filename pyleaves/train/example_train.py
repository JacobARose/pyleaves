"""
Created on Mon Feb 10 03:23:32 2019

script: pyleaves/pyleaves/train/example_train.py

@author: JacobARose
"""


def main(dataset_name='PNAS',
         model_name='vgg16',
         experiment_dir=r'/media/data/jacob/Fossil_Project/vgg16/PNAS',
         gpu_ids = [0],
         tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
         batch_size=64,
         target_size=(224,224),
         base_learning_rate=0.001,
         num_epochs=100,
         preprocessing='imagenet',
         augment_images=False):
            
#     reset_keras_session()
    tf.reset_default_graph()
    dataset_config = DatasetConfig(dataset_name=dataset_name,
                                   label_col='family',
                                   target_size=target_size,
                                   channels=3,
                                   low_class_count_thresh=3,
                                   data_splits={'val_size':0.2,'test_size':0.2},
                                   tfrecord_root_dir=tfrecord_root_dir,
                                   num_shards=10)

    train_config = TrainConfig(model_name=model_name,
                     batch_size=batch_size,
                     frozen_layers=(0,-4),
                     base_learning_rate=base_learning_rate,
                     buffer_size=1000,
                     num_epochs=num_epochs,
                     preprocessing=preprocessing,
                     augment_images=augment_images,
                     seed=3)

    experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                         train_config=train_config)

    ############################################
    #TODO: Move config definitions outside main() for:
    #    1. simplifying overall logic in main & segregating configuration to section marked by if __name__=='__main__'
    #    2. Moving towards defining most or all run parameters in separate config files
    ############################################
    
    
    trainer = BaseTrainer(experiment_config=experiment_config)


    train_data = trainer.get_data_loader(subset='train')
    val_data = trainer.get_data_loader(subset='val')
    test_data = trainer.get_data_loader(subset='test')
    
#     AUTOTUNE = tf.data.experimental.AUTOTUNE
#     train_data = tfds.load("mnist", split='train').shuffle(1000).batch(batch_size).repeat().prefetch(AUTOTUNE)
    
    model_params = trainer.get_model_params('train')
    fit_params = trainer.get_fit_params()
    callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'weights_best.h5'), logs_dir=experiment_dir, restore_best_weights=False)

    model = build_model(**model_params)  #name='shallow', num_classes=10000, frozen_layers=(0,-4), input_shape=(224,224,3), base_learning_rate=0.0001)


    history = model.fit(train_data,
                 steps_per_epoch = fit_params['steps_per_epoch'],
                 epochs=fit_params['epochs'],
                 validation_data=val_data,
                 validation_steps=fit_params['validation_steps'],
                 callbacks=callbacks
                 )
    return history
    
    
if __name__=='__main__':    
    '''
    Example:
    python /home/jacob/pyleaves/pyleaves/train/example_train.py -d PNAS -m resnet_50_v2 -gpu 3 -bsz 64
    
    
    python example_train.py -d PNAS -m resnet_50_v2 -gpu 3 -bsz 64
    
    Possible models:
    [
    'shallow',
    'vgg16',
    'xception',
    'resnet_50_v2',
    'resnet_101_v2'
    ]
    
    '''
   
    import argparse
    import numpy as np
    import os

    import tensorflow as tf
    tf.enable_eager_execution()
    
    import mlflow
    import mlflow.tensorflow
    mlflow.set_tracking_uri(r'/media/data/jacob/Fossil_Project/mlflow')
    
    mlflow.tensorflow.autolog()    
    
    
    print(mlflow.tracking.get_tracking_uri())
    print(mlflow.get_artifact_uri())
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', default='PNAS', type=str, help='Name of dataset of images to use for creating TFRecords')
    parser.add_argument('-m', '--model_name', default='vgg16', type=str, help='Name of model to train')
    parser.add_argument('-gpu', '--gpu_id', default=0, type=int, help='integer number of gpu to train on')
    
    parser.add_argument('-tfrec', '--tfrecord_dir', default=r'/media/data/jacob/Fossil_Project/tfrecord_data', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    parser.add_argument('-bsz', '--batch_size', default=64, type=int, help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default=0.001, type=float, help='Starting learning rate')
    parser.add_argument('-epochs', '--num_epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-f',default='')
    args = parser.parse_args()
    
    from pyleaves.utils import ensure_dir_exists, set_visible_gpus   
    set_visible_gpus([args.gpu_id])
    
    from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history

    ####
    from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, one_hot_encode_labels #, one_hot_decode_labels
    from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
    from pyleaves.leavesdb.db_query import get_label_encodings as _get_label_encodings, load_from_db
    from pyleaves.leavesdb.tf_utils.tf_utils import reset_keras_session, train_val_test_split as _train_val_test_split
    from pyleaves.models.keras_models import build_model
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig
    from pyleaves.train.base_train import BaseTrainer
    

    if args.model_name in ['vgg16', 'resnet_50_v2','resnet_101_v2']:
        target_size=(224,224)
    elif args.model_name=='xception':
        target_size=(299,299)
    else:
        target_size=(224,224) 
    

    learning_rates = [0.001,1e-4,1e-5,1e-6]
    histories = []
    for lr in learning_rates:
        
        experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',args.model_name,args.dataset_name,f'lr-{lr}')
        
        args.base_learning_rate=lr
        args.target_size=target_size
        mlflow.log_params(args.__dict__)        
        
        print('STARTING LEARNING RATE =',lr)
        history = main(args.dataset_name,
             args.model_name,
             experiment_dir,
             [args.gpu_id],
             args.tfrecord_dir,
             batch_size=args.batch_size,
             target_size=target_size,
             base_learning_rate=lr,
             num_epochs=args.num_epochs,
             preprocessing='imagenet',
             augment_images=True)
#              args.base_learning_rate,
        
    
        histories.append((lr,history))
    #     mlflow_log_best_history(history)

        mlflow_log_history(history)
    
#         tf.reset_default_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#####################################################################################################################    
    
    
    
    
    
    
    
    
    
    
# class Experiment:

#     def __init__(self,
#                  model_name='shallow',
#                  dataset_name='PNAS',
#                  experiment_name='PNAS_shallow',
#                  data_root_dir=r'/media/data/jacob',
#                  experiment_root_dir=r'/media/data/jacob/experiments',
#                  input_shape=(224,224,3),
#                  low_class_count_thresh=0,
#                  frozen_layers=(0,-4),
#                  base_learning_rate=0.001,
#                  batch_size=64,
#                  val_size=0.3,
#                  test_size=0.3,
#                  seed=17):
#         '''
#         Arguments:
#             model_name, str : model to build from definitions stored in pyleaves.model.keras_models.py
#             dataset_name, str : dataset to load from TFRecords generated by pyleaves.leavesdb.tf_utils.create_tfrecords.py
#             experiment_name, str : Chosen unique label for storing experiment weights, logs and results. Should contain dataset_model and any relevant metadata
#             data_root_dir, str : Absolute root directory containing subdirs for each dataset's tfrecords
#                    e.g. data_root_dir = /media/data/jacob/
#                                                          |Fossil/
#                                                                 |train/
#                                                                       |train-00000-of-00010.tfrecord
#                                                                       |train-00001-of-00010.tfrecord
#                                                                       ...
#                                                                       |train-00009-of-00010.tfrecord
#                                                                       |train-00010-of-00010.tfrecord
#                                                                 |val/
#                                                                     |...
#                                                                 |test/
#                                                                      |...
#                                                          |PNAS/
#             experiment_root_dir, str : root dir in which to store the experiment's results directory named after experiment_name
#             input_shape, tuple : input shape for images with dimension order (h, w, c)
#             low_class_count_thresh, int :
#             frozen_layers, tuple :
#             base_learning_rate, float :
#             batch_size, int :
#             val_size, float :
#             test_size, float :
#             seed, int :


#         '''

#         self.model_name = model_name
#         self.dataset_name = dataset_name
#         self.experiment_name = experiment_name
#         self.root_dir = os.path.join(data_root_dir, dataset_name)
# #         self.output_dir = os.path.join(output_dir,model_name)
#         self.experiment_root_dir = experiment_root_dir

#         self.input_shape = input_shape
#         self.low_class_count_thresh = low_class_count_thresh
#         self.frozen_layers = frozen_layers
#         self.base_learning_rate = base_learning_rate

#         self.batch_size = batch_size
#         self.seed = seed
#         self.val_size=val_size
#         self.test_size=test_size
#         self.verbose=True

        
#         self.initialize_experiment_paths()
#         self.load_metadata_from_db()
#         self.format_loaded_metadata()
#         self.data_subsets = self.train_val_test_split()

#         self.label_encodings = self.get_label_encodings()
        
#         self.dataset_builder = DatasetBuilder(root_dir=self.root_dir,
#                                               num_classes=self.num_classes,
#                                               batch_size=self.batch_size,
#                                               seed=self.seed)
#         self.tfrecord_splits = self.dataset_builder.collect_subsets(self.root_dir)


#         self.callbacks = self.get_callbacks()

#     def initialize_experiment_paths(self):
#         exp_dir = os.path.join(self.experiment_root_dir,self.experiment_name)
#         self.experiment_paths = {
#             'experiment_dir':exp_dir,
#             'weights_filepath':os.path.join(exp_dir,self.experiment_name+'_model-weights.hs'),
#             'logs_dir':os.path.join(exp_dir,self.experiment_name+'-logs_dir')
#         }
        
        
        
#     def load_metadata_from_db(self):
#         self.metadata, self.db = load_from_db(dataset_name=self.dataset_name)
#         return self.metadata
#     def format_loaded_metadata(self):
#         metadata=self.metadata
#         low_class_count_thresh=self.low_class_count_thresh
#         verbose=self.verbose

#         data_df = encode_labels(metadata)
#         data_df = filter_low_count_labels(data_df, threshold=low_class_count_thresh, verbose = verbose)
#         data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)
#         paths = data_df['path'].values.reshape((-1,1))
#         labels = data_df['label'].values
#         self.x = paths
#         self.y = labels
#         return self.x, self.y

#     def train_val_test_split(self):
#         x=self.x
#         y=self.y
#         test_size=self.test_size
#         val_size=self.val_size
#         verbose=self.verbose

#         self.data_subsets, self.metadata_subsets = _train_val_test_split(x, y, test_size=test_size, val_size=val_size, verbose=verbose)
#         return self.data_subsets

#     def get_label_encodings(self):
#         dataset=self.dataset_name
#         low_count_thresh=self.low_class_count_thresh

#         self.label_encodings = _get_label_encodings(dataset=dataset, low_count_thresh=low_count_thresh)
#         self.num_classes = len(self.label_encodings)
#         return self.label_encodings


#     def build_model(self):
#         model_name=self.model_name
#         num_classes=self.num_classes
#         frozen_layers=self.frozen_layers
#         input_shape=self.input_shape
#         base_learning_rate=self.base_learning_rate

#         return build_model(name=model_name,
#                            num_classes=num_classes,
#                            frozen_layers=frozen_layers,
#                            input_shape=input_shape,
#                            base_learning_rate=base_learning_rate)

#     def __setattr__(self, name, val):
#         self.__dict__[name] = val

#     def set_model_params(self,
#                          model_name=None,
#                          num_classes=None,
#                          frozen_layers=None,
#                          input_shape=None,
#                          base_learning_rate=None):

#         if model_name: self.model_name=model_name
#         if num_classes: self.num_classes=num_classes
#         if frozen_layers: self.frozen_layers=frozen_layers
#         if input_shape: self.input_shape=input_shape
#         if base_learning_rate: self.base_learning_rate=base_learning_rate

#     def get_callbacks(self):
#         return _get_callbacks(weights_best=self.experiment_paths['weights_filepath'],
#                               logs_dir=self.experiment_paths['logs_dir'],
#                               restore_best_weights=False)



#     def get_dataset(self,subset='train', batch_size=32, num_classes=None):
#         if num_classes is None:
#             num_classes=self.num_classes
#         return self.dataset_builder.get_dataset(subset=subset, batch_size=batch_size, num_classes=num_classes)
    