'''
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
from pyleaves.datasets.base_dataset import calculate_class_weights
from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history
from pyleaves.models.resnet import ResNet, ResNetGrayScale
from pyleaves.models.vgg16 import VGG16, VGG16GrayScale
from stuf import stuf




class ModelBuilder:

    def __init__(self, config):
        """

        Parameters
        ----------
        config : stuf.stuf
            Expects user to input the config set of parameters defined in pyleaves.config.config_v2,
            which contains the necessary parameters to build the model in addition to more parameters for managing it.

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        self.config = config
        self.params = stuf({'model':{}})

    def get_model_params(self, num_classes=None):
        config = self.config.model_config
        if num_classes is None:
            num_classes = self.config.num_classes

        model_params = stuf(
                               name=config.name,
                               num_classes=num_classes,
                               frozen_layers=config.frozen_layers,
                               input_shape=(config.input_shape),
                               base_learning_rate=config.base_learning_rate,
                               regularization=config.regularization,
                               model_dir=config.model_dir
                           )
        self.params['model'] = model_params
        return model_params

    @property
    def model(self):
        return self.model_factory.model

    def build_model(self, num_classes=None):
        """Short summary.

        Parameters
        ----------
        num_classes : int, default=None
            Number of classes is used to determine the size of the last model layer.

        Returns
        -------
        tensorflow.keras.models.Model
            Built and compiled tensorflow model.

        Examples
        -------
        >>>

        """

        model_params = self.get_model_params(num_classes=num_classes)
        self.model_name = model_params.name
        if self.model_name == 'vgg16':
            self.model_factory = VGG16GrayScale(model_params)
        elif self.model_name.startswith('resnet'):
            self.model_factory = ResNet(model_params)
        self.model_factory.build_model()
        self.metrics_names = self.model.metrics_names

        return self.model






class BaseTrainer:

    def __init__(self, config, model_builder, data_manager, logger, callbacks=None, sess=None):
        self.config = config
        self.data_manager = data_manager
        self.logger = logger
        self.callbacks = callbacks
        self.sess = sess
        self.model_builder = model_builder or ModelBuilder(config)
        # with sess.graph.as_default():
        self.model_builder.build_model()

        # self.initialize_variables()

        self.params = stuf({
                            'model':self.model_builder.params['model'],
                            'fit_params':stuf({}),
                            'test_params':stuf({})
        })

    def initialize_variables(self):
        if not tf.executing_eagerly():
            self.sess.run(tf.global_variables_initializer())

    @property
    def class_weights(self):
        '''
        Loads the class weights based on class distribution in train set
        '''
        return load_csv_data(self.config.data_config.class_weights_filepath)

    @property
    def model(self):
        return self.model_builder.model

    @property
    def metrics_names(self):
        return self.model_builder.metrics_names

    @property
    def metadata(self):
        _meta = stuf({})
        for stage in self.config.stages:
            _meta[stage] = {}
            for group in self.config.data_config[stage]['file_groups']:
                num_samples = self.data_manager.get_num_samples_by_file_group(file_group=group, dataset_stage=stage)#'dataset_A')
                _meta[stage][group] = {'num_samples':num_samples}
        return _meta

    def get_fit_params(self, dataset_stage='dataset_A'):
        cfg = self.config.model_config
        params = {'steps_per_epoch' : self.metadata[dataset_stage]['train']['num_samples']//cfg.batch_size,
                  'validation_steps' : self.metadata[dataset_stage]['val']['num_samples']//cfg.batch_size,
                  'epochs' : cfg.num_epochs
                 }
        self.params['fit_params'].update(stuf(
                                            {dataset_stage:params}
                                            ))
        return params

    def get_test_params(self, dataset_stage='dataset_A'):
        cfg = self.config.model_config
        params = {'steps' : self.metadata[dataset_stage]['test']['num_samples']//cfg.batch_size
                 }
        self.params['test_params'].update(stuf(
                                            {dataset_stage:params}
                                            ))
        return params

    def train(self, train_data=None, val_data=None, dataset_stage='dataset_A', class_weights=None):

        train_data = train_data or self.data_manager.get_data_loader('train', dataset_stage)
        val_data = val_data or self.data_manager.get_data_loader('val', dataset_stage)
        params = self.get_fit_params(dataset_stage)

        return self.fit(train_data,
                         steps_per_epoch = params['steps_per_epoch'],
                         epochs=params['epochs'],
                         validation_data=val_data,
                         class_weight=class_weights,
                         validation_steps=params['validation_steps'],
                         callbacks=self.callbacks,
                         history_name=dataset_stage)

    def test(self, test_data=None, dataset_stage='dataset_A'):

        test_data = test_data or self.data_manager.get_data_loader('test', dataset_stage)
        params = self.get_test_params(dataset_stage)
        return self.evaluate(test_data,
                             steps=params['steps'],
                             log_name=dataset_stage+'_test',
                             callbacks=self.callbacks)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            history_name=''):

    #TODO CHECK IS USE_MULTIPROCESSING STILL WORKS/HELPS IF USING TF.DATA WITH AUTOTUNE

        history = self.model.fit(x=x,
                                 y=y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 validation_data=validation_data,
                                 shuffle=shuffle,
                                 class_weight=class_weight,
                                 sample_weight=sample_weight,
                                 initial_epoch=initial_epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps,
                                 validation_freq=validation_freq,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)

        # self.logger.log_metrics(history=history.history, log_name=history_name)
        self.logger.log_history(history, history_name=history_name)
        # mlflow_log_history(history, history_name=history_name)

        return history

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 log_name=''):

        results = self.model.evaluate(x=x,
                                      y=y,
                                      batch_size=batch_size,
                                      verbose=verbose,
                                      sample_weight=sample_weight,
                                      steps=steps,
                                      callbacks=callbacks,
                                      max_queue_size=max_queue_size,
                                      workers=workers,
                                      use_multiprocessing=use_multiprocessing)

        history = {k:v for k, v in zip(self.metrics_names,results)}

        self.logger.log_metrics(history=history, log_name=log_name)
        # for k, v in history.items():
        #     mlflow.log_metric(log_name+k, v)

        return history



    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):

        predictions = self.model.predict(x=x,
                batch_size=batch_size,
                verbose=verbose,
                steps=steps,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing)

        return predictions

    def save_model(self, filepath='saved_model'):
        self.model_builder.model_factory.save(filepath = os.path.join(self.config.model_config['model_dir'],filepath), model=self.model)

    def load_model(self, filepath='saved_model'):
        return self.model_builder.model_factory.load(filepath = os.path.join(self.config.model_config['model_dir'],filepath))



# if __name__ == '__main__':

    # dataset_config = DatasetConfig(dataset_name='PNAS',
    #                                label_col='family',
    #                                target_size=(224,224),
    #                                channels=3,
    #                                low_class_count_thresh=3,
    #                                data_splits={'val_size':0.2,'test_size':0.2},
    #                                tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
    #                                num_shards=10)
    #
    # train_config = TrainConfig(model_name='vgg16',
    #                  batch_size=64,
    #                  frozen_layers=(0,-4),
    #                  base_learning_rate=1e-4,
    #                  buffer_size=1000,
    #                  num_epochs=100,
    #                  preprocessing='imagenet',
    #                  augment_images=True,
    #                  seed=3)
    #
    #
    #
    # experiment_config = ExperimentConfig(dataset_config=dataset_config,
    #                                      train_config=train_config)
    #
    # trainer = BaseTrainer(experiment_config=experiment_config)
    #
    # ##LOAD AND PLOT 1 BATCH OF IMAGES AND LABELS FROM FOSSIL DATASET
    # experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',trainer.config.model_name,trainer.config.dataset_name)
    #
    # train_data = trainer.get_data_loader(subset='train')
    # val_data = trainer.get_data_loader(subset='val')
    # test_data = trainer.get_data_loader(subset='test')
    #
    #
    # for imgs, labels in train_data.take(1):
    #     labels = [trainer.label_encodings[np.argmax(label)] for label in labels.numpy()]
    #     imgs = (imgs.numpy()+1)/2
    #     plot_image_grid(imgs, labels, 4, 8)
