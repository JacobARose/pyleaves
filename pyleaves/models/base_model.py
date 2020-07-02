# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: base_model.py


'''

DEPRECATED (4/1/2020): All functionality moved to pyleaves/base/base_model.py

This script is for defining a custom BaseModel class for building and managing tensorflow/keras models and metadata in coordination with an instance of a BaseTrainer or one of its subclasses.

Created by:
Jacob Rose
3/2/20 9:06 PM

'''

import numpy as np
import os
join = os.path.join
import tempfile
import tensorflow as tf
from pyleaves.utils import ensure_dir_exists, validate_filepath
from pyleaves.train.metrics import METRICS
# from pyleaves.utils import set_visible_gpus
# set_visible_gpus([7])


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    model_json = model.to_json()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)

    return model

def get_model_default_param(config, param):
    model_name = config.model_name
    color_type = config.color_type
#     import pdb; pdb.set_trace()
    if param=='input_shape':
        target_size=(224,224)
        num_channels=3
        if model_name=='xception':
            target_size=(299,299)
        elif model_name=='vgg16' and color_type=='grayscale':
            num_channels=1
        return (*target_size,num_channels)




class BaseModel:
    '''
    Base Class that implements basic model load/save methods for subclasses. Model building to be delegated to each individual subclass.
    '''
    def __init__(self, model_config, name=''):

        self.name = name
        self.config = model_config

        self.num_classes = model_config.num_classes
        self.frozen_layers = model_config.frozen_layers
        self.input_shape = model_config.input_shape
        self.base_learning_rate = model_config.base_learning_rate
        self.regularization = model_config.regularization

        self.init_dirs()

#         self.model = self.build_model()


#         self.weights = self.model.get_weights()
#         json_config = self.model.to_json()
#         self.model.save_weights(weights_filepath)
#         with open(config_filepath,'w') as json_file:
#             json_file.write(json_config)

    def build_base(self):
        '''
        Implement this method in subclasses.
        '''
        return None

    def build_head(self):
        '''
        Implement this method in subclasses.
        '''
        return None

    def build_model(self):

        base = self.build_base()

        model = self.build_head(base)

        model = self.add_regularization(model)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.base_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=METRICS)
        self.model = model
        return model


    def save_weights(self, filepath = None):
        '''
        Save model weights
        '''
        if filepath is None:
            filepath = self.weights_filepath
        self.model.save_weights(filepath)

    def load_weights(self, filepath = None):
        '''
        Load model weights
        '''
        if filepath is None:
            filepath = self.weights_filepath
        self.model.load_weights(filepath)


    def save(self, filepath = './saved_model.h5'):
        '''
        Save full model architecture + weights
        '''
        tf.keras.models.save_model(model=self.model,
                                   filepath=filepath,
                                   overwrite=True,
                                   include_optimizer=True,
                                   save_format='h5')
        print(f'Saved model {self.name} at path: {filepath}')

    def load(self, filepath = './saved_model.h5'):
        '''
        Load full model architecture + weights
        '''
        model = tf.keras.models.load_model(filepath=filepath,
                                   compile=True)

        print(f'Loaded model {self.name} from path: {filepath}')

    def add_regularization(self, model):

        if self.regularization is not None:
            if 'l2' in self.regularization:
                regularizer = tf.keras.regularizers.l2(self.regularization['l2'])
            elif 'l1' in self.regularization:
                regularizer = tf.keras.regularizers.l1(self.regularization['l1'])

            model = add_regularization(model, regularizer)

        return model


    def init_dirs(self):

        self.model_dir = self.config.model_dir
        ensure_dir_exists(self.model_dir)
        if 'weights_filepath' in self.config:
            assert validate_filepath(self.config['weights_filepath'],file_type='h5')
            self.weights_filepath = self.config['weights_filepath']
        else:
            self.weights_filepath = join(self.model_dir,f'{self.name}-model_weights.h5')
        if 'config_filepath' in self.config:
            assert validate_filepath(self.config['config_filepath'],file_type='json')
            self.config_filepath = self.config['config_filepath']
        else:
            self.config_filepath = join(self.model_dir,f'{self.name}-model_config.json')

        self.config['weights_filepath'] = self.weights_filepath
        self.config['config_filepath'] = self.config_filepath


# TODO: Implement export and import() methods for save_format='tf' rather than 'h5'
