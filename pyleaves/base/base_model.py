# @Author: Jacob A Rose
# @Date:   Wed, April 1st 2020, 5:52 pm
# @Email:  jacobrose@brown.edu
# @Filename: base_model.py


'''

This script is for defining a custom BaseModel class for building and managing tensorflow/keras models and metadata in coordination with an instance of a BaseTrainer or one of its subclasses.

Created by:
Jacob Rose
3/2/20 9:06 PM

'''
# import pdb;pdb.set_trace();print(__file__)

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
    #TODO REFACTOR, Functionality currently moved to config_v2.py (4/18/20)
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


def get_keras_preprocessing_function(model_name: str, input_format=tuple, x_col='path', y_col='label'):
    '''
    #TODO REFACTOR, Functionality currently moved to config_v2.py (4/18/20)
    if input_dict_format==True:
        Includes value unpacking in preprocess function to accomodate TFDS {'image':...,'label':...} format
    '''
    if model_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif model_name == 'xception':
        from tensorflow.keras.applications.xception import preprocess_input
    elif model_name in ['resnet_50_v2','resnet_101_v2']:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    else:
        preprocess_input = lambda x: x

    if input_format=='dict':
        def preprocess_func(input_example):
            x = input_example[x_col]
            y = input_example[y_col]
            return preprocess_input(x), y
        _temp = {x_col:tf.zeros([4, 32, 32, 3]), y_col:tf.zeros(())}
        preprocess_func(_temp)

    elif input_format=='tuple':
        def preprocess_func(x, y):
            return preprocess_input(x), y
        _temp = ( tf.zeros([4, 32, 32, 3]), tf.zeros(()) )
        preprocess_func(*_temp)
    else:
        print('''input_format must be either dict or tuple, corresponding to data organized as:
              tuple: (x, y)
              or
              dict: {'image':x, 'label':y}
              ''')
        return None

    return preprocess_func

# import pdb;pdb.set_trace();print(__file__)



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
        if 'lr' in model_config:
            self.lr = model_config.lr
        else:
            self.lr = model_config.base_learning_rate

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

    def build_head(self, base):
        '''
        Implement this method in subclasses.
        '''
        return None

    def build_model(self, regularization=True):
        # import pdb; pdb.set_trace()

        # from tensorflow.keras import backend as K
        # K.clear_session()
        # K.reset_default_graph()
        # K.reset_eager_session

        base = self.build_base()
        model = self.build_head(base)

        if regularization:
            model = self.add_regularization(model)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr),
                      loss='categorical_crossentropy',
                      metrics=METRICS)
        self.model = model
        return model

    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     x, y = data
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         print(y_true.shape, y_pred.shape)
    #         loss = self.compiled_loss(y, y_pred,
    #                             regularization_losses=self.losses)
    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     # Update metrics (includes the metric that tracks the loss)
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}




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
        # return self.model

    def save(self, filepath = './saved_model', model=None):
        '''
        Save full model architecture + weights
        '''
        if model is None:
            model=self.model
        tf.keras.models.save_model(model=model,
                                   filepath=filepath,
                                   overwrite=True,
                                   include_optimizer=True,
                                   save_format='tf')
        print(f'Saved model {self.name} at path: {filepath}')

    def load(self, filepath = './saved_model'):
        '''
        Load full model architecture + weights
        '''
        self.model = tf.keras.models.load_model(filepath=filepath,
                                            compile=True)

        print(f'Loaded model {self.name} from path: {filepath}')
        return self.model

    def add_regularization(self, model):
        """
        Takes an existing model and adds either l1 or l2 regularization to every layer.

        Parameters
        ----------
        model : type
            a Functional or Sequential keras model

        """

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
        if 'checkpoint_filepath' in self.config:
            assert validate_filepath(self.config['checkpoint_filepath'],file_type='json')
            self.checkpoint_filepath = self.config['checkpoint_filepath']
        else:
            self.checkpoint_filepath = join(self.model_dir,f'{self.name}-checkpoint.h5')

        self.config['weights_filepath'] = self.weights_filepath
        self.config['config_filepath'] = self.config_filepath

        self.base_model_filepath = os.path.join(self.model_dir, self.name+'-saved_base_model')

# TODO: Implement export and import() methods for save_format='tf' rather than 'h5'
# import pdb;pdb.set_trace();print(__file__)


class Model(BaseModel):
    '''
    Inherits from BaseClass that implements basic model load/save methods for subclasses. Model building to be delegated to each individual subclass.
    '''

    @classmethod
    def add_regularization(cls, model, l1: float=None, l2: float=None):
        
        if l1 is not None:
            regularizer = tf.keras.regularizers.l1(l1)
        elif l2 is not None:
            regularizer = tf.keras.regularizers.l2(l2)
        else:
            return model

        model = add_regularization(model, regularizer)

        return model