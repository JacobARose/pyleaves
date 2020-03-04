'''

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
# from pyleaves.utils import set_visible_gpus
# set_visible_gpus([7])



class BaseModel:
    
    def __init__(self, name=''):
        
        self.name = name
        self.model = self.build_model()
        

#         weights_filepath = join(filepath,'weights.h5')
#         config_filepath = join(filepath,'modelconfig.json')
        
#         self.weights = self.model.get_weights()
#         json_config = self.model.to_json()
#         self.model.save_weights(weights_filepath)
#         with open(config_filepath,'w') as json_file:
#             json_file.write(json_config)        
        
    def build_model(self):
        
        pass
    
    
    def save(self, filepath = './saved_model.h5'):
        
        tf.keras.models.save_model(model=self.model,
                                   filepath=filepath,
                                   overwrite=True,
                                   include_optimizer=True,
                                   save_format='h5')
        print('Saved model {self.name} at path: {filepath}')
        
    def load(self, filepath = './saved_model.h5'):
        
        model = tf.keras.models.load_model(filepath=filepath,
                                   compile=True)
        
        print('Loaded model {self.name} from path: {filepath}')
        
        
# TODO: Implement export and import() methods for save_format='tf' rather than 'h5'



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
















