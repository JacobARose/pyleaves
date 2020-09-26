#!/usr/bin/env python
# coding: utf-8
"""

python ~/projects/pyleaves/pyleaves/pipelines/baseline_testing_pipeline.py 





    Raises:
        e: [description]

    Returns:
        [type]: [description]
"""





# # Programmatically transferring images into class-specific hierarchical directories
# 
# ### Transforming a 2-level hierarchy into single level in preparation for using an ImageDataGenerator from tf.keras
# 
# 
# 
# We have a dataset of images from 3 different collections (i.e. original publishing source) with overlapping subsets of class labels in the form of biological family (e.g. Theaceae, Rosaceae).
# 
# ## source structure:
# ```
# ./wolfe/
#         Acanthaceae/
#             image_0.jpg
#             image_1.jpg
#             ...
#         Achariaceae/
#         ...
#         Winteraceae/
#         Zygophyllaceae/
# ./axelrod/
#         Acanthaceae/
#             ...
#         ...
# ./klucking/
#         ...
# ```
# 
# 
# ## target structure:
# 
# ```
# Acanthaceae/
#     image_0.jpg
#     image_1.jpg
#     ...
# Achariaceae/
# ...
# Winteraceae/
# Zygophyllaceae/
# 
# 
# 
# ```
# 
# 



import os
import shutil
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
from pathlib import Path

def get_collection_files(collection_dir, position=0, leave=False):
    collection_files = {}
    for family in tqdm(os.listdir(collection_dir), desc=f'collection {Path(collection_dir).name}', position=position, leave=leave):
        family_dir = os.path.join(collection_dir, family)
        fam_files = [os.path.join(family_dir, file) for file in os.listdir(family_dir)]
        if family not in collection_files.keys():
            collection_files[family] = []
        collection_files[family].extend(fam_files)
        
    return collection_files
    
def gather_collections(collection_dirs, leave=False):
    all_files = {}
    for collection_dir in tqdm(collection_dirs, desc=f'PNAS collections', position=0, leave=leave):
        collection_files = get_collection_files(collection_dir, position=1, leave=leave)
        for family, fam_files in collection_files.items():
            if family not in all_files.keys():
                all_files[family] = []
            all_files[family].extend(fam_files)
    return all_files


from typing import List
def create_full_dataset_copy(collection_dirs: List[str]):
    all_files = gather_collections(collection_dirs, leave=True)

    leave=True
    for family, family_files in  tqdm(all_files.items(), desc='Family directories', position=0, leave=leave):
        family_dir = os.path.join(output_root_dir, family)
        os.makedirs(family_dir)
        for file in tqdm(family_files, desc = f'{family} files', position=1, leave=False):
            shutil.copy(file, family_dir)

PNAS_dir = '/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/PNAS_2020-06'
output_root_dir = '/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/PNAS_2020-06/PNAS_family'
collections = ['wolfe','axelrod','klucking']
collection_dirs = [os.path.join(PNAS_dir, collection) for collection in collections]
collection_dirs


# 
# ################################################################################################################################################
# ################################################################################################################################################
# # NEW SECTION
# ################################################################################################################################################
# ################################################################################################################################################
# 
# 
# 
# ## Proposed procedure for experiment logging:
# (last edited 9/26/2020)
# 1. Debug code until a baseline end-to-end example is working.
# 
# 2. Upload notebook checkpoint to neptune
# 
# 3. Perform series of variations/experiments by slightly altering the notebook or its inputs, upload metrics to neptune either all under the same experiment name, or a different one for each
# 
# 4. Once an optimal set of parameters/hyperparameters is reached or new conclusions made, restore the notebook back to the state that resulted in the optimal run.
# 
# 5. Create final checkpoint for notebook with name and detailed summary of conclusions
# 
# 
# ### Extra things to try later:
# 
# 1. After 5., export the notebook and a conda environment file or docker image to permanent location, ideally along with a writeup or even paper notes or blogpost.
# 


from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neptunecontrib.api.table import log_table
#hide
# Image plotting utils
def show_batch(image_batch, label_batch, image_data_gen=True, class_names=None):
    fig = plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        
        if image_data_gen:
            plt.title(class_names[label_batch[n].argmax()])
        else:
            plt.title(label_batch[n])
        
        plt.axis('off')
    return fig


def summarize_sample(x, y):
    y_int=y
    y_encoding = 'sparse int'
    if isinstance(y, np.ndarray):
        y_int = np.argmax(y, axis=-1)
        if y.ndim>=1 and y.shape[-1] > 1:
            y_encoding = 'one hot'
    print(f'y = {y_int} [{y_encoding} encoded]')
    print(f'y.dtype = {y.dtype}, x.dtype = {x.dtype}\n')
    print(f'y.shape = {y.shape},\ny.min() = {y.min():.3f} | y.max() = {y.max():.3f},\ny.mean() = {y.mean():.3f} | y.std() = {y.std():.3f}\n')
    print(f'x.shape = {x.shape},\nx.min() = {x.min():.3f} | x.max() = {x.max():.3f},\nx.mean() = {x.mean():.3f} | x.std() = {x.std():.3f}')

    plt.imshow(x)

from pyleaves.utils import set_tf_config
gpu = set_tf_config(gpu_num=None, num_gpus=1, wait=0)

import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()
from pyleaves.utils.pipeline_utils import build_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from pprint import pprint
from box import Box
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
import hydra


neptune_project_name = 'jacobarose/jupyter-testing-ground'
neptune_experiment_name = 'baseline-PNAS_family'




# params = Box({
#               'image_dir': '/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/PNAS_2020-06/PNAS_family',
#               'log_dir': '/media/data_cifs_lrs/projects/prj_fossils/users/jacob/tensorboard_log_dir',          
#               'validation_split': 0.1,
#               'target_size':(299,299),
#               'batch_size':32,
#               'num_epochs': 30,
#               'seed': 20,
#               'rescale': None, #1.0/255,
#               'preprocess_input': "tensorflow.keras.applications.resnet_v2.preprocess_input",
#               'color_mode': 'rgb',
#               'early_stopping': {'monitor':"val_loss",
#                                 'patience':10,
#                                 'min_delta':0.01,
#                                 'restore_best_weights':True}
# })

# model_config = Box({
#                     'model_name': "resnet_50_v2",
#                     'optimizer':"Adam",
#                     'num_classes':None, #params.num_classes,
#                     'weights': "imagenet",
#                     'frozen_layers':None, #(0,-4),
#                     'input_shape':None,#(*params.target_size,3),
#                     'lr':1e-5,
#                     'lr_momentum':None,#0.9,
#                     'regularization':{},#{"l2": 1e-4},
#                     'loss':'categorical_crossentropy',
#                     'METRICS':['f1','accuracy'],
#                     'head_layers': [256,128]
#                     })



#     featurewise_center=False, samplewise_center=False,
#     featurewise_std_normalization=False, samplewise_std_normalization=False,
#     zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
#     height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
#     channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
#     vertical_flip=False, rescale=1.0/255, preprocessing_function=preprocess_input,
#     data_format=None, validation_split=validation_split, dtype=np.uint8







@hydra.main(config_path='configs', config_name='baseline_testing_config')
def main(config):


    OmegaConf.set_struct(config, False)

    params = config

    
    data_augs = {k:v for k,v in OmegaConf.to_container(params.data_augs).items() if k != "preprocessing_function"}


    if params.data_augs.preprocessing_function == "tensorflow.keras.applications.resnet_v2.preprocess_input":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        # params.data_augs.pop('preprocessing_function') # = preprocess_input
        print("Using preprocessing function: tensorflow.keras.applications.resnet_v2.preprocess_input")
    else:
        preprocess_input = None
        print("Using no preprocess_input function")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_augs,
                                                              preprocessing_function = preprocess_input)
                                                            # rescale=params.rescale,
                                                            # preprocessing_function=preprocess_input,
                                                            # validation_split=params.validation_split)

    train_data = datagen.flow_from_directory(
        params.image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=None,
        class_mode='categorical', batch_size=params.batch_size, shuffle=True, seed=params.seed,
        subset='training', interpolation='nearest')


    val_data = datagen.flow_from_directory(
        params.image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=None,
        class_mode='categorical', batch_size=params.batch_size, shuffle=False, seed=params.seed,
        subset='validation', interpolation='nearest')



    params.num_samples_train = train_data.samples
    params.num_samples_val = val_data.samples
    params.num_classes = train_data.num_classes
    steps_per_epoch=params.num_samples_train//params.batch_size
    validation_steps=params.num_samples_val//params.batch_size
    # model_config = Box({
    #                     'model_name': "resnet_50_v2",
    #                     'optimizer':"Adam",
    #                     'num_classes':params.num_classes,
    #                     'weights': "imagenet",
    #                     'frozen_layers':None, #(0,-4),
    #                     'input_shape':(*params.target_size,3),
    #                     'lr':1e-5,
    #                     'lr_momentum':None,#0.9,
    #                     'regularization':{},#{"l2": 1e-4},
    #                     'loss':'categorical_crossentropy',
    #                     'METRICS':['f1','accuracy'],
    #                     'head_layers': [256,128]
    #                     })
    model_config = params
    model_config.num_classes = params.num_classes
    model_config.input_shape = (*params.target_size,3)

    neptune_params = {}
    for k,v in {**params.to_dict(), **model_config.to_dict()}.items():
        if type(v)==dict:
            neptune_params[k] = str(v)
        else:
            neptune_params[k] = v
            
    model = build_model(model_config)

    ################################################################################
    ################################################################################
    ################################################################################


    neptune.init(project_qualified_name=neptune_project_name)

    from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
    callbacks = [TensorBoard(log_dir=params.log_dir, profile_batch=2),
                NeptuneMonitor(),
                EarlyStopping(monitor=params.early_stopping.monitor,
                            patience=params.early_stopping.patience,
                            min_delta=params.early_stopping.min_delta, 
                            verbose=1, 
                            restore_best_weights=params.early_stopping.restore_best_weights)]


    with neptune.create_experiment(name=neptune_experiment_name, params=neptune_params) as experiment:
        model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

        print('[BEGINNING TRAINING]')
        try:
            history = model.fit(train_data,
                                epochs=params.num_epochs,
                                callbacks=callbacks,
                                validation_data=val_data,
                                validation_freq=1,
                                shuffle=True,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps=validation_steps,
                                verbose=1)

        except Exception as e:
            raise e
        from pyleaves.utils.pipeline_utils import evaluate_performance

        model.save(config.saved_model_path)
        print('[STAGE COMPLETED]')
        print(f'Saved trained model to {config.saved_model_path}')
        subset='val'
        y, y_hat, y_prob = evaluate(model, val_data, experiment=experiment, subset=subset)
        predictions = pd.DataFrame({'y':y,'y_pred':y_hat,'y_prob':y_prob})
        log_table(f'{subset}_labels_w_predictions',predictions, experiment=experiment)
        print('TEST RESULTS:')

    print(['[FINISHED TRAINING AND TESTING]'])

    return predictions


import pandas as pd
from sklearn.metrics import classification_report#, confusion_matrix

def evaluate(model, val_data, y=None, output_dict: bool=True, experiment=None, subset='val'):
    num_samples = val_data.samples
    batch_size = val_data.batch_size
    steps = int(np.ceil(num_samples/batch_size))

    y_true = val_data.labels
    y_prob = model.predict(val_data, steps=steps, verbose=1)
    
    classes = val_data.class_indices
    target_names = list(classes.keys())
    labels = [classes[text_label] for text_label in target_names]

    y_hat = y_prob.argmax(axis=1)
    print('y_hat.shape = ', y_hat.shape)
    y = y_true.argmax(axis=1)
    print('y.shape = ', y.shape)
    try:
        report = classification_report(y, y_hat, labels=labels, target_names=target_names, output_dict=output_dict)
        if type(report)==dict:
            report = pd.DataFrame(report)
        
        
        log_table(f'{subset}_classification_report', report, experiment=experiment)
    except Exception as e:
        import pdb; pdb.set_trace()
        print(e)

    return y, y_hat, y_prob





if __name__=='__main__':
    main()



    ## TODO Saturday: plot image grid with color coded labels with correct/incorrect status of a trained model's prediction on a random batch.

#     # get a random batch of images
# image_batch, label_batch = next(iter(validation_generator))
# # turn the original labels into human-readable text
# label_batch = [class_names[np.argmax(label_batch[i])] for i in range(batch_size)]
# # predict the images on the model
# predicted_class_names = model.predict(image_batch)
# predicted_ids = [np.argmax(predicted_class_names[i]) for i in range(batch_size)]
# # turn the predicted vectors to human readable labels
# predicted_class_names = np.array([class_names[id] for id in predicted_ids])
# # some nice plotting
# plt.figure(figsize=(10,9))
# for n in range(30):
#     plt.subplot(6,5,n+1)
#     plt.subplots_adjust(hspace = 0.3)
#     plt.imshow(image_batch[n])
#     if predicted_class_names[n] == label_batch[n]:
#         color = "blue"
#         title = predicted_class_names[n].title()
#     else:
#         color = "red"
#         title = f"{predicted_class_names[n].title()}, correct:{label_batch[n]}"
#     plt.title(title, color=color)
#     plt.axis('off')
# _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
# plt.show()