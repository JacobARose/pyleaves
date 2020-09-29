#!/usr/bin/env python
# coding: utf-8
"""

python ~/projects/pyleaves/pyleaves/pipelines/baseline_testing_pipeline_tf_data.py 


python ~/projects/pyleaves/pyleaves/pipelines/baseline_testing_pipeline_tf_data.py target_size=[299,299] batch_size=32 num_epochs=60 'frozen_layers=[0,-4]' num_parallel_calls=4



python ~/projects/pyleaves/pyleaves/pipelines/baseline_testing_pipeline_tf_data.py dataset@dataset=Leaves_family_4 target_size=[299,299] batch_size=32 num_epochs=80 'frozen_layers=[0,-4]' num_parallel_calls=4


python ~/projects/pyleaves/pyleaves/pipelines/baseline_testing_pipeline_tf_data.py dataset@dataset=Fossil_family_4 target_size=[299,299] batch_size=32 num_epochs=80 'frozen_layers=[0,-4]' num_parallel_calls=4



    Raises:
        e: [description]

    Returns:
        [type]: [description]
"""

# 
# ################################################################################################################################################
# ################################################################################################################################################
# # NEW SECTION
# ################################################################################################################################################
# ################################################################################################################################################
# 
# 
from pyleaves.utils.pipeline_utils import evaluate_performance
import os
import shutil
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
from pathlib import Path


from typing import Union, Dict, List
import pandas as pd
from boltons.dictutils import OneToOne
# from pyleaves.utils import set_tf_config
# gpu = set_tf_config(gpu_num=None, num_gpus=1, wait=0)

# import tensorflow as tf
# from tensorflow.keras import backend as K
# K.clear_session()

# from pyleaves.utils.pipeline_utils import build_model
# from tensorflow.keras.applications.resnet_v2 import preprocess_input
from pprint import pprint
from box import Box
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

import neptune
# import neptune_tensorboard as neptune_tb

from neptunecontrib.monitoring.keras import NeptuneMonitor



def load_data_from_tensor_slices(data: pd.DataFrame, cache_paths: Union[bool,str]=True, training=False, seed=None, x_col='path', y_col='label', dtype=None):
    import tensorflow as tf
    dtype = dtype or tf.uint8
    num_samples = data.shape[0]

    def load_img(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    x_data = tf.data.Dataset.from_tensor_slices(data[x_col].values.tolist())
    y_data = tf.data.Dataset.from_tensor_slices(data[y_col].values.tolist())
    data = tf.data.Dataset.zip((x_data, y_data))

    data = data.take(num_samples)
    
    # TODO TEST performance and randomness of the order of shuffle and cache when shuffling full dataset each iteration, but only filepaths and not full images.
    if training:
        data = data.shuffle(num_samples,seed=seed, reshuffle_each_iteration=True)
    if cache_paths:
        if isinstance(cache_paths, str):
            data = data.cache(cache_paths)
        else:
            data = data.cache()

    data = data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=dtype),y), num_parallel_calls=-1)
    return data



def decode_int2str(labels: List[int], class_decoder: Dict[int, str]) -> List[str]:
    return [class_decoder[label] for label in labels]

def encode_str2int(labels: List[str], class_encoder: Dict[str, int]) -> List[int]:
    return [class_encoder[label] for label in labels]


def img_data_gen_2_tf_data(data, 
                           training=False,
                           target_size=(256,256),
                           batch_size=16,
                           seed=None,
                           preprocess_input=None,
                           augmentations: Dict[str,float]=None,
                           num_parallel_calls=-1,
                           cache=False,
                           class_encodings: Dict[str,int]=None):
    from pyleaves.utils.pipeline_utils import flip, _cond_apply
    import tensorflow as tf
    augmentations = augmentations or {}
    num_samples = data.samples
    num_classes = data.num_classes
    class_encoder = OneToOne(data.class_indices)
    paths = data.filepaths
    labels = data.labels

    if class_encodings:
        #Encode according to a previously established str<->int mapping in class_encodings
        text_labels = decode_int2str(labels=labels, class_decoder=class_encoder.inv)
        labels = encode_str2int(labels=text_labels, class_encoder=class_encodings)

    prepped_data = pd.DataFrame.from_records([{'path':path, 'label':label} for path, label in zip(paths, labels)])
    tf_data = load_data_from_tensor_slices(data=prepped_data, training=training, seed=seed, x_col='path', y_col='label', dtype=tf.float32)
    
    if preprocess_input is not None:
        tf_data = tf_data.map(lambda x,y: (preprocess_input(x), y), num_parallel_calls=num_parallel_calls)
    
    from functools import partial
    target_size = tuple(target_size)
    resize = partial(tf.image.resize, size=target_size)
    print('target_size = ', target_size)
    tf_data = tf_data.map(lambda x,y: (resize(x), tf.one_hot(y, depth=num_classes)), num_parallel_calls=num_parallel_calls)
    

    tf_data = tf_data.repeat().batch(batch_size)
    
    if cache:
        tf_data = tf_data.cache()

    for aug in augmentations:
        if 'flip' in aug:
            dataset = dataset.map(lambda x, y: _cond_apply(x, y, flip, prob=aug['flip'], seed=seed), num_parallel_calls=num_parallel_calls)    

    tf_data = tf_data.prefetch(-1)
    return {'data':tf_data, 'data_iterator':data, 'encoder':class_encoder, 'num_samples':num_samples, 'num_classes':num_classes}



def load_Fossil_family_4_subset(params, subset='test', preprocess_input=None, class_encodings: Dict[str,int]=None):
    import tensorflow as tf

    if subset=='train':
        image_dir = "/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/data_splits/Fossil_family_4_2020-06/train"
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=params.validation_split)
        train_iter = datagen.flow_from_directory(
            image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=list(class_encodings.keys()),
            class_mode='categorical', batch_size=params.batch_size, shuffle=True, seed=params.seed,
            subset='training', interpolation='nearest')

        # with tf.device('/cpu:0'):
        data_info = img_data_gen_2_tf_data(train_iter,
                                            training=True,
                                            target_size=params.target_size,
                                            batch_size=params.batch_size,
                                            seed=params.seed,
                                            preprocess_input=preprocess_input,
                                            augmentations=params.augmentations,
                                            num_parallel_calls=params.num_parallel_calls,
                                            class_encodings=class_encodings)
        return data_info

    elif subset=='val':
        image_dir = "/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/data_splits/Fossil_family_4_2020-06/train"
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=params.validation_split)
        val_iter = datagen.flow_from_directory(
                            image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=list(class_encodings.keys()),
                            class_mode='categorical', batch_size=params.batch_size, shuffle=False, seed=params.seed,
                            subset='validation', interpolation='nearest')



        # with tf.device('/cpu:0'):
        data_info = img_data_gen_2_tf_data(val_iter,
                                            training=False,
                                            target_size=params.target_size,
                                            batch_size=params.batch_size,
                                            seed=params.seed,
                                            preprocess_input=preprocess_input,
                                            cache=True,
                                            class_encodings=class_encodings)
        return data_info

    elif subset=='test':
        image_dir = "/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/data_splits/Fossil_family_4_2020-06/test"
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()#rescale = data_augs['rescale'],
                                                                    # preprocessing_function = preprocess_input)
        test_iter = test_datagen.flow_from_directory(
            image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=list(class_encodings.keys()),
            class_mode='categorical', batch_size=params.batch_size, shuffle=False, seed=params.seed, interpolation='nearest')

        # with tf.device('/cpu:0'):
        data_info = img_data_gen_2_tf_data(test_iter,
                                                training=False,
                                                target_size=params.target_size,
                                                batch_size=params.batch_size,
                                                seed=params.seed,
                                                preprocess_input=preprocess_input,
                                                class_encodings=class_encodings)
        return data_info






    










from omegaconf import OmegaConf, ListConfig, DictConfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neptunecontrib.api.table import log_table
from neptunecontrib.monitoring.keras import NeptuneMonitor
#hide
# Image plotting utils
# def show_batch(image_batch, label_batch, image_data_gen=True, class_names=None):
#     fig = plt.figure(figsize=(10, 10))
#     for n in range(25):
#         ax = plt.subplot(5, 5, n+1)
#         plt.imshow(image_batch[n])
        
#         if image_data_gen:
#             plt.title(class_names[label_batch[n].argmax()])
#         else:
#             plt.title(label_batch[n])
        
#         plt.axis('off')
#     return fig


# Image plotting utils
def show_batch(image_batch, label_batch, title='', class_names=None):
    fig = plt.figure(figsize=(15, 15))
    if class_names is not None:
        label_batch = [class_names[l] for l in np.argmax(label_batch, axis=1)]
    elif label_batch.ndim==2:
        label_batch = np.argmax(label_batch, axis=1)
    
    img_shape = image_batch.shape
    img_max = np.max(image_batch)
    img_min = np.min(image_batch)
    scaled_image_batch = (image_batch-img_min)/(img_max-img_min)

    title = f'{title}|min={img_min:.2f}|max={img_max:.2f}|dtype={image_batch.dtype}|shape={img_shape}\n(Scaled to [0,1] for visualization)'
    plt.suptitle(title)

    num_batches = np.min([image_batch.shape[0], 25])

    for n in range(num_batches):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(scaled_image_batch[n])
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


def parse_params(params: DictConfig):

    if params.dataset_name == "Leaves_family_4":
        params.train_image_dir: "/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/data_splits/Leaves_family_4_2020-03/train"
        params.test_image_dir: "/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/data_splits/Leaves_family_4_2020-03/test"

    elif params.dataset_name == "PNAS_family_4":
        params.train_image_dir: "/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/PNAS_2020-06/PNAS_family_4/train"
        params.test_image_dir: "/media/data_cifs_lrs/projects/prj_fossils/data/processed_data/PNAS_2020-06/PNAS_family_4//test"


    params.regularization = params.regularization or {}
    params.lr = float(params.lr)
    params.data_augs.validation_split = float(params.data_augs.validation_split)
    try:
        params.data_augs.rescale = float(params.data_augs.rescale)
    except:
        params.data_augs.rescale = None
    
    data_augs = {k:v for k,v in OmegaConf.to_container(params.data_augs, resolve=True).items() if k != "preprocessing_function"}

    return params, data_augs








import hydra



@hydra.main(config_path='baseline_configs', config_name='baseline_testing_config')
def main(config):

    OmegaConf.set_struct(config, False)

    from pyleaves.utils import set_tf_config
    config.task = config.task or 1
    gpu = set_tf_config(gpu_num=None, num_gpus=1, wait=(config.task+1)*2)

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

    neptune_project_name = 'jacobarose/jupyter-testing-ground'
    neptune_experiment_name = f'baseline-{config.dataset_name}'

    params = config

    params, data_augs = parse_params(params=params)


    if params.data_augs.preprocessing_function == "tensorflow.keras.applications.resnet_v2.preprocess_input":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        print("Using preprocessing function: tensorflow.keras.applications.resnet_v2.preprocess_input")


    elif params.data_augs.preprocessing_function == "tf.keras.applications.mobilenet.preprocess_input":
        from tensorflow.keras.applications.mobilenet import preprocess_input

    elif params.data_augs.preprocessing_function == "tf.keras.applications.inception_v3.preprocess_input":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    else:
        preprocess_input = None
        print("Using no preprocess_input function")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_augs,
                                                              preprocessing_function = preprocess_input)

    train_iter = datagen.flow_from_directory(
        params.train_image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=None,
        class_mode='categorical', batch_size=params.batch_size, shuffle=True, seed=params.seed,
        subset='training', interpolation='nearest')


    val_iter = datagen.flow_from_directory(
        params.train_image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=None,
        class_mode='categorical', batch_size=params.batch_size, shuffle=False, seed=params.seed,
        subset='validation', interpolation='nearest')


    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = data_augs['rescale'],
                                                                preprocessing_function = preprocess_input)

    test_iter = test_datagen.flow_from_directory(
        params.test_image_dir, target_size=params.target_size, color_mode=params.color_mode, classes=None,
        class_mode='categorical', batch_size=params.batch_size, shuffle=False, seed=params.seed, interpolation='nearest')




    # with tf.device('/cpu:0'):
    train_data_info = img_data_gen_2_tf_data(train_iter,
                                            training=True,
                                            target_size=params.target_size,
                                            batch_size=params.batch_size,
                                            seed=params.seed,
                                            preprocess_input=preprocess_input,
                                            num_parallel_calls=params.num_parallel_calls)
    # with tf.device('/cpu:0'):
    val_data_info = img_data_gen_2_tf_data(val_iter,
                                        training=False,
                                        target_size=params.target_size,
                                        batch_size=params.batch_size,
                                        seed=params.seed,
                                        preprocess_input=preprocess_input)
    # with tf.device('/cpu:0'):
    test_data_info = img_data_gen_2_tf_data(test_iter,
                                            training=False,
                                            target_size=params.target_size,
                                            batch_size=params.batch_size,
                                            seed=params.seed,
                                            preprocess_input=preprocess_input)

    train_data = train_data_info['data']
    val_data = val_data_info['data']
    test_data = test_data_info['data']

    params.class_encodings = dict(train_data_info['encoder'])
    params.num_samples_train = train_iter.samples
    params.num_samples_val = val_iter.samples
    params.num_samples_test = test_iter.samples
    params.num_classes = train_iter.num_classes
    steps_per_epoch=int(np.ceil(params.num_samples_train/params.batch_size))
    validation_steps=int(np.ceil(params.num_samples_val/params.batch_size))
    test_steps=int(np.ceil(params.num_samples_test/params.batch_size))

    neptune_params = {}
    for k,v in OmegaConf.to_container(params, resolve=True).items():
        if type(v)==dict:
            neptune_params[k] = str(v)
        elif type(v)==ListConfig:
            neptune_params[k] = list(v)
        else:
            neptune_params[k] = v



    model_config = params
    model_config.num_classes = params.num_classes
    model_config.input_shape = (*params.target_size,3)

            
    model = build_model(model_config)

    ################################################################################
    ################################################################################
    ################################################################################


    neptune.init(project_qualified_name=neptune_project_name)

    from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
    callbacks = [TensorBoard(log_dir=params.log_dir, profile_batch=0),
                NeptuneMonitor(),
                EarlyStopping(monitor=params.early_stopping.monitor,
                            patience=params.early_stopping.patience,
                            min_delta=params.early_stopping.min_delta, 
                            verbose=1, 
                            restore_best_weights=params.early_stopping.restore_best_weights)]
    from pyleaves.utils import pipeline_utils
    upload_source_files=[__file__, pipeline_utils.__file__]

    # neptune_experiment_name = config.misc.experiment_name
    tags = OmegaConf.to_container(params.tags, resolve=True)


    with neptune.create_experiment(name=neptune_experiment_name,
                                   params=neptune_params,
                                   upload_source_files=upload_source_files) as experiment:
        model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

        class_names = train_data_info['encoder'].inv

        image_batch, label_batch = next(iter(train_data))
        fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='train', class_names=class_names)
        experiment.log_image('train_image_batch', fig)

        image_batch, label_batch = next(iter(val_data))
        fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='val', class_names=class_names)
        experiment.log_image('val_image_batch', fig)


        image_batch, label_batch = next(iter(test_data))
        fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='test', class_names=class_names)
        experiment.log_image('test_image_batch', fig)

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

        model.save(config.saved_model_path)
        print('[STAGE COMPLETED]')
        print(f'Saved trained model to {config.saved_model_path}')
        subset='test'
        # num_samples = data_iter.samples
        # batch_size = data_iter.batch_size
        # steps = int(np.ceil(num_samples/batch_size))
        y_true = test_iter.labels
        classes = test_iter.class_indices
        eval_iter = test_data.unbatch().take(len(y_true)).batch(params.batch_size)

        y, y_hat, y_prob = evaluate(model, eval_iter, y_true=y_true, classes=classes, steps=test_steps, experiment=experiment, subset=subset)

        print('y_prob.shape =', y_prob.shape)
        predictions = pd.DataFrame({'y':y,'y_pred':y_hat})
        log_table(f'{subset}_labels_w_predictions',predictions, experiment=experiment)

        y_prob_df = pd.DataFrame(y_prob, columns=list(classes.keys()))
        log_table(f'{subset}_probabilities',y_prob_df,experiment=experiment)


        if "zero_shot_test" in params:
            class_encodings = train_data_info['encoder']
            test_data_info = load_Fossil_family_4_subset(params, subset='test', preprocess_input=preprocess_input, class_encodings=class_encodings)
            test_iter = test_data_info['data_iterator']
            test_data = test_data_info['data']
            y_true = test_iter.labels
            classes = test_iter.class_indices
            eval_iter = test_data.unbatch().take(len(y_true)).batch(params.batch_size)

            steps = np.int(np.ceil(test_data_info['num_samples']/params.batch_size))
            subset='zero-shot-test_Fossil_family_4'
            y, y_hat, y_prob = evaluate(model, eval_iter, y_true=y_true, classes=classes, steps=steps, experiment=experiment, subset=subset)

            print('y_prob.shape =', y_prob.shape)
            predictions = pd.DataFrame({'y':y,'y_pred':y_hat})
            log_table(f'{subset}_labels_w_predictions',predictions, experiment=experiment)

            y_prob_df = pd.DataFrame(y_prob, columns=list(classes.keys()))
            log_table(f'{subset}_probabilities',y_prob_df,experiment=experiment)







    print(['[FINISHED TRAINING AND TESTING]'])

    return predictions


import pandas as pd
from sklearn.metrics import classification_report#, confusion_matrix
from neptunecontrib.monitoring.metrics import log_confusion_matrix


def evaluate(model, data_iter, y_true, steps: int, classes, output_dict: bool=True, experiment=None, subset='val'):
    # num_classes = data_iter.num_classes


    y_prob = model.predict(data_iter, steps=steps, verbose=1)
    
    target_names = list(classes.keys())
    labels = [classes[text_label] for text_label in target_names]

    y_hat = y_prob.argmax(axis=1)
    print('y_hat.shape = ', y_hat.shape)
    if y_true.ndim > 1:
        y_true = y_true.argmax(axis=1)
    print('y_true.shape = ', y_true.shape)
    try:
        report = classification_report(y_true, y_hat, labels=labels, target_names=target_names, output_dict=output_dict)
        if type(report)==dict:
            report = pd.DataFrame(report)
        log_table(f'{subset}_classification_report', report, experiment=experiment)
        log_confusion_matrix(y_true, y_hat)

    except Exception as e:
        import pdb; pdb.set_trace()
        print(e)


    # from pyleaves.utils.callback_utils import NeptuneVisualizationCallback
    # callbacks = [NeptuneMonitor()]
    # NeptuneVisualizationCallback(test_data, num_classes=num_classes, text_labels=target_names, steps=steps, subset_prefix=subset, experiment=experiment),
    test_results = model.evaluate(data_iter, steps=steps, verbose=1)

    print('TEST RESULTS:\n',test_results)

    print('Results:')
    for m, result in zip(model.metrics_names, test_results):
        print(f'{m}: {result}')
        experiment.log_metric(f'{subset}_{m}', result)

    return y_true, y_hat, y_prob





if __name__=='__main__':
    main()

from typing import Dict

def vis_saliency_maps(model, imgs, labels, classes: Dict[int,str], dataset_name=''):
    from vis.visualization import visualize_saliency
    from vis.utils import utils
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np

    # Find the index of the to be visualized layer above
    layer_index = utils.find_layer_idx(model, 'visualized_layer')

    # Swap softmax with linear
    model.layers[layer_index].activation = tf.keras.activations.linear
    model = utils.apply_modifications(model)  

    # Numbers to visualize
    indices_to_visualize = [ 0, 12, 38, 83, 112, 74, 190 ]

    # Visualize
    for index_to_visualize in indices_to_visualize:
    # Get input
        input_image = imgs[index_to_visualize]

        input_class = np.argmax(labels[index_to_visualize])
        input_class_name = classes[input_class]
    # Matplotlib preparations
        fig, axes = plt.subplots(1, 2)
        # Generate visualization
        visualization = visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)
        axes[0].imshow(input_image) 
        axes[0].set_title('Original image')
        axes[1].imshow(visualization)
        axes[1].set_title('Saliency map')
    fig.suptitle(f'{dataset_name} target = {input_class_name}')
    plt.show()













#     ## TODO Saturday: plot image grid with color coded labels with correct/incorrect status of a trained model's prediction on a random batch.

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



# # =============================================
# # Activation Maximization code
# # =============================================
# # from vis.visualization import visualize_activation
# # from vis.utils import utils
# import matplotlib.pyplot as plt

# # Find the index of the to be visualized layer above
# layer_index = utils.find_layer_idx(model, 'visualized_layer')

# # Swap softmax with linear
# model.layers[layer_index].activation = activations.linear
# model = utils.apply_modifications(model)  

# # Classes to visualize
# classes_to_visualize = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
# classes = {
#   0: 'airplane',
#   1: 'automobile',
#   2: 'bird',
#   3: 'cat',
#   4: 'deer',
#   5: 'dog',
#   6: 'frog',
#   7: 'horse',
#   8: 'ship',
#   9: 'truck'
# }

# # Visualize
# for number_to_visualize in classes_to_visualize:
#   visualization = visualize_activation(model, layer_index, filter_indices=number_to_visualize, input_range=(0., 1.))
#   plt.imshow(visualization)
#   plt.title(f'CIFAR10 target = {classes[number_to_visualize]}')
#   plt.show()