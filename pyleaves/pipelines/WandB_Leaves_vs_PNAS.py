#!/usr/bin/env python
# coding: utf-8
"""

python ~/projects/pyleaves/pyleaves/pipelines/WandB_baseline_finetuning_pipeline.py 




python ~/projects/pyleaves/pyleaves/pipelines/WandB_Leaves_vs_PNAS.py \
                            'WandB_dataset_0@WandB_dataset_0=Leaves-PNAS' \
                            'pretrain.model_name="resnet_50_v2"' \
                            'pretrain.target_size=[768,768]' \
                            'pretrain.num_epochs=120' \
                            'pretrain.augmentations.flip=1.0' \
                            'pretrain.augmentations.rotate=1.0' \
                            'pretrain.augmentations.sbc=0.0' \
                            'pretrain.lr=1e-4' \
                            'pretrain.batch_size=10' \
                            'pretrain.regularization.l2=1e-3' \
                            'pretrain.kernel_l2=1e-5' \
                            'pretrain.preprocess_input="tensorflow.keras.applications.resnet_v2.preprocess_input"' \
                            'pretrain.early_stopping.patience=15' \
                            'pretrain.head_layers=[512,256]' \
                            'pretrain.frozen_layers="bn"' \
                            'pretrain.num_parallel_calls=5' \
                            'tags=["Baseline"]' \
                            'pipeline.stage_0.params.fit_class_weights=False' \
                            'use_tfrecords=True'



python ~/projects/pyleaves/pyleaves/pipelines/WandB_Leaves_vs_PNAS.py \
                            'WandB_dataset_0@WandB_dataset_0=Leaves-PNAS' \
                            'pretrain.model_name="resnet_50_v2"' \
                            'pretrain.target_size=[512,512]' \
                            'pretrain.num_epochs=120' \
                            'pretrain.augmentations.flip=1.0' \
                            'pretrain.augmentations.rotate=1.0' \
                            'pretrain.augmentations.sbc=0.0' \
                            'pretrain.lr=3e-4' \
                            'pretrain.batch_size=10' \
                            'pretrain.regularization.l2=1e-3' \
                            'pretrain.kernel_l2=1e-5' \
                            'pretrain.preprocess_input="tensorflow.keras.applications.resnet_v2.preprocess_input"' \
                            'pretrain.early_stopping.patience=15' \
                            'pretrain.head_layers=[512,256]' \
                            'pretrain.frozen_layers="bn"' \
                            'pretrain.num_parallel_calls=5' \
                            'tags=["Baseline"]' \
                            'pipeline.stage_0.params.fit_class_weights=False' \
                            'use_tfrecords=True'















python ~/projects/pyleaves/pyleaves/pipelines/WandB_baseline_finetuning_pipeline.py target_size=[299,299] batch_size=32 num_epochs=60 'frozen_layers=[0,-4]' num_parallel_calls=4



python ~/projects/pyleaves/pyleaves/pipelines/WandB_baseline_finetuning_pipeline.py dataset/0@dataset/0=Leaves_family_4 dataset/1@dataset/1=Fossil_family_4 pretrain.target_size=[768,768] pretrain.batch_size=16 pretrain.num_epochs=80 'pretrain.lr=3e-5' 'pretrain.frozen_layers="bn"' finetune.batch_size=16 finetune.num_epochs=80 finetune.lr=1e-3 'finetune.frozen_layers="bn"' num_parallel_calls=4


python ~/projects/pyleaves/pyleaves/pipelines/WandB_baseline_finetuning_pipeline.py dataset@dataset=Fossil_family_4 target_size=[299,299] batch_size=32 num_epochs=80 'frozen_layers=[0,-4]' num_parallel_calls=4


python ~/projects/pyleaves/pyleaves/pipelines/WandB_Leaves_vs_PNAS.py \
                            'WandB_dataset_0@WandB_dataset_0=Leaves_family_4-PNAS_family_100_test' \
                            'pretrain.model_name="resnet_50_v2"' \
                            'pretrain.target_size=[768,768]' \
                            'pretrain.num_epochs=120' \
                            'pretrain.augmentations.flip=1.0' \
                            'pretrain.augmentations.rotate=1.0' \
                            'pretrain.augmentations.sbc=0.0' \
                            'pretrain.lr=1e-4' \
                            'pretrain.batch_size=10' \
                            'pretrain.regularization.l2=1e-3' \
                            'pretrain.preprocess_input="tensorflow.keras.applications.resnet_v2.preprocess_input"' \
                            'pretrain.early_stopping.patience=15' \
                            'pretrain.head_layers=[512,256]' \
                            'pretrain.frozen_layers=[0,-1]' \
                            'pretrain.num_parallel_calls=5' \
                            'tags=["Baseline"]' \
                            'pipeline.stage_0.params.fit_class_weights=True'









python ~/projects/pyleaves/pyleaves/pipelines/WandB_Leaves_vs_PNAS.py \
                            'WandB_dataset_0@WandB_dataset_0=PNAS_family_100' \
                            'pretrain.model_name="resnet_50_v2"' \
                            'pretrain.target_size=[768,768]' \
                            'pretrain.num_epochs=120' \
                            'pretrain.augmentations.flip=1.0' \
                            'pretrain.augmentations.rotate=1.0' \
                            'pretrain.augmentations.sbc=0.0' \
                            'pretrain.lr=1e-4' \
                            'pretrain.batch_size=12' \
                            'pretrain.regularization.l2=1e-3' \
                            'pretrain.preprocess_input="tensorflow.keras.applications.resnet_v2.preprocess_input"' \
                            'pretrain.early_stopping.patience=15' \
                            'pretrain.head_layers=[512,256]' \
                            'pretrain.frozen_layers=[0,-1]' \
                            'pretrain.num_parallel_calls=5' \
                            'tags=["Baseline"]' \
                            'pipeline.stage_0.params.fit_class_weights=True'


python ~/projects/pyleaves/pyleaves/pipelines/WandB_Leaves_vs_PNAS.py \
                            'WandB_dataset_0@WandB_dataset_0=PNAS_family_100' \
                            'pretrain.model_name="resnet_50_v2"' \
                            'pretrain.target_size=[512,512]' \
                            'pretrain.num_epochs=120' \
                            'pretrain.augmentations.flip=1.0' \
                            'pretrain.augmentations.rotate=1.0' \
                            'pretrain.augmentations.sbc=0.0' \
                            'pretrain.lr=1e-5' \
                            'pretrain.batch_size=16' \
                            'pretrain.regularization.l2=1e-4' \
                            'pretrain.preprocess_input="tensorflow.keras.applications.resnet_v2.preprocess_input"' \
                            'pretrain.early_stopping.patience=15' \
                            'pretrain.head_layers=[512,256]' \
                            'pretrain.frozen_layers=null' \
                            'pretrain.weights=null' \
                            'pretrain.num_parallel_calls=4' \
                            'pipeline.stage_0.params.fit_class_weights=True' \
                            'tags=["weighted-crossentropy","train on PNAS","test on PNAS"]'


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
import sys
import shutil
from tqdm.auto import tqdm
from pathlib import Path

from typing import Union, Dict, List
import pandas as pd
from boltons.dictutils import OneToOne
from pprint import pprint
from box import Box
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
# import neptune
# from neptunecontrib.monitoring.metrics import log_confusion_matrix
# from neptunecontrib.monitoring.keras import NeptuneMonitor
# from neptunecontrib.api.table import log_table
from omegaconf import OmegaConf, ListConfig, DictConfig
import hydra
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback



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


def class_counts(y: np.ndarray) -> Dict[int,int]:
    return dict(zip(*np.unique(y, return_counts=True)))#.shape

def calc_class_weights(y: np.ndarray, balanced: bool=False) -> Dict[int,int]:    
    """

    Args:
        y (np.ndarray): sequence of all labels as ints
        balanced (bool, optional): Defaults to False.
            if False, returns all weights as 1's. Else, weight each class by its sample population

    Returns:
        Dict[int,int]: Maps {integer labels: label weight}
    """    
    counts = class_counts(y)
    total_samples = np.sum(list(counts.values()))
    total_classes = len(counts)
    class_weights = {}
    for class_i,count_i in counts.items():
        class_weights[class_i] = total_samples/(count_i*total_classes)
        
    if not balanced:
        class_weights = {i:w for i,w in zip(list(class_weights.keys()),list(np.ones_like(list(class_weights.values()))))}

    return class_weights


def data_df_2_tf_data(data, 
                      x_col='archive_path',
                      y_col='family',
                      training=False,
                      target_size=(256,256),
                      batch_size=16,
                      seed=None,
                      preprocess_input=None,
                      augmentations: Dict[str,float]=None,
                      num_parallel_calls=-1,
                      cache=False,
                      class_encodings: Dict[str,int]=None,
                      shuffle_first: bool=False,
                      fit_class_weights=False,
                      subset_key='train',
                      repeat=True,
                      use_tfrecords=False,
                      samples_per_shard=400,
                      tfrecord_dir='.'):
    """Helper function for loading data queried from WandB into tf.data.Datasets

    Args:
        data (pd.DataFrame): [description]
        x_col (str): Defaults to 'archive_path'
        y_col (str): Defaults to 'family'
        training (bool, optional): Defaults to False.
        target_size (tuple, optional): Defaults to (256,256).
        batch_size (int, optional): Defaults to 16.
        seed ([type], optional): Defaults to None.
        preprocess_input ([type], optional): Defaults to None.
            Function to apply for preprocessing input
        augmentations (Dict[str,float], optional): Defaults to None.
            Dict mapping the names of augmentations to their desired probability of application
        num_parallel_calls (int, optional): Defaults to -1.
        cache (bool, optional): Defaults to False.
        class_encodings (Dict[str,int], optional): Defaults to None.
            Optional dict mapping string labels to their int counterpart. Use this if it's desired to limit a dataset to a class list and encoding scheme already established by a prior training stage.
        shuffle_first (bool, optional): Defaults to False.
            If True, perform a full shuffle of the data prior to creating tf.data.Dataset
        fit_class_weights (bool, optional): Defaults to False.
            If False, return a dictionary of class weights all weighted to 1. Does not apply the class weights, only calculates them.

    Returns:
        [type]: [description]
    """        
    from pyleaves.utils.pipeline_utils import flip, rotate, rgb2gray_1channel, rgb2gray_3channel, sat_bright_con, _cond_apply, load_data_from_tfrecords
    import tensorflow as tf
    from tensorflow.keras import backend as K

    
    if shuffle_first:
        data = data.sample(frac=1)

    if class_encodings:
        data = data[data[y_col].apply(lambda x: x in list(class_encodings.keys()))]
        data = data.assign(y_true=data[y_col].apply(lambda x: class_encodings[x]),
                           x_true=data[x_col])

    
    paths = data[x_col].values.tolist()
    labels = data[y_col].values.tolist()
    
    if class_encodings:
        class_list = sorted(list(class_encodings.keys()))
    else:
        class_list = sorted(list(set(labels)))
    
    augmentations = augmentations or {}
    num_samples = len(paths)
    num_classes = len(class_list)
    class_encoder = OneToOne({label:i for i, label in enumerate(class_list)})

    labels = [class_encoder[l] for l in labels]
    

    if class_encodings is not None:
        #Encode according to a previously established str<->int mapping in class_encodings
        text_labels = decode_int2str(labels=labels, class_decoder=class_encoder.inv)
        labels = encode_str2int(labels=text_labels, class_encoder=class_encodings)

    # class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = calc_class_weights(labels, balanced=fit_class_weights)        
    # class_weights = {i:w for i,w in class_weights.items() if i in class_encodings.inv}
    
    ####################

    prepped_data = pd.DataFrame.from_records([{'path':path, 'label':label} for path, label in zip(paths, labels)])

    training = bool('train' in subset_key)
    if use_tfrecords:
        if target_size[0] > 768:
            tfrecord_target_shape = (*target_size,3)
        else:
            tfrecord_target_shape = (768,768,3)
        prepped_data = (paths, labels)
        tf_data = load_data_from_tfrecords(tfrecord_dir=tfrecord_dir,
                                           data=prepped_data,
                                           target_shape=tfrecord_target_shape,
                                           samples_per_shard=samples_per_shard,
                                           subset_key=subset_key,
                                           num_classes=num_classes)
    else:
        tf_data = load_data_from_tensor_slices(data=prepped_data, training=training, seed=seed, x_col='path', y_col='label', dtype=tf.float32)
        
    ####################
    # import pdb; pdb.set_trace()
    if preprocess_input is not None:
        tf_data = tf_data.map(lambda x,y: (preprocess_input(x), K.cast(y, dtype='uint8')), num_parallel_calls=num_parallel_calls)
    
    from functools import partial
    target_size = tuple(target_size)
    resize = partial(tf.image.resize, size=target_size)
    print('target_size = ', target_size)
    # import pdb; pdb.set_trace()
    tf_data = tf_data.map(lambda x,y: (resize(x), tf.one_hot(y, depth=num_classes)), num_parallel_calls=num_parallel_calls)

    if repeat:
        tf_data = tf_data.repeat()
    # TODO collect augmentation functions in a list and execute as a formal pipeline, abstracting away the logging & validation of results
    for aug in augmentations.keys():
        if 'flip' in aug:
            tf_data = tf_data.map(lambda x, y: _cond_apply(x, y, flip, prob=augmentations[aug], seed=seed), num_parallel_calls=num_parallel_calls)  
        if 'rotate' in aug:
            tf_data = tf_data.map(lambda x, y: _cond_apply(x, y, rotate, prob=augmentations[aug], seed=seed), num_parallel_calls=num_parallel_calls)
        if 'sbc' in aug:
            "sbc = saturation, brightness, contrast"
            tf_data = tf_data.map(lambda x, y: _cond_apply(x, y, sat_bright_con, prob=augmentations[aug], seed=seed), num_parallel_calls=num_parallel_calls)
    tf_data = tf_data.map(lambda x,y: rgb2gray_3channel(x, y), num_parallel_calls=-1)

    tf_data = tf_data.batch(batch_size)
    tf_data = tf_data.prefetch(-1)
    return {'data':tf_data, 
            "x_true":paths,
            "y_true":labels,
            'encoder':class_encoder,
            'num_samples':num_samples,
            'num_classes':num_classes,
            'class_weights':class_weights,
            'data_table':data}


def get_experiment_data(dataset_name='Fossil', threshold=4, test_size=0.3, version='latest', validation_split=0.1, seed=None,
                        preprocess_input=lambda x: x, target_size=(256,256), batch_size=16, augmentations={}, num_parallel_calls=1, fit_class_weights=False, artifact_name=None,
                        use_tfrecords=False, samples_per_shard=400, tfrecord_dir='.'):

    from pyleaves.utils.WandB_artifact_utils import load_dataset_from_artifact, load_Leaves_Minus_PNAS_dataset, load_Leaves_Minus_PNAS_test_dataset

    if dataset_name == "Leaves-PNAS":
        train_df, val_df = load_dataset_from_artifact(dataset_name=dataset_name, threshold=threshold, test_size=test_size, version=version, artifact_name=None)
        _, test_df = load_dataset_from_artifact(dataset_name='PNAS', threshold=100, test_size=0.5, version='latest')

    else:
        train_df, test_df = load_dataset_from_artifact(dataset_name=dataset_name, threshold=threshold, test_size=test_size, version=version, artifact_name=artifact_name)
        train_df, val_df = train_test_split(train_df, test_size=validation_split, random_state=seed, shuffle=True, stratify=train_df.family)

    train_data_info = data_df_2_tf_data(train_df,
                                        x_col='archive_path',
                                        y_col='family',
                                        training=True,
                                        preprocess_input=preprocess_input,
                                        seed=seed,
                                        target_size=target_size,
                                        batch_size=batch_size,
                                        augmentations=augmentations,
                                        num_parallel_calls=num_parallel_calls,
                                        cache=False,
                                        shuffle_first=True,
                                        fit_class_weights=fit_class_weights,
                                        subset_key='train',
                                        use_tfrecords=use_tfrecords,
                                        samples_per_shard=samples_per_shard,
                                        tfrecord_dir=tfrecord_dir)

    val_data_info = data_df_2_tf_data(val_df,
                                        x_col='archive_path',
                                        y_col='family',
                                        training=False,
                                        preprocess_input=preprocess_input,
                                        seed=seed,
                                        target_size=target_size,
                                        batch_size=batch_size,
                                        num_parallel_calls=num_parallel_calls,
                                        cache=True,
                                        shuffle_first=True,
                                        class_encodings=train_data_info['encoder'],
                                        subset_key='val',
                                        use_tfrecords=use_tfrecords,
                                        samples_per_shard=samples_per_shard,
                                        tfrecord_dir=tfrecord_dir)

    test_data_info = data_df_2_tf_data(test_df,
                                        x_col='archive_path',
                                        y_col='family',
                                        training=False,
                                        preprocess_input=preprocess_input,
                                        seed=seed,
                                        target_size=target_size,
                                        batch_size=batch_size,
                                        num_parallel_calls=num_parallel_calls,
                                        cache=True,
                                        shuffle_first=True,
                                        class_encodings=train_data_info['encoder'],
                                        subset_key='test',
                                        use_tfrecords=use_tfrecords,
                                        samples_per_shard=samples_per_shard,
                                        tfrecord_dir=tfrecord_dir)
    return train_data_info, val_data_info, test_data_info

#region
# Image plotting utils
def show_batch(image_batch, label_batch, title='', class_names=None):
    fig = plt.figure(figsize=(15, 15))

    # import pdb;pdb.set_trace()
    if label_batch.ndim==2:
        label_batch = np.argmax(label_batch, axis=-1)

    if isinstance(label_batch, (np.ndarray,)):
        label_batch = label_batch.tolist()
    if class_names is not None:
        label_batch = [class_names[l] for l in label_batch]

    
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
#endregion







@hydra.main(config_path='baseline_configs', config_name='WandB_Leaves_vs_PNAS_config')
def main(config):
    OmegaConf.set_struct(config, False)
    from pyleaves.utils import set_tf_config
    config.task = config.task or 1
    set_tf_config(gpu_num=None, num_gpus=1, wait=(config.task+1)*2)

    import tensorflow as tf
    from tensorflow.keras import backend as K
    K.clear_session()
    from pyleaves.utils.pipeline_utils import build_model
    from pyleaves.utils.callback_utils import ConfusionMatrixCallback
    from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
    from pyleaves.utils import pipeline_utils

    config.pretrain.regularization = config.pretrain.regularization or {}
    # config.pretrain.lr = float(config.pretrain.lr)
    # config.pretrain.augmentations['flip'] = float(config.pretrain.augmentations['flip'])
    config.pretrain.validation_split = float(config.pretrain.validation_split)

    if config.pretrain.preprocess_input == "tensorflow.keras.applications.resnet_v2.preprocess_input":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        print("Using preprocessing function: tensorflow.keras.applications.resnet_v2.preprocess_input")
    elif config.pretrain.preprocess_input == "tf.keras.applications.mobilenet.preprocess_input":
        from tensorflow.keras.applications.mobilenet import preprocess_input
        print("Using preprocessing function: tensorflow.keras.applications.mobilenet.preprocess_input")
    elif config.pretrain.preprocess_input == "tf.keras.applications.inception_v3.preprocess_input":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        print("Using preprocessing function: tensorflow.keras.applications.inception_v3.preprocess_input")
    else:
        preprocess_input = None
        print("Using no preprocess_input function")
    os.makedirs(os.path.dirname(config.pretrain.saved_model_path), exist_ok=True)

#region
#     from pyleaves.utils.WandB_artifact_utils import load_Leaves_Minus_PNAS_dataset, load_Leaves_Minus_PNAS_test_dataset

#     if config.dataset_name["0"] == "Leaves_family_4-PNAS_family_100_test":
#         train_df, test_df, pnas_train_df = load_Leaves_Minus_PNAS_test_dataset()

#         if config.pretrain.validation_split:
#             train_df, val_df = train_test_split(train_df, test_size=config.pretrain.validation_split, random_state=config.seed, shuffle=True, stratify=train_df.family)

#         train_data_info = data_df_2_tf_data(train_df,
#                                             x_col='archive_path',
#                                             y_col='family',
#                                             training=True,
#                                             preprocess_input=preprocess_input,
#                                             seed=config.seed,
#                                             target_size=config.pretrain.target_size,
#                                             batch_size=config.pretrain.batch_size,
#                                             augmentations=config.pretrain.augmentations,
#                                             num_parallel_calls=config.pretrain.num_parallel_calls,
#                                             cache=False,
#                                             shuffle_first=True,
#                                             fit_class_weights=config.pipeline.stage_0.params.fit_class_weights)

#         val_data_info = data_df_2_tf_data(val_df,
#                                           x_col='archive_path',
#                                           y_col='family',
#                                           training=False,
#                                           preprocess_input=preprocess_input,
#                                           seed=config.seed,
#                                           target_size=config.pretrain.target_size,
#                                           batch_size=config.pretrain.batch_size,
#                                           num_parallel_calls=config.pretrain.num_parallel_calls,
#                                           cache=True,
#                                           shuffle_first=True,
#                                           class_encodings=train_data_info['encoder'])

#         test_data_info = data_df_2_tf_data(test_df,
#                                            x_col='archive_path',
#                                            y_col='family',
#                                            training=False,
#                                            preprocess_input=preprocess_input,
#                                            seed=config.seed,
#                                            target_size=config.pretrain.target_size,
#                                            batch_size=config.pretrain.batch_size,
#                                            num_parallel_calls=config.pretrain.num_parallel_calls,
#                                            cache=True,
#                                            shuffle_first=True,
#                                            class_encodings=train_data_info['encoder'])
#
#         # pnas_train_data_info = data_df_2_tf_data(pnas_train_df,
#         #                                          x_col='archive_path',
#         #                                          y_col='family',
#         #                                          training=True,
#         #                                          preprocess_input=preprocess_input,
#         #                                          seed=config.seed,
#         #                                          target_size=config.pretrain.target_size,
#         #                                          batch_size=config.pretrain.batch_size,
#         #                                          augmentations=config.pretrain.augmentations,
#         #                                          num_parallel_calls=config.pretrain.num_parallel_calls,
#         #                                          cache=True,
#         #                                          shuffle_first=True,
#         #                                          fit_class_weights=config.pipeline.stage_0.params.fit_class_weights)
# 

#     elif config.dataset_name["0"] == "PNAS_family_100":
#         _, test_df, train_df = load_Leaves_Minus_PNAS_test_dataset()

#         if config.pretrain.validation_split:
#             train_df, val_df = train_test_split(train_df, test_size=config.pretrain.validation_split, random_state=config.seed, shuffle=True, stratify=train_df.family)

#         train_data_info = data_df_2_tf_data(train_df,
#                                             x_col='archive_path',
#                                             y_col='family',
#                                             training=True,
#                                             preprocess_input=preprocess_input,
#                                             seed=config.seed,
#                                             target_size=config.pretrain.target_size,
#                                             batch_size=config.pretrain.batch_size,
#                                             augmentations=config.pretrain.augmentations,
#                                             num_parallel_calls=config.pretrain.num_parallel_calls,
#                                             cache=False,
#                                             shuffle_first=True,
#                                             fit_class_weights=config.pipeline.stage_0.params.fit_class_weights)

#         val_data_info = data_df_2_tf_data(val_df,
#                                           x_col='archive_path',
#                                           y_col='family',
#                                           training=False,
#                                           preprocess_input=preprocess_input,
#                                           seed=config.seed,
#                                           target_size=config.pretrain.target_size,
#                                           batch_size=config.pretrain.batch_size,
#                                           num_parallel_calls=config.pretrain.num_parallel_calls,
#                                           cache=True,
#                                           shuffle_first=True,
#                                           class_encodings=train_data_info['encoder'])

#         test_data_info = data_df_2_tf_data(test_df,
#                                            x_col='archive_path',
#                                            y_col='family',
#                                            training=False,
#                                            preprocess_input=preprocess_input,
#                                            seed=config.seed,
#                                            target_size=config.pretrain.target_size,
#                                            batch_size=config.pretrain.batch_size,
#                                            num_parallel_calls=config.pretrain.num_parallel_calls,
#                                            cache=True,
#                                            shuffle_first=True,
#                                            class_encodings=train_data_info['encoder'])
#endregion

    train_data_info, val_data_info, test_data_info = get_experiment_data(dataset_name=config.pretrain.dataset_name,
                                                                         threshold=config.pretrain.threshold,
                                                                         test_size=config.pretrain.test_size,
                                                                         version='latest',
                                                                         validation_split=config.pretrain.validation_split, 
                                                                         seed=config.seed,
                                                                         preprocess_input=preprocess_input,
                                                                         target_size=config.pretrain.target_size,
                                                                         batch_size=config.pretrain.batch_size,
                                                                         augmentations=config.pretrain.augmentations, 
                                                                         num_parallel_calls=config.num_parallel_calls,
                                                                         fit_class_weights=config.pipeline.stage_0.fit_class_weights,
                                                                         artifact_name=config.pretrain.artifact_name,
                                                                         use_tfrecords=config.use_tfrecords,
                                                                         samples_per_shard=config.samples_per_shard,
                                                                         tfrecord_dir=config.tfrecord_dir)

    train_data = train_data_info['data']
    val_data = val_data_info['data']
    test_data = test_data_info['data']

    class_weights = train_data_info['class_weights']

    stage_0_config = config.pipeline.stage_0
    stage_0_config.num_samples_train = train_data_info['num_samples']
    stage_0_config.num_samples_val = val_data_info['num_samples']
    stage_0_config.num_samples_test = test_data_info['num_samples']
    stage_0_config.num_classes = train_data_info['num_classes']
    steps_per_epoch=int(np.ceil(stage_0_config.num_samples_train/config.pretrain.batch_size))
    validation_steps=int(np.ceil(stage_0_config.num_samples_val/config.pretrain.batch_size))
    test_steps=int(np.ceil(stage_0_config.num_samples_test/config.pretrain.batch_size))

    ################################################################################

    model_config = config.pretrain
    model_config.num_classes = stage_0_config.num_classes
    model_config.input_shape = (*config.pretrain.target_size,3)            
    model = build_model(model_config)

    ################################################################################
    ################################################################################
    ################################################################################

    id = config.run_id or wandb.util.generate_id()


    run = wandb.init(entity=config.entity, 
                     project=config.project_name,
                     name=config.run_name,
                     job_type=config.job_type,
                     tags=config.tags,
                     sync_tensorboard=True)
    run.config.update(OmegaConf.to_container(config, resolve=True))

    initial_epoch = wandb.run.step or 0
    if wandb.run.resumed:
        # restore the best model
        print(f'Restoring model from checkpoint at epoch {initial_epoch}')
        model = tf.keras.models.load_model(wandb.restore("model-best.h5").name)



    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=3, min_lr=1e-5)

    class_names = train_data_info['encoder'].inv
    # train_cb = lambda : ((img, label) for img, label in iter(train_data.take(12).unbatch()))
    val_cb = lambda : ((img, label) for img, label in iter(val_data.take(12).unbatch()))

    val_imgs, val_labels = [], []
    for img, lbl in val_cb():
        val_imgs.append(img)
        val_labels.append(lbl)
    val_imgs = np.stack([img for img in val_imgs])
    val_labels = np.stack([lbl for lbl in val_labels])
    callbacks = [reduce_lr,
                 TensorBoard(log_dir=config.log_dir, histogram_freq=2, write_images=True),
                 WandbCallback(save_model=True,
                               monitor='val_loss', #log_gradients=False,#True,
                               data_type='image',#                               training_data=(train_imgs,train_labels),
                               labels=list(class_names.values()),
                               predictions=64,
                               generator=tf.data.Dataset.from_generator(val_cb,
                                                                       (tf.float64, tf.float64),
                                                                       (tf.TensorShape(list(val_imgs.shape)), tf.TensorShape(list(val_labels.shape))))),
                 EarlyStopping(monitor=config.pretrain.early_stopping.monitor,
                            patience=config.pretrain.early_stopping.patience,
                            min_delta=config.pretrain.early_stopping.min_delta, 
                            verbose=1,
                            restore_best_weights=config.pretrain.early_stopping.restore_best_weights),
                ConfusionMatrixCallback(config.log_dir,
                                        val_imgs=val_imgs,
                                        val_labels=val_labels,
                                        classes=list(class_names.values()),
                                        freq=1,
                                        seed=config.seed,
                                        log_wandb=True)]

    # image_batch, label_batch = next(iter(train_data))
    image_iters = [('train', iter(train_data)), ('val', iter(val_data)), ('test', iter(test_data))]

    for subset_name, subset_iter in image_iters:
        print(f'[INFO] logging images from {subset_name}.')
        image_batch, label_batch = next(iter(subset_iter))
        fig = show_batch(image_batch.numpy(), label_batch.numpy(), title=subset_name, class_names=class_names)
        wandb.log({f'{subset_name}_image_batch': [wandb.Image(fig)]}, commit=False)

    # image_batch, label_batch = next(iter(val_data))
    # fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='val', class_names=class_names)
    # wandb.log({'val_image_batch': [wandb.Image(fig)]}, commit=False)

    # image_batch, label_batch = next(iter(test_data))
    # fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='test', class_names=class_names)
    # wandb.log({'test_image_batch': [wandb.Image(fig)]})#, commit=False)

    print('[BEGINNING STAGE_0: PRE-TRAINING+VALIDATION]')
    # import pdb;pdb.set_trace()
    try:
        history = model.fit(train_data,
                            epochs=config.pretrain.num_epochs,
                            callbacks=callbacks,
                            validation_data=val_data,
                            validation_freq=1,
                            shuffle=True,
                            class_weight=class_weights,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            initial_epoch=initial_epoch,
                            verbose=1)

    except Exception as e:
        raise e

    model.save(config.pretrain.saved_model_path)
    
    artifact = wandb.Artifact(type='model', name=f'{config.pretrain.model_name}-{config.dataset_name["0"]}')
    if config.pretrain.saved_model_path.endswith('h5'):
        artifact.add_file(config.pretrain.saved_model_path, name='trained_model')
    else:
        artifact.add_dir(config.pretrain.saved_model_path, name='trained_model')
    run.log_artifact(artifact)

    print('[STAGE_0 COMPLETED]')
    print(f'Saved trained model to {config.pretrain.saved_model_path}')
    
    print('BEGINNING STAGE_1: TEST EVALUATION' )
    perform_evaluation_stage(model, test_data_info, class_encoder=train_data_info['encoder'], batch_size=config.pretrain.batch_size, subset='test')


    # subset='test'
    # test_iter = test_data_info['data_iterator']
    # y_true = test_iter.labels
    # classes = test_iter.class_indices
    # eval_iter = test_data.unbatch().take(len(y_true)).batch(config.batch_size)
    # y, y_hat, y_prob = evaluate(model, eval_iter, y_true=y_true, classes=classes, steps=test_steps, experiment=experiment, subset=subset)

    # print('y_prob.shape =', y_prob.shape)
    # predictions = pd.DataFrame({'y':y,'y_pred':y_hat})
    # log_table(f'{subset}_labels_w_predictions',predictions, experiment=experiment)
    # y_prob_df = pd.DataFrame(y_prob, columns=list(classes.keys()))
    # log_table(f'{subset}_probabilities',y_prob_df,experiment=experiment)

    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    ###############################################################################
    ###############################################################################
    
    ###############################################################################
    ###############################################################################
    ###############################################################################





    # # if "zero_shot_test" in config:
    # class_encodings = train_data_info['encoder']

    # train_data_info = load_data_by_subset(config.finetune.train_image_dir,
    #                                     subset='train',
    #                                     preprocess_input=preprocess_input,
    #                                     validation_split=config.finetune.validation_split,
    #                                     seed=config.seed,
    #                                     target_size=config.finetune.target_size,
    #                                     batch_size=config.finetune.batch_size,
    #                                     augmentations=config.finetune.augmentations,
    #                                     num_parallel_calls=config.finetune.num_parallel_calls,
    #                                     class_encodings=class_encodings)

    # val_data_info = load_data_by_subset(config.finetune.train_image_dir,
    #                                     subset='validation',
    #                                     preprocess_input=preprocess_input,
    #                                     validation_split=config.finetune.validation_split,
    #                                     seed=config.seed,
    #                                     target_size=config.finetune.target_size,
    #                                     batch_size=config.finetune.batch_size,
    #                                     augmentations=config.finetune.augmentations,
    #                                     num_parallel_calls=config.finetune.num_parallel_calls,
    #                                     class_encodings=class_encodings)

    # test_data_info = load_data_by_subset(config.finetune.test_image_dir,
    #                                     subset='test',
    #                                     preprocess_input=preprocess_input,
    #                                     validation_split=config.finetune.validation_split,
    #                                     seed=config.seed,
    #                                     target_size=config.finetune.target_size,
    #                                     batch_size=config.finetune.batch_size,
    #                                     augmentations=config.finetune.augmentations,
    #                                     num_parallel_calls=config.finetune.num_parallel_calls,
    #                                     class_encodings=class_encodings)
    # # test_data_info = load_Fossil_family_4_subset(config, subset='test', preprocess_input=preprocess_input, class_encodings=class_encodings)

    # train_data = train_data_info['data']
    # val_data = val_data_info['data']
    # test_data = test_data_info['data']

    # stage_2_config = config.pipeline.stage_2
    # # stage_2_config.class_encodings = dict(train_data_info['encoder'])
    # stage_2_config.num_samples_train = train_data_info['num_samples']
    # stage_2_config.num_samples_val = val_data_info['num_samples']
    # stage_2_config.num_samples_test = test_data_info['num_samples']
    # stage_2_config.num_classes = train_data_info['num_classes']
    # config.finetune.stage = stage_2_config
    # steps_per_epoch=int(np.ceil(stage_2_config.num_samples_train/config.finetune.batch_size))
    # validation_steps=int(np.ceil(stage_2_config.num_samples_val/config.finetune.batch_size))
    # test_steps=int(np.ceil(stage_2_config.num_samples_test/config.finetune.batch_size))
    # ################################################################################
    # model_config = config.finetune
    # model_config.num_classes = model_config.stage.num_classes
    # model_config.input_shape = (*config.finetune.target_size,3)            
    # model = build_model(model_config)

    # ################################################################################
    # ################################################################################
    # ################################################################################
    # # neptune.init(project_qualified_name=project_name)

    # # neptune_config = log_neptune_config(config)


    # callbacks = [TensorBoard(log_dir=config.log_dir, histogram_freq=2, write_grads=True, write_images=True),
    #              WandbCallback(),
    #              EarlyStopping(monitor=config.finetune.early_stopping.monitor,
    #                         patience=config.finetune.early_stopping.patience,
    #                         min_delta=config.finetune.early_stopping.min_delta, 
    #                         verbose=1, 
    #                         restore_best_weights=config.finetune.early_stopping.restore_best_weights)]

    # class_names = train_data_info['encoder'].inv
    # image_batch, label_batch = next(iter(train_data))
    # fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='train', class_names=class_names)
    # wandb.log({'target_train_image_batch': fig}, commit=False)

    # image_batch, label_batch = next(iter(val_data))
    # fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='val', class_names=class_names)
    # wandb.log({'target_val_image_batch': fig}, commit=False)

    # image_batch, label_batch = next(iter(test_data))
    # fig = show_batch(image_batch.numpy(), label_batch.numpy(), title='test', class_names=class_names)
    # wandb.log({'target_test_image_batch': fig})

    # print('BEGINNING STAGE_2: FINETUNE+VALIDATION' )
    # try:
    #     history = model.fit(train_data,
    #                         epochs=config.finetune.num_epochs,
    #                         callbacks=callbacks,
    #                         validation_data=val_data,
    #                         validation_freq=1,
    #                         shuffle=True,
    #                         steps_per_epoch=steps_per_epoch,
    #                         validation_steps=validation_steps,
    #                         verbose=1)

    # except Exception as e:
    #     raise e

    # model.save(config.finetune.saved_model_path)
    # print('[STAGE_2 COMPLETED]')
    # print(f'Saved finetuned model to {config.finetune.saved_model_path}')
    # ################################################################################
    # ################################################################################
    # ################################################################################

    # print('BEGINNING STAGE_3: FINAL TEST EVALUATION' )
    # perform_evaluation_stage(model, test_data_info, batch_size=config.finetune.batch_size, experiment=None, subset='final_test')
    # print('[STAGE_3 COMPLETED]')

    # print(['[FINISHED PRETRAINING, TESTING, FINETUNING AND FINAL TESTING]'])

    return



# def plot_classification_report(report_df: pd.DataFrame):
#     """

#     Example:

#     report = classification_report(y_true, y_hat, labels=labels, target_names=target_names, output_dict=True)
#     fig = plot_classification_report(report_df=report.T)

#     Args:
#         report_df (pd.DataFrame): [description]

#     Returns:
#         [type]: [description]
#     """    
#     fig, axes =  plt.subplots(4,1, figsize=(20,15), sharex=True)
#     report_df.iloc[:-3,:].plot(kind='bar', ax=axes, subplots=True, rot=90)#, figsize=(20,15))
#     plt.subplots_adjust(left=0.2, bottom=0.2)
#     ##########################################
#     summary_report = report_df.iloc[-3:,:].T
#     col_headers = summary_report.columns.tolist()
#     row_headers = summary_report.index.tolist()
#     summary_text = []
#     for row_name, row in summary_report.iterrows():
#         summary_text.append([f'{val:.2f}' for val in row.values])

#     plt.table(cellText=summary_text,
#               rowLabels=row_headers,
#               colLabels=col_headers,
#               bbox=(0.0, -1.2, 0.5, 0.5),
#               loc='bottom')
#     return fig
    



def perform_evaluation_stage(model, test_data_info, class_encoder, batch_size, subset='final_test'):
    """Evaluate data in test_data_info on a trained model and log to neptune experiment, prefixed with label subset

    Args:
        model ([type]): [description]
        test_data_info ([type]): [description]
        batch_size ([type]): [description]
        experiment ([type], optional): [description]. Defaults to None.
        subset (str, optional): [description]. Defaults to 'final_test'.

    Returns:
        [type]: [description]
    """    
    # test_iter = test_data_info['data_iterator']
    test_data = test_data_info['data']
    test_steps=int(np.ceil(test_data_info['num_samples']/batch_size))
    y_true = np.array(test_data_info['y_true'])
    eval_iter = test_data.unbatch().take(len(y_true)).batch(batch_size)
    y, y_hat, y_prob = evaluate(model, eval_iter, y_true=y_true, steps=test_steps, class_encoder=class_encoder, subset=subset, test_data_info=test_data_info)

    print('y_prob.shape =', y_prob.shape)
    predictions = pd.DataFrame({'y':y,'y_pred':y_hat})
    predictions.to_csv(f'{subset}_labels_w_predictions.csv')


    class_list = list(class_encoder.keys())
    y_prob_df = pd.DataFrame(y_prob, columns=class_list)
    y_prob_df.to_csv(f'{subset}_probabilities.csv')

    artifact = wandb.Artifact(f'{subset}_probabilities', type='result')
    artifact.add_file(f'{subset}_labels_w_predictions.csv',name=f'{subset}_labels_w_predictions.csv')
    artifact.add_file(f'{subset}_probabilities.csv',name=f'{subset}_probabilities.csv')
    wandb.run.use_artifact(artifact)
    wandb.run.log_artifact(artifact)

    wandb.sklearn.plot_confusion_matrix(y, y_hat, class_list)

    return y, y_hat, y_prob








def evaluate(model, data_iter, y_true, steps: int, output_dict: bool=True, class_encoder=None, subset='val', test_data_info=None):
    # num_classes = data_iter.num_classes

    y_prob = model.predict(data_iter, steps=steps, verbose=1)
    target_names = list(class_encoder.keys())
    labels = [class_encoder[text_label] for text_label in target_names]

    y_hat = y_prob.argmax(axis=1)
    print('y_hat.shape = ', y_hat.shape)
    if y_true.ndim > 1:
        y_true = y_true.argmax(axis=1)
    print('y_true.shape = ', y_true.shape)

    if test_data_info is not None:
        data_table = test_data_info['data_table']
        data_table = data_table.assign(**{class_encoder.inv[i]:y_prob[:,i] for i in range(y_prob.shape[1])})
        data_table = data_table.assign(y_pred=y_hat)

        table = wandb.Table(dataframe=data_table)
        wandb.log({"test_data_with_probabilities" : table}) #wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")


    try:
        report = classification_report(y_true, y_hat, labels=labels, target_names=target_names, output_dict=output_dict)
        if type(report)==dict:
            report = pd.DataFrame(report).T

        report.to_csv(f'{subset}_classification_report.csv')
        report_table = wandb.Table(dataframe=report)
        wandb.log({f'{subset}_classification_report':report_table})

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
        wandb.log({f'{subset}_{m}': result})

    return y_true, y_hat, y_prob



# TODO Implement the below wandb plot predictions function
 
# wandb.log() – Logs custom objects – these can be images, videos, audio files, HTML, plots, point clouds etc. Here we use wandb.log to log images of Simpson characters overlaid with actual and predicted labels.

# def make_predictions(nn_model):
#   predicted_images = []
#   for i in range(20):
#     character = character_names[i]
#     # Read in a character image from the test dataset
#     image = cv2.imread(np.random.choice([k for k in glob.glob('simpsons-dataset/kaggle_simpson_testset/kaggle_simpson_testset/*.*') if character in k]))
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Resize image and normalize it
#     pic = cv2.resize(image, (64, 64)).astype('float32') / 255.
    
#     # Get predictions for the character
#     prediction = nn_model.predict(pic.reshape(1, 64, 64,3))[0]
    
#     # Get true name of the character
#     name = character.split('_')[0].title()
    
#     # Format predictions to string to overlay on image
#     text = sorted(['{:s} : {:.1f}%'.format(character_names[k].split('_')[0].title(), 100*v) for k,v in enumerate(prediction)], 
#         key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
    
#     # Upscale image
#     img = cv2.resize(img, (352, 352))
    
#     # Create background to overlay text on
#     cv2.rectangle(img, (0,260),(215,352),(255,255,255), -1)
    
#     # Add text to image -  We add the true probabilities and predicted probabilities on each of the images in the test dataset
#     font = cv2.FONT_HERSHEY_DUPLEX
#     cv2.putText(img, 'True Name : %s' % name, (10, 280), font, 0.7,(73,79,183), 2, cv2.LINE_AA)
#     for k, t in enumerate(text):
#         cv2.putText(img, t, (10, 300+k*18), font, 0.65,(0,0,0), 2, cv2.LINE_AA)
        
#     # Add predicted image from test dataset with annotations to array
#     predicted_images.append(wandb.Image(img, caption="Actual: %s" % name))     
        
#   # Log images from test set to wandb automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
#   # You can use wandb.log() to log any images, video, audio, 3D objects like point clouds, plots, HTML etc.
#   # For full details please see https://docs.wandb.com/library/python/log
#   wandb.log({"predictions": predicted_images})




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