# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: create_tfrecords.py


'''
TBD

'''
import argparse
import concurrent
import cv2
# import dataset
from more_itertools import chunked, collapse, unzip
from functools import partial
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stuf import stuf
import sys
import tensorflow as tf
# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# tf.enable_eager_execution()

from tensorflow.data.experimental import AUTOTUNE
from pyleaves.utils.img_utils import TFRecordCoder, plot_image_grid
from pyleaves.configs.config_v1 import DatasetConfig
from pyleaves import leavesdb
from pyleaves.leavesdb.tf_utils.tf_utils import (train_val_test_split,
                                                load_and_format_dataset_from_db,
                                                check_if_tfrecords_exist)
from pyleaves.utils import ensure_dir_exists, set_visible_gpus
from pyleaves.leavesdb.tf_utils.tf_utils import bytes_feature, int64_feature, float_feature
from pyleaves.tests.test_utils import timeit, timelined_benchmark, draw_timeline, map_decorator
from pyleaves.leavesdb.experiments_db import DataBase, Table, TFRecordsTable, EXPERIMENTS_DB, EXPERIMENTS_SCHEMA, TFRecordItem

def save_tfrecords(data: list,
                   output_dir,
                   file_prefix,
                   target_size=(224,224),
                   num_channels=3,
                   num_classes=100,
                   num_shards=10,
                   TFRecordItem_factory=None,
                   tfrecords_table=TFRecordsTable(db_path=EXPERIMENTS_DB),
                   verbose=True):
    """
    Input data as a list of tuples containing (filepath, label) pairs to be split into
    {num_shards} shards, then loaded in parallel and saved to each shard.

    Note: This differs from save_trainvaltest_tfrecords() mainly by moving some of the original functionality
    outside the function, requiring the user to configure how the data is saved with more manual control.

    Parameters
    ----------
    data : list
        List of tuples representing image file paths and their encoded labels.
    output_dir : type
        Directory in which to save TFRecord shards.
    file_prefix : str
        The string that should serve as filename prefix to distinguish these shards from
        any others in the same directory
    target_size : type
        Description of parameter `target_size`.
    num_channels : type
        Description of parameter `num_channels`.
    num_classes : type
        Description of parameter `num_classes`.
    num_shards : type
        Description of parameter `num_shards`.
    verbose : type
        Description of parameter `verbose`.

    Returns
    -------
    pyleaves.utils.img_utils.TFRecordCoder
        The coder object used to encode the shards, containings a file_logs attribute
        referencing all created TFRecord files.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    file_logs = {}
    coders = {}
    num_samples=len(data)
    print(f'Starting to split {file_prefix} subset with {num_samples} total samples into {num_shards} shards')
    coder = TFRecordCoder(data,
                          output_dir,
                          subset=file_prefix,
                          target_size=target_size,
                          num_channels=num_channels,
                          num_shards=num_shards,
                          num_classes=num_classes,
                          TFRecordItem_factory=TFRecordItem_factory,
                          tfrecords_table=tfrecords_table)
    coder.execute_convert()
    file_logs.update({file_prefix:coder.filepath_log})

    coder.file_logs = file_logs
    coder.tfrecord_dir_tree = output_dir
    return coder





##################################################################
#TODO Remove these functions from script, their functionality should be defined elsewhere (likely img_utils or tf_utils)
def encode_example(img, label_int):
    shape = img.shape
    img_buffer = encode_image(img).tostring()

    features = {
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/bytes': bytes_feature(img_buffer),
        'label': int64_feature(label_int)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def decode_example(serialized_example):
    feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/bytes': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
                            }
    features = tf.io.parse_single_example(serialized_example,features=feature_description)

    img = tf.image.decode_jpeg(features['image/bytes'], channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return img, label
##################################################################

def encode_image(img):
    '''
    Encode image array as jpg prior to constructing Examples for TFRecords for compressed file size.
    '''
    return tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)

def decode_image(img_string, channels=3):
    return tf.io.decode_image(img_string,channels=channels)
##################################################################
def load_and_encode_example(path, label, target_size = (224,224)):
    img = load_image(path, target_size=target_size)
    return encode_example(img,label)

##################################################################







##################################################################
##################################################################
# Rename this malarkey
def save_trainvaltest_tfrecords(dataset_name='PNAS',
                                output_dir = os.path.expanduser(r'~/data'),
                                target_size=(224,224),
                                num_channels=3,
                                low_count_threshold=10,
                                val_size=0.3,
                                test_size=0.3,
                                num_shards=10,
                                data_splits=None,
                                metadata_splits=None,
                                verbose=True):
    '''
    Load images from dataset_name, split into train, val, and test sets.
    Then iterate over these subsets and feed to function that shards data,
    then distributes data to process pool,
    where each process writes its shard to an individual TFRecord.

    '''
    if (not data_splits) or (not metadata_splits):
        data_splits, metadata_splits = load_and_format_dataset_from_db(dataset_name=dataset_name, low_count_threshold=low_count_threshold, val_size=val_size, test_size=test_size)

    os.makedirs(output_dir, exist_ok=True)
    file_logs = {}
    coders = {}

#     file_log = {'label_map':metadata_splits['label_map']}
    for split_name, split_data in data_splits.items():
        y_col = 'label'
        if 'y' in split_data.keys(): y_col = 'y'
        num_samples = len(split_data[y_col])
        num_classes = metadata_splits[split_name]['num_classes']
        if num_samples > 0:
            print(f'Starting to split {split_name} subset with {num_samples} total samples into {num_shards} shards')
            coder = TFRecordCoder(split_data, output_dir, subset=split_name, target_size=target_size, num_channels=num_channels, num_shards=num_shards, num_classes=num_classes)
            coder.execute_convert()
            coder.label_map = metadata_splits['label_map']
            file_logs.update({split_name:coder.filepath_log})
#             coders.update({split_name:coder})

    coder.file_logs = file_logs
    coder.tfrecord_dir_tree = output_dir
    return coder




def main(config=None,  record_subdirs=None, data_splits=None, metadata_splits=None):
    '''
    Example Jupyter notebook command:

        %run create_tfrecords.py -d Fossil -o /home/jacob/data -thresh 3 -val 0.3 -test 0.3

    '''
    if config is None:
        config = DatasetConfig()

    #TODO add ability to save TFRecords with configurable num_channels, currently just saves rgb

    print(f'running main in {__file__}, tfrecord_root_dir = {config.tfrecord_root_dir}, record_subdirs = {record_subdirs}')

    if record_subdirs is None:
        config.output_dir = os.path.join(tfrecord_root_dir,config.dataset_name,f'num_channels-3_thresh-{config.low_class_count_threshold}',f"val_size={config.data_splits['val_size']}-test_size={config.data_splits['test_size']}")
    else:
        config.output_dir = os.path.join(config.tfrecord_root_dir, *record_subdirs)

    file_log = check_if_tfrecords_exist(config.output_dir)

    print('filename_log = ', file_log)
#     if file_log == None:
    if True:
        file_log = {'train':[],'val':[],'test':[]}
        print('Entering save_trainvaltest_tfrecords')
        coder = save_trainvaltest_tfrecords(dataset_name=config.dataset_name,
                                                   output_dir=config.output_dir,
                                                   target_size=config.target_size,
                                                   num_channels=config.num_channels,
                                                   low_count_threshold=config.low_class_count_thresh,
                                                   val_size=config.data_splits['val_size'],
                                                   test_size=config.data_splits['test_size'],
                                                   num_shards=config.num_shards,
                                                   data_splits=data_splits,
                                                   metadata_splits=metadata_splits)
#         label_map = coder.label_map

        for subset, records in coder.file_logs.items():
            file_log[subset] = [os.path.join(config.output_dir,subset,record_fname) for record_fname in sorted(records)]

        return coder, file_log


    else:
        print(f'Found {len(file_log.keys())} subsets of tfrecords already saved, skipping creation process.')

# def read():
#
#     coder, file_log = main(DatasetConfig(dataset_name='Leaves'))#, target_size=(768,768))
#
# #     tfrecord_paths = file_log['train']
# #     tfrecord_paths = file_log['val']
#     tfrecord_paths = file_log['test']
#
# #     data = coder.read_tfrecords(tfrecord_paths)
# #     label_map = coder.label_map
#
#     data = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
#             .apply(lambda x: tf.data.TFRecordDataset(x)) \
#             .map(self.decode_example,num_parallel_calls=AUTOTUNE) \
#             .batch(batch_size,drop_remainder=drop_remainder) \
#             .prefetch(AUTOTUNE)
#
# #     for imgs, labels in data.take(1):
# #         labels = [coder.label_map[label] for label in labels.numpy()]
# #         plot_image_grid(imgs, labels, 4, 8)
#
#     for i, (imgs, labels) in enumerate(data):
#         print(i, imgs.shape, len(labels))
#
#     return coder


if __name__ == "__main__":

    '''
    Instructions:

    Open a jupyter console and type "%run create_tfrecords.py"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', default='PNAS', type=str,help='Name of dataset of images to use for creating TFRecords')
    parser.add_argument('-o', '--output_dir', default=r'/media/data/jacob/Fossil_Project/tfrecord_data', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    parser.add_argument('-thresh', '--low_count_threshold', default=3, type=int, help='Min population of a class below which we will exclude the class entirely.')
    parser.add_argument('-val', '--val_size', default=0.3, type=float, help='Fraction of train to use as validation set. Calculated after splitting train and test')
    parser.add_argument('-test', '--test_size', default=0.3, type=float, help='Fraction of full dataset to use as test set. Remaining fraction will be split into train/val sets.')
    parser.add_argument('-shards', '--num_shards', default=10, type=int, help='Number of shards to split each data subset into')
    parser.add_argument('-time', '--timeit', default=False, type=bool, help='If set to True, run speed tests on generated TFRecords with tf.data.Dataset readers.')
    args = parser.parse_args()


    config = DatasetConfig(dataset_name=args.dataset_name,
                           label_col='family',
                           target_size=(224,224),
                           low_class_count_thresh=args.low_count_threshold,
                           data_splits={'val_size':args.val_size,'test_size':args.test_size},
                           tfrecord_root_dir=args.output_dir,
                           num_shards=args.num_shards)


    main(config)
