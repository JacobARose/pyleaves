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

# import contextlib2
# from object_detection.dataset_tools import tf_record_creation_util
# from sklearn.model_selection import train_test_split
from stuf import stuf
import sys
import tensorflow as tf
# tf.enable_eager_execution()

from tensorflow.data.experimental import AUTOTUNE
# from pyleaves.data_pipeline.preprocessing import generate_encoding_map #encode_labels, filter_low_count_labels, one_hot_encode_labels, one_hot_decode_labels
from pyleaves.analysis.img_utils import TFRecordCoder, plot_image_grid
from pyleaves.config import DatasetConfig
from pyleaves import leavesdb
from pyleaves.leavesdb.tf_utils.tf_utils import (train_val_test_split,
                                                load_and_format_dataset_from_db,
                                                check_if_tfrecords_exist)
from pyleaves.utils import ensure_dir_exists, set_visible_gpus
from pyleaves.tests.test_utils import timeit, timelined_benchmark, draw_timeline, map_decorator

# gpu_ids = [0]
# set_visible_gpus(gpu_ids)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
##################################################################
def encode_example(img, label_int):
    shape = img.shape
    img_buffer = encode_image(img).tostring()

    features = {
        'image/height': _int64_feature(shape[0]),
        'image/width': _int64_feature(shape[1]),
        'image/channels': _int64_feature(shape[2]),
        'image/bytes': _bytes_feature(img_buffer),
        'label': _int64_feature(label_int)
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
# def encode_image(img):
#     '''
#     Encode image array as jpg prior to constructing Examples for TFRecords for compressed file size.
#     '''
#     img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)
#     return cv2.imencode('.jpg', img)[1]
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
def save_labels_int2text_tfrecords(labels):
    '''TBD: Save dict mapping of int2text labels in separate tfrecord to reduce size of records'''

##################################################################
##################################################################
##################################################################

def save_trainvaltest_tfrecords(dataset_name='PNAS',
                                output_dir = os.path.expanduser(r'~/data'),
                                target_size=(224,224),
                                num_channels=3,
                                low_count_threshold=10,
                                val_size=0.3,
                                test_size=0.3,
                                num_shards=10,
                                verbose=True):
    '''
    Load images from dataset_name, split into train, val, and test sets.
    Then iterate over these subsets and feed to function that shards data,
    then distributes data to process pool,
    where each process writes its shard to an individual TFRecord.

    '''
    data_splits, metadata_splits = load_and_format_dataset_from_db(dataset_name=dataset_name, low_count_threshold=low_count_threshold, val_size=val_size, test_size=test_size)

    os.makedirs(output_dir, exist_ok=True)
    file_logs = {}
#     file_log = {'label_map':metadata_splits['label_map']}
    for split_name, split_data in data_splits.items():
        num_samples = len(split_data['label'])
        num_classes = metadata_splits[split_name]['num_classes']
        print(f'Starting to split {split_name} subset with {num_samples} total samples into {num_shards} shards')
        
        coder = TFRecordCoder(split_data, output_dir, subset=split_name, target_size=target_size, num_channels=num_channels, num_shards=num_shards, num_classes=num_classes)
        coder.execute_convert()        
        coder.label_map = metadata_splits['label_map']
        file_logs.update({split_name:coder.filepath_log})
    
    coder.file_logs = file_logs
#         file_log.update({split_name:coder.filepath_log})
    return coder #file_log




def main(config=None):
    '''
    Example Jupyter notebook command:

        %run create_tfrecords.py -d Fossil -o /home/jacob/data -thresh 3 -val 0.3 -test 0.3

    '''
#     config=None
    if config is None:
        config = DatasetConfig() #dataset_name='PNAS')
    
    dataset_name = config.dataset_name
    target_size = config.target_size
    #TODO add ability to save TFRecords with configurable num_channels, currently just saves rgb
    num_channels = config.num_channels
    low_count_threshold = config.low_class_count_thresh
    val_size = config.data_splits['val_size']
    test_size = config.data_splits['test_size']
    tfrecord_root_dir = config.tfrecord_root_dir
    num_shards = config.num_shards
    
    output_dir = os.path.join(tfrecord_root_dir,dataset_name,f'num_channels-{num_channels}_thresh-{low_count_threshold}')

    file_log = check_if_tfrecords_exist(output_dir)

    print('filename_log = ', file_log)
#     if file_log == None:
    if True:
        file_log = {'train':[],'val':[],'test':[]}
        print('Entering save_trainvaltest_tfrecords')
        coder = save_trainvaltest_tfrecords(dataset_name=dataset_name,
                                                   output_dir=output_dir,
                                                   target_size=target_size,
                                                   num_channels=num_channels,
                                                   low_count_threshold=low_count_threshold,
                                                   val_size=val_size,
                                                   test_size=test_size,
                                                   num_shards=num_shards)
        label_map = coder.label_map

        for subset, records in coder.file_logs.items():
            file_log[subset] = [os.path.join(output_dir,subset,record_fname) for record_fname in sorted(records)]
            
        return coder, file_log
            
    
    else:
        print(f'Found {len(file_log.keys())} subsets of tfrecords already saved, skipping creation process.')

#     return file_log

def read():
    
    coder, file_log = main(DatasetConfig(dataset_name='Leaves'))#, target_size=(768,768))
    
#     tfrecord_paths = file_log['train']
#     tfrecord_paths = file_log['val']
    tfrecord_paths = file_log['test']
    
#     data = coder.read_tfrecords(tfrecord_paths)        
#     label_map = coder.label_map
    
    data = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
            .apply(lambda x: tf.data.TFRecordDataset(x)) \
            .map(self.decode_example,num_parallel_calls=AUTOTUNE) \
            .batch(batch_size,drop_remainder=drop_remainder) \
            .prefetch(AUTOTUNE)    

#     for imgs, labels in data.take(1):
#         labels = [coder.label_map[label] for label in labels.numpy()]
#         plot_image_grid(imgs, labels, 4, 8)

    for i, (imgs, labels) in enumerate(data):        
        print(i, imgs.shape, len(labels))
            
    return coder
            

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
