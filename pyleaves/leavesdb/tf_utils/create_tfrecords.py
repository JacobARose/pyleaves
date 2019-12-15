'''
TBD

'''
import argparse
import cv2
# import dataset
from more_itertools import chunked, collapse, unzip
import os
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from stuf import stuf
import sys
import tensorflow as tf
# tf.enable_eager_execution()
i = 0
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
    print('ready')

if tf.executing_eagerly():
    i+=1
    print('executing eagerly: ',tf.executing_eagerly())
    print('i = ',i)
    print(__name__)

    
from tensorflow.data.experimental import AUTOTUNE
    
# from pyleaves.data_pipeline.preprocessing import generate_encoding_map #encode_labels, filter_low_count_labels, one_hot_encode_labels, one_hot_decode_labels
from pyleaves.analysis.img_utils import load_image
from pyleaves import leavesdb
from pyleaves.leavesdb.tf_utils.tf_utils import (train_val_test_split,
                                                load_and_format_dataset_from_db,
                                                check_if_tfrecords_exist)
from pyleaves.utils import ensure_dir_exists

from pyleaves.tests.test_utils import timeit, timelined_benchmark, draw_timeline, map_decorator

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
    img_buffer = encode_image(img)
    
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

#     height = tf.cast(features['image/height'], tf.int32)
#     width = tf.cast(features['image/width'], tf.int32)
#     channels = tf.cast(features['image/channels'], tf.int32)
    
    img = tf.image.decode_jpeg(features['image/bytes'], channels=3)#channels.eval(session=tf.Session()))
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    return img, label
##################################################################
def encode_image(img):
    '''
    Encode image array as jpg prior to constructing Examples for TFRecords for compressed file size.
    '''
    return cv2.imencode('.jpg', img)[1].tostring()

def decode_image(img_string, channels=3):
    return tf.io.decode_image(img_string,channels=channels)
##################################################################
def load_and_encode_example(path, label, target_size = (224,224)):
    img = load_image(path, target_size=target_size)  
    return encode_example(img,label)
##################################################################
def save_labels_int2text_tfrecords(labels):
    '''TBD: Save dict mapping of int2text labels in separate tfrecord to reduce size of records'''
    
def create_tfrecord_shard(shard_filepath, 
                          img_filepaths,
                          labels,
                          target_size = (224,224), 
                          verbose=True):
    '''
    Function for passing a list of image filpaths and labels to be saved in a single TFRecord file
    located at shard_filepath.
    '''
#     writer = tf.python_io.TFRecordWriter(shard_filepath)
    writer = tf.io.TFRecordWriter(shard_filepath)
    
    img_filepaths = list(img_filepaths)
    labels = list(labels)
    
    num_samples = len(labels)
    for i in range(num_samples):

        path, label = img_filepaths[i], labels[i]
        
        if verbose & (not i % 10):
            sys.stdout.flush()
            print(img_filepaths[i],f'-> {i}/{num_samples} samples in shard',end='\r')
            sys.stdout.flush()
            
        example = load_and_encode_example(path,label,target_size)
        if example is not None:
            writer.write(example)
    writer.close()
    
    print('Finished saving TFRecord at: ', shard_filepath)
    
def create_tfrecord_shards(img_filepaths,
                           labels,
                           output_dir,
                           output_base_name='train',
                           target_size=(224,224),
                           num_shards=10):
    
    
    
    total_samples = len(labels)
    
    zipped_data = zip(img_filepaths, labels)
    sharded_data = chunked(zipped_data, total_samples//num_shards)
        
    os.makedirs(output_dir, exist_ok=True)
    num_finished_samples = 0
    for shard_i, shard in enumerate(sharded_data):
        shard_fname = f'{output_base_name}-{str(shard_i).zfill(5)}-of-{str(num_shards).zfill(5)}.tfrecord'
        shard_filepath = os.path.join(output_dir,shard_fname)

        shard_img_filepaths, shard_labels = unzip(shard)
        
        create_tfrecord_shard(shard_filepath, shard_img_filepaths, shard_labels, target_size = target_size, verbose=True)
        
        num_finished_samples += len(list(shard))
        print(f'Finished: {num_finished_samples}/{total_samples} total samples, {shard_i}/{num_shards} total shards', end='\r')
        
    return os.listdir(output_dir)
##################################################################
def demo_save_tfrecords(dataset_name='PNAS',
                        output_dir = os.path.expanduser(r'~/data'), 
                        target_size=(224,224),
                        low_count_threshold=10,
                        val_size=0.3, 
                        test_size=0.3,
                        num_shards=10):
    
    data_splits = load_and_format_dataset_from_db(dataset_name=dataset_name, low_count_threshold=low_count_threshold, val_size=val_size, test_size=test_size)
    
    os.makedirs(output_dir, exist_ok=True)
    filename_log = {} 
    for split_name, split_data in data_splits.items():
        
        if split_name == 'label_map':
            filename_log[split_name] = split_data
            continue
        
        split_filepaths = list(collapse(split_data['path']))
        split_labels = split_data['label']
        
        num_samples = len(split_labels)
        print('Starting ',split_name, f' with {num_samples} samples')
        
        saved_tfrecord_files = create_tfrecord_shards(split_filepaths, 
                                                      split_labels,
                                                      output_dir=os.path.join(output_dir,split_name),
                                                      output_base_name=split_name,
                                                      target_size=target_size,
                                                      num_shards=num_shards)
        
        filename_log.update({split_name:saved_tfrecord_files})
    return filename_log

def preprocessing(img):
    '''TBD'''
    return img

def _parse_function(example_proto):
    img, label = decode_example(example_proto)
    return img, label


def build_TFRecordDataset(filenames, batch_size=32):
    dataset_generator_fun = lambda x: tf.data.TFRecordDataset(x)
    time_consuming_map = _parse_function

    optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
        .apply(dataset_generator_fun) \
        .map(time_consuming_map,num_parallel_calls=AUTOTUNE) \
        .repeat() \
        .batch(batch_size,drop_remainder=True) \
        .prefetch(AUTOTUNE) \

    return optimized_dataset

def build_naive_TFRecordDataset(filenames, batch_size=32):
    dataset_generator_fun = lambda x: tf.data.TFRecordDataset(x)
    naive_map = _parse_function
    
    naive_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
                    .flat_map(dataset_generator_fun) \
                    .map(naive_map) \
                    .repeat() \
                    .batch(batch_size, drop_remainder=True)
    return naive_dataset    
        
#     optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
#         .interleave(dataset_generator_fun,num_parallel_calls=AUTOTUNE) \
#         .map(time_consuming_map,num_parallel_calls=AUTOTUNE) \
#         .batch(_batch_map_num_items,drop_remainder=True) \
#         .prefetch(AUTOTUNE) \
# #         .unbatch()
#     return optimized_dataset


def main():

    '''
    Example Jupyter notebook command:
        
        %run create_tfrecords.py -d Fossil -o /home/jacob/data -thresh 3 -val 0.3 -test 0.3
    
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', default='PNAS', type=str,help='Name of dataset of images to use for creating TFRecords')
    parser.add_argument('-o', '--output_dir', default=r'/media/data/jacob', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    parser.add_argument('-thresh', '--low_count_threshold', default=10, type=int, help='Min population of a class below which we will exclude the class entirely.')
    parser.add_argument('-val', '--val_size', default=0.3, type=float, help='Fraction of train to use as validation set. Calculated after splitting train and test')
    parser.add_argument('-test', '--test_size', default=0.3, type=float, help='Fraction of full dataset to use as test set. Remaining fraction will be split into train/val sets.')
    args = parser.parse_args()    
    
    
    
    dataset_name = args.dataset_name #'PNAS'
    output_dir = os.path.join(args.output_dir,dataset_name)
#     output_dir = f'/home/jacob/data/{dataset_name}'
    
    low_count_threshold = args.low_count_threshold
    val_size = args.val_size
    test_size = args.test_size
    
    filename_log = check_if_tfrecords_exist(output_dir)
    
    if filename_log == None:
        filename_log = demo_save_tfrecords(dataset_name=dataset_name,
                                           output_dir=output_dir)
        label_map = filename_log.pop('label_map', None)
        
        for key, records in filename_log.items():
            filename_log[key] = [os.path.join(output_dir,key,record_fname) for record_fname in sorted(records)]
    else:
        print(f'Found {len(filename_log.keys())} subsets of tfrecords already saved, skipping creation process.')
        label_map = load_and_format_dataset_from_db(dataset_name=dataset_name,
                                                    low_count_threshold=low_count_threshold, 
                                                    val_size=val_size,
                                                    test_size=test_size,
                                                    verbose=False)['label_map']
        
    train_dataset = build_TFRecordDataset(sorted(filename_log['train']))
#     val_dataset = build_TFRecordDataset(sorted(filename_log['val']))
#     test_dataset = build_TFRecordDataset(sorted(filename_log['test']))
    
    
    ##PLOT 1 BATCH OF IMAGES WITH TEXT LABELS
    
    from pyleaves.analysis.img_utils import plot_image_grid
    
    for imgs, labels in train_dataset.take(1):
        labels = [label_map[label] for label in labels.numpy()]
        plot_image_grid(imgs, labels, 4, 8)
        
        
    ##TEST ITERATION TIME
    
    filenames = filename_log['train']
    TFRecord_timeline = timeit(build_TFRecordDataset(filenames))
    
    
    naive_TFRecord_timeline = timeit(build_naive_TFRecordDataset(filenames, batch_size=32))
    
#     draw_timeline(TFRecord_timeline, "TFRecord", 15)


if __name__ == "__main__":
    
    '''
    Instructions:
    
    Open a jupyter console and type "%run create_tfrecords.py"
    '''
    
    main()