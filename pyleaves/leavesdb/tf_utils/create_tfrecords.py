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
import numpy as np
import pandas as pd

# import contextlib2
# from object_detection.dataset_tools import tf_record_creation_util
# from sklearn.model_selection import train_test_split
from stuf import stuf
import sys
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.data.experimental import AUTOTUNE
# from pyleaves.data_pipeline.preprocessing import generate_encoding_map #encode_labels, filter_low_count_labels, one_hot_encode_labels, one_hot_decode_labels
from pyleaves.analysis.img_utils import load_image, TFRecordCoder
from pyleaves.config import DatasetConfig
from pyleaves import leavesdb
from pyleaves.leavesdb.tf_utils.tf_utils import (train_val_test_split,
                                                load_and_format_dataset_from_db,
                                                check_if_tfrecords_exist)
from pyleaves.utils import ensure_dir_exists, set_visible_gpus
from pyleaves.tests.test_utils import timeit, timelined_benchmark, draw_timeline, map_decorator

gpu_ids = [0]
set_visible_gpus(gpu_ids)

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

# def create_tfrecord_shard(shard_filepath,
#                           img_filepaths,
#                           labels,
#                           target_size = (224,224),
#                           verbose=True):
#     '''
#     Function for passing a list of image filpaths and labels to be saved in a single TFRecord file
#     located at shard_filepath.
#     '''
# #     writer = tf.python_io.TFRecordWriter(shard_filepath)
#     writer = tf.io.TFRecordWriter(shard_filepath)
#     img_filepaths = list(img_filepaths)
#     labels = list(labels)

#     num_samples = len(labels)
#     for i in range(num_samples):
#         path, label = img_filepaths[i], labels[i]
#         if verbose & (not i % 10):
#             print(img_filepaths[i],f'-> {i}/{num_samples} samples in shard',end='\r'); sys.stdout.flush()

#         example = load_and_encode_example(path,label,target_size)
#         if example is not None:
#             writer.write(example)
#     writer.close()

#     print('Finished saving TFRecord at: ', shard_filepath, '\n')

# def create_shard(data_packet):

#     shard_id, shard = data_packet['enumerated_shard']
#     output_dir = data_packet['output_dir']
#     output_base_name = data_packet['output_base_name']
#     target_size = data_packet['target_size']
#     num_shards = data_packet['num_shards']

#     print('SHARD_ID : ', shard_id)
#     shard_fname = f'{output_base_name}-{str(shard_id).zfill(5)}-of-{str(num_shards).zfill(5)}.tfrecord'
#     print('Creating shard : ', shard_fname)

#     shard_filepath = os.path.join(output_dir,shard_fname)
#     shard_img_filepaths, shard_labels = unzip(shard)
#     create_tfrecord_shard(shard_filepath, shard_img_filepaths, shard_labels, target_size = target_size) #, verbose=False)
#     return (shard_filepath, list(zip(shard_img_filepaths, shard_labels)))

# def multiprocess_create_tfrecord_shards(img_filepaths,
#                            labels,
#                            output_dir,
#                            output_base_name='train',
#                            target_size=(224,224),
#                            num_shards=10,
#                            verbose=True):

#     num_processes = os.cpu_count()//2
#     total_samples = len(labels)

#     zipped_data = zip(img_filepaths, labels)
#     sharded_data = chunked(zipped_data, total_samples//num_shards)
#     os.makedirs(output_dir, exist_ok=True)

#     sharded_data = list(sharded_data)
#     shard_ids = list(range(len(sharded_data)))

#     data_packets = []
#     for enumerated_shard in zip(shard_ids, sharded_data):
#         data_packets.append({'enumerated_shard':enumerated_shard,
#                              'output_dir':output_dir,
#                              'output_base_name':output_base_name,
#                              'target_size':target_size,
#                              'num_shards':num_shards})

#     result = list(map(create_shard, data_packets))

#     return os.listdir(output_dir), result

# def create_tfrecord_shards(img_filepaths,
#                            labels,
#                            output_dir,
#                            output_base_name='train',
#                            target_size=(224,224),
#                            num_shards=10):



#     total_samples = len(labels)

#     zipped_data = zip(img_filepaths, labels)
#     sharded_data = chunked(zipped_data, total_samples//num_shards)

#     os.makedirs(output_dir, exist_ok=True)
#     num_finished_samples = 0
#     for shard_i, shard in enumerate(sharded_data):
#         shard_fname = f'{output_base_name}-{str(shard_i).zfill(5)}-of-{str(num_shards).zfill(5)}.tfrecord'
#         shard_filepath = os.path.join(output_dir,shard_fname)

#         shard_img_filepaths, shard_labels = unzip(shard)

#         create_tfrecord_shard(shard_filepath, shard_img_filepaths, shard_labels, target_size = target_size, verbose=True)

#         num_finished_samples += len(list(shard))
# #         print('\n')
#         print(f'{output_base_name} - Finished: {num_finished_samples}/{total_samples} total samples, {shard_i+1}/{num_shards} total shards', end='\n')

#     return os.listdir(output_dir)
##################################################################
##################################################################


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """
    Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
        exit_stack: A context2.ExitStack used to automatically closed the TFRecords
            opened in this function.
        base_path: The base path for all shards
        num_shards: The number of shards
    Returns:
        The list of opened TFRecords. Position k in the list corresponds to shard k.
      """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
          for idx in range(num_shards)
                                  ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
                  ]

    return tfrecords
#         num_shards=10
#         output_filebase='/path/to/train_dataset.record'

#         with contextlib2.ExitStack() as tf_record_close_stack:
#             output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
#                                                                                       tf_record_close_stack,
#                                                                                       output_filebase,
#                                                                                       num_shards)
#             for index, example in examples:
#                 tf_example = create_tf_example(example)
#                 output_shard_index = index % num_shards
#                 output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
def create_shard(examples, shard_filepath):
    
    writer = tf.io.TFRecordWriter(shard_filepath)
    for i, example in enumerate(paths, labels):
        writer.write(example)
    return shard_filepath

def tf_data_create_tfrecord_shards(data_dict,
                           output_dir,
                           output_base_name='train',
                           target_size=(224,224),
                           num_shards=10,
                           verbose=True):

    data_splits, metadata_splits = load_and_format_dataset_from_db(dataset_name=dataset_name, low_count_threshold=low_count_threshold, val_size=val_size, test_size=test_size)

    os.makedirs(output_dir, exist_ok=True)
    filename_log = {'label_map':metadata_splits['label_map']}
    split_name='train';split_data=data_splits['train']
    data_dict=split_data
    output_dir=os.path.join(output_dir,split_name)
    output_base_name=split_name
    
    
    
    
    coder = TFRecordCoder(data_dict, output_dir, subset=output_base_name)
    img_read = coder.get_image_reader
    i=0
    staged_dataset = coder.stage_dataset(data_dict)
#     new_dataset_paths = coder.execute_conversion(coder.input_dataset)
#     img_filepaths = data_dict['path']
#     labels = data_dict['label']
    
    os.makedirs(output_dir, exist_ok=True)
    total_samples = len(data_dict['label'])
#     zipped_data = zip(img_filepaths, labels)
    shards = [];# i=0
    for i in range(num_shards):
        shard_filepath = coder.gen_shard_filepath(i, output_dir, output_base_name, num_shards)
        create_shard_i = partial(create_shard, shard_filepath=shard_filepath)
        
        
        shard = staged_dataset.shard(num_shards, i)
        shard = shard.map(coder.parse_img2example, num_parallel_calls=AUTOTUNE)
        shard = shard.prefetch(AUTOTUNE)
        shard = shard.apply(create_shard_i)
        shards.append(shard)
    return shard
#     sharded_data = chunked(zipped_data, total_samples//num_shards)


#     sharded_data = list(sharded_data)
#     shard_ids = list(range(len(sharded_data)))

#     data_packets = []
#     for enumerated_shard in zip(shard_ids, sharded_data):
#         shard_filepath = coder.gen_shard_filepath(enumerated_shard[0], output_dir, output_base_name, num_shards)
        
#        data_packets.append({'enumerated_shard':enumerated_shard,
#                              'output_dir':output_dir,
#                              'output_base_name':output_base_name,
#                              'target_size':target_size,
#                              'num_shards':num_shards})

#     result = list(map(create_shard, data_packets))

#     return os.listdir(output_dir), result


def save_trainvaltest_tfrecords(dataset_name='PNAS',
                                output_dir = os.path.expanduser(r'~/data'),
                                target_size=(224,224),
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
    filename_log = {'label_map':metadata_splits['label_map']}
    split_name='train';split_data=data_splits['train']
    data_dict=split_data
    output_dir=os.path.join(output_dir,split_name)
    output_base_name=split_name
    
    for split_name, split_data in data_splits.items():

        
        num_samples = len(split_data['label'])
        print(f'Starting to split {split_name} subset with {num_samples} total samples into {num_shards} shards')

        saved_tfrecord_files = tf_data_create_tfrecord_shards(
                                                      data_dict=split_data,
                                                      output_dir=os.path.join(output_dir,split_name),
                                                      output_base_name=split_name,
                                                      target_size=target_size,
                                                      num_shards=num_shards,
                                                      verbose=verbose)

        filename_log.update({split_name:saved_tfrecord_files})
    return filename_log



# def save_trainvaltest_tfrecords(dataset_name='PNAS',
#                                 output_dir = os.path.expanduser(r'~/data'),
#                                 target_size=(224,224),
#                                 low_count_threshold=10,
#                                 val_size=0.3,
#                                 test_size=0.3,
#                                 num_shards=10,
#                                 verbose=True):
#     '''
#     Load images from dataset_name, split into train, val, and test sets.
#     Then iterate over these subsets and feed to function that shards data,
#     then distributes data to process pool,
#     where each process writes its shard to an individual TFRecord.
#
#     '''
#     data_splits, metadata_splits = load_and_format_dataset_from_db(dataset_name=dataset_name, low_count_threshold=low_count_threshold, val_size=val_size, test_size=test_size)
#
#     os.makedirs(output_dir, exist_ok=True)
#     filename_log = {'label_map':metadata_splits['label_map']}
#     for split_name, split_data in data_splits.items():
#
#         split_filepaths = list(collapse(split_data['path']))
#         split_labels = split_data['label']
#         num_samples = len(split_labels)
#         print('Starting to split ',split_name, f' with {num_samples} total samples into {num_shards} shards')
#
#         saved_tfrecord_files, result = tf_data_create_tfrecord_shards(split_filepaths,
#                                                       split_labels,
#                                                       output_dir=os.path.join(output_dir,split_name),
#                                                       output_base_name=split_name,
#                                                       target_size=target_size,
#                                                       num_shards=num_shards,
#                                                       verbose=verbose)
#         # saved_tfrecord_files, result = multiprocess_create_tfrecord_shards(split_filepaths,
#         #                                               split_labels,
#         #                                               output_dir=os.path.join(output_dir,split_name),
#         #                                               output_base_name=split_name,
#         #                                               target_size=target_size,
#         #                                               num_shards=num_shards,
#         #                                               verbose=verbose)
#
#         filename_log.update({split_name:saved_tfrecord_files})
#     return filename_log

##################################################################
# def demo_save_tfrecords(dataset_name='PNAS',
#                         output_dir = os.path.expanduser(r'~/data'),
#                         target_size=(224,224),
#                         low_count_threshold=10,
#                         val_size=0.3,
#                         test_size=0.3,
#                         num_shards=10):

#     data_splits, metadata_splits = load_and_format_dataset_from_db(dataset_name=dataset_name, low_count_threshold=low_count_threshold, val_size=val_size, test_size=test_size)

#     os.makedirs(output_dir, exist_ok=True)
#     filename_log = {'label_map':metadata_splits['label_map']}
#     for split_name, split_data in data_splits.items():

#         split_filepaths = list(collapse(split_data['path']))
#         split_labels = split_data['label']

#         num_samples = len(split_labels)
#         print('Starting to split ',split_name, f' with {num_samples} total samples into {num_shards} shards')

#         saved_tfrecord_files = create_tfrecord_shards(split_filepaths,
#                                                       split_labels,
#                                                       output_dir=os.path.join(output_dir,split_name),
#                                                       output_base_name=split_name,
#                                                       target_size=target_size,
#                                                       num_shards=num_shards)

#         filename_log.update({split_name:saved_tfrecord_files})
#     return filename_log

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


def main(config=None):
    '''
    Example Jupyter notebook command:

        %run create_tfrecords.py -d Fossil -o /home/jacob/data -thresh 3 -val 0.3 -test 0.3

    '''
    config = DatasetConfig()
    
    dataset_name = config.dataset_name
    target_size = config.target_size
    low_class_count_thresh = config.low_class_count_thresh
    val_size = config.data_splits['val_size']
    test_size = config.data_splits['test_size']
    tfrecord_root_dir = config.tfrecord_root_dir
    num_shards = config.num_shards
    low_count_threshold=low_class_count_thresh
    output_dir = os.path.join(tfrecord_root_dir,dataset_name)

    print('pre filename log')
    filename_log = check_if_tfrecords_exist(output_dir)

    print('filename_log = ', filename_log)
    if filename_log == None:
        print('Entering save_trainvaltest_tfrecords')
        filename_log = save_trainvaltest_tfrecords(dataset_name=dataset_name,
                                                   output_dir=output_dir,
                                                   target_size=target_size,
                                                   low_count_threshold=low_class_count_thresh,
                                                   val_size=val_size,
                                                   test_size=test_size,
                                                   num_shards=num_shards)
        label_map = filename_log.pop('label_map', None)

        for key, records in filename_log.items():
            filename_log[key] = [os.path.join(output_dir,key,record_fname) for record_fname in sorted(records)]
    else:
        print(f'Found {len(filename_log.keys())} subsets of tfrecords already saved, skipping creation process.')

    return filename_log
#         if args.timeit == True:
#             label_map = load_and_format_dataset_from_db(dataset_name=dataset_name,
#                                                      low_count_threshold=low_count_threshold,
#                                                         val_size=val_size,
#                                                         test_size=test_size,
#                                                         verbose=False)['label_map']


            ##PLOT 1 BATCH OF IMAGES WITH TEXT LABELS


            # train_dataset = build_TFRecordDataset(sorted(filename_log['train']))
            # val_dataset = build_TFRecordDataset(sorted(filename_log['val']))
            # test_dataset = build_TFRecordDataset(sorted(filename_log['test']))
            # from pyleaves.analysis.img_utils import plot_image_grid
            # for imgs, labels in train_dataset.take(1):
            #     labels = [label_map[label] for label in labels.numpy()]
            #     plot_image_grid(imgs, labels, 4, 8)
            ##TEST ITERATION TIME

#             filenames = filename_log['train']
#             TFRecord_timeline = timeit(build_TFRecordDataset(filenames))
#             naive_TFRecord_timeline = timeit(build_naive_TFRecordDataset(filenames, batch_size=32))

    #     draw_timeline(TFRecord_timeline, "TFRecord", 15)


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
                           low_class_count_thresh=args.low_count_thresh,
                           data_splits={'val_size':args.val_size,'test_size':args.test_size},
                           tfrecord_root_dir=args.output_dir,
                           num_shards=args.num_shards)


    main(config)
