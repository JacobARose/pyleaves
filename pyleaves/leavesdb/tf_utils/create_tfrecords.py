'''
TBD

'''
import cv2
import dataset
from more_itertools import chunked, collapse, unzip
import os
from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, one_hot_encode_labels, one_hot_decode_labels
from pyleaves.analysis.img_utils import load_image
from pyleaves import leavesdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from stuf import stuf
import sys

import tensorflow as tf
tf.enable_eager_execution()


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

def encode_example(img, label_int):
    
    shape = img.shape
    
    features = {
        'image/height': _int64_feature(shape[0]),
        'image/width': _int64_feature(shape[1]),
        'image/channels': _int64_feature(shape[2]),
        'image/bytes': _bytes_feature(img.tostring()),
        'label': _int64_feature(int(label_int))
    }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def decode_example(serialized_example):
    return tf.train.Example.FromString(serialized_example)

def encode_image(img):
    '''
    Encode image array as jpg prior to constructing Examples for TFRecords for compressed file size.
    '''
    return cv2.imencode('.jpg', img)[1]

def _check_image(img):
    if type(img) != type(None):
        return True
    else:
        return False

def create_tfrecord_shard(shard_filepath, 
                          img_filepaths,
                          labels,
                          target_size = (224,224), 
                          verbose=True):
    '''
    Function for passing a list of image filpaths and labels to be saved in a single TFRecord file
    located at shard_filepath.
    '''
    writer = tf.python_io.TFRecordWriter(shard_filepath)
    
    img_filepaths = list(img_filepaths)
    labels = list(labels)
    
    num_samples = len(labels)
    
    failed_images = []
    for i in range(num_samples):
        
        if verbose & (not i % 10):
            print(img_filepaths[i],f'-> {i}/{num_samples}')
            sys.stdout.flush()

        img = load_image(img_filepaths[i], target_size=target_size)
        label = labels[i]
    
        if not _check_image(img):
            #Store path and label of images that returned None from load_image, cotninue to next example
            failed_images.append((img_filepaths[i],label))
            continue
            
        example = encode_example(img,label)
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
        shard_fname = f'{output_base_name}-{shard_i:.5d}-of-{num_shards:.5d}.tfrecord'
        shard_filepath = os.path.join(output_dir,shard_fname)

        shard_img_filepaths, shard_labels = unzip(shard)
        
        create_tfrecord_shard(shard_filepath, shard_img_filepaths, shard_labels, target_size = target_size, verbose=True)
        
        num_finished_samples += len(list(shard_labels))
        print(f'Finished: {num_finished_samples}/{total_samples} samples, {shard_i}/{num_shards} shards')
        
    return os.listdir(output_dir)
        
def train_val_test_split(image_paths, one_hot_labels, data_df, test_size=0.7, val_size=0.7, random_seed=2376):

    train_paths, test_paths, train_labels, test_labels  = train_test_split(image_paths, one_hot_labels, test_size=test_size, random_state=random_seed, shuffle=True, stratify=data_df['label'])
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=val_size, random_state=random_seed, shuffle=True, stratify=train_labels)


    train_data = {'path': train_paths, 'label': train_labels}
    val_data = {'path': val_paths, 'label': val_labels}
    test_data = {'path': test_paths, 'label': test_labels}

    data_splits = {'train': train_data,
                  'val': val_data,
                  'test': test_data}
    return data_splits
    
def load_and_format_dataset_from_db(dataset_name='PNAS', low_count_threshold=10, val_size=0.7, test_size=0.7):

    local_db = leavesdb.init_local_db()
    print(local_db)
    db = dataset.connect(f'sqlite:///{local_db}', row_type=stuf)
    data = leavesdb.db_query.load_data(db, dataset=dataset_name)
    
    data_df = encode_labels(data)
    
    data_df = filter_low_count_labels(data_df, threshold=low_count_threshold, verbose = True)
    data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)
    image_paths = data_df['path'].values.reshape((-1,1))
    labels = data_df['label'].values
#     one_hot_labels = one_hot_encode_labels(data_df['label'].values)
    data_splits = train_val_test_split(image_paths, labels, data_df, val_size=val_size, test_size=test_size)

    return data_splits

    
def demo_save_tfrecords(dataset_name='PNAS',
                        output_dir = os.path.expanduser(r'~/data'), 
                        target_size=(224,224),
                        low_count_threshold=10,
                        val_size=0.7, 
                        test_size=0.7,
                        num_shards=10):
    
    data_splits = load_and_format_dataset_from_db(dataset_name=dataset_name, low_count_threshold=low_count_threshold, val_size=val_size, test_size=test_size)
    
    os.makedirs(output_dir, exist_ok=True)
    filename_log = {} 
    for split_name, split_data in data_splits.items():
        
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


# filename_log = demo_save_tfrecords()


feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/bytes': tf.io.FixedLenFeature([], tf.string, default_value=0),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

def _parse_function(example_proto):
    example = tf.parse_single_example(example_proto, feature_description)
    img = example['image/bytes']
    return 

filenames = filename_log['train']

def build_TFRecordDataset(filenames):
#     dataset = tf.data.TFRecordDataset(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset - dataset.shuffle(buffer_size=len(filenames))
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=1,block_length=1)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


# def _parse_function(example_proto):
#   # Parse the input `tf.Example` proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_description)







# import numpy as np
# a = np.random.randn(2,2)
# b = _bytes_feature(tf.io.serialize_tensor(a))

# c = tf.io.parse_tensor(b,np.float64)

# print('a',a)
# print('b',b)
# print('c',c)

def main():
    
    filename_log = demo_save_tfrecords()
    
    train_dataset = build_TFRecordDataset(sorted(filename_log['train']))
    val_dataset = build_TFRecordDataset(filename_log['val'])
    test_dataset = build_TFRecordDataset(filename_log['test'])
    