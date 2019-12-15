'''
TBD

'''
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
    
# from pyleaves.data_pipeline.preprocessing import encode_labels, filter_low_count_labels, one_hot_encode_labels, one_hot_decode_labels
from pyleaves.analysis.img_utils import load_image
from pyleaves import leavesdb
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, load_and_format_dataset_from_db
from pyleaves.utils import ensure_dir_exists

from pyleaves.tests.test_utils import timelined_benchmark, draw_timeline, map_decorator

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
    for i in range(num_samples):

        path, label = img_filepaths[i], labels[i]
        
        if verbose & (not i % 50):
            print(img_filepaths[i],f'-> {i}/{num_samples}',end='\r')
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
        print(f'Finished: {num_finished_samples}/{total_samples} samples, {shard_i}/{num_shards} shards')
        
    return os.listdir(output_dir)
##################################################################

# def train_val_test_split(image_paths, one_hot_labels, data_df, test_size=0.3, val_size=0.3, random_seed=2376):

#     train_paths, test_paths, train_labels, test_labels  = train_test_split(image_paths, one_hot_labels, test_size=test_size, random_state=random_seed, shuffle=True, stratify=data_df['label'])
#     train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=val_size, random_state=random_seed, shuffle=True, stratify=train_labels)

    
#     print(f'train samples: {len(train_labels)}')
#     print(f'val samples: {len(val_labels)}')
#     print(f'test samples: {len(test_labels)}')

#     train_data = {'path': train_paths, 'label': train_labels}
#     val_data = {'path': val_paths, 'label': val_labels}
#     test_data = {'path': test_paths, 'label': test_labels}

#     data_splits = {'train': train_data,
#                   'val': val_data,
#                   'test': test_data}
#     return data_splits
    

# def load_and_format_dataset_from_db(dataset_name='PNAS', low_count_threshold=10, val_size=0.3, test_size=0.3):

#     local_db = leavesdb.init_local_db()
#     print(local_db)
#     db = dataset.connect(f'sqlite:///{local_db}', row_type=stuf)
#     data = leavesdb.db_query.load_data(db, dataset=dataset_name)
    
#     data_df = encode_labels(data)
    
#     data_df = filter_low_count_labels(data_df, threshold=low_count_threshold, verbose = True)
#     data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)
#     image_paths = data_df['path'].values.reshape((-1,1))
#     labels = data_df['label'].values
# #     one_hot_labels = one_hot_encode_labels(data_df['label'].values)
#     data_splits = train_val_test_split(image_paths, labels, data_df, val_size=val_size, test_size=test_size)

#     return data_splits

    
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

# def map_decorator(func):
#     '''
#     Use wrappers for mapped function
#     - To run mapped function in an eager context, you have to wrap them inside a tf.py_function call.
#         '''
#     def wrapper(steps, times, values):
#         # Use a tf.py_function to prevent auto-graph from compiling the method
#         return tf.py_function(
#             func,
#             inp=(steps, times, values),
#             Tout=(steps.dtype, times.dtype, values.dtype)
#         )
#     return wrapper



def _parse_function(example_proto):
    img, label = decode_example(example_proto)
    return img, label


def test_load_and_encode_example():
    dummy_sample = {'path':r'/media/data_cifs/sven2/leaves/sorted/Fossils_DataSource/New_Fossil_Dataset/I. Approved families/Adoxaceae/Sambucus newtoni/CU_0141 Sambucus newtoni.tif',
                   'label':0}
    serialized_example = load_and_encode_example(**dummy_sample)
    img, label = decode_example(serialized_example)
    return img, label


def check_if_tfrecords_exist(output_dir):
    '''if tfrecords already exist, return dictionary with mappings to their paths. Otherwise return None.'''    
    tfrecords = None
    if not ensure_dir_exists(output_dir):
        return tfrecords
    
    subset_dirs = os.listdir(output_dir)
    if len(subset_dirs) > 0:
        tfrecords = {}
        for subset in subset_dirs:
            subset_path = os.path.join(output_dir,
                                      subset)
            subset_filenames = os.listdir(subset_path)
            tfrecords[subset] = sorted([os.path.join(subset_path,filename) for filename in subset_filenames])
    return tfrecords


# def build_TFRecordDataset(filenames):

#     tfrecord_files = tf.data.Dataset.from_tensor_slices(filenames)
#     tfrecord_files = tfrecord_files.shuffle(buffer_size=len(filenames))
#     dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=4)
# #     dataset = dataset.shuffle(buffer_size=len(filenames))
#     dataset = dataset.map(_parse_function, num_parallel_calls=16)
# #     dataset = dataset.interleave(lambda x: tf.Print(x,[x]))
#     return dataset


def build_TFRecordDataset(filenames):
    dataset_generator_fun = lambda x: tf.data.TFRecordDataset(x)
    _batch_map_num_items = 1#2
    time_consuming_map = _parse_function
    AUTOTUNE = 1
    optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
        .apply(dataset_generator_fun) \
        .map(time_consuming_map,num_parallel_calls=AUTOTUNE) \
        .batch(_batch_map_num_items,drop_remainder=True) \
        .prefetch(AUTOTUNE) \
#         .unbatch()
    return optimized_dataset


#     optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
#         .interleave(dataset_generator_fun,num_parallel_calls=AUTOTUNE) \
#         .map(time_consuming_map,num_parallel_calls=AUTOTUNE) \
#         .batch(_batch_map_num_items,drop_remainder=True) \
#         .prefetch(AUTOTUNE) \
# #         .unbatch()
#     return optimized_dataset





#     dataset = dataset.interleave(lambda x:
#                                 tf.data.TFRecordDataset(x))#.map(
# #                                     _parse_function, num_parallel_calls=1))#,
# #                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.map(_parse_function, num_parallel_calls=1)
#     dataset = dataset.repeat()
#     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    
#     dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=1,block_length=1)
#     dataset = dataset.map(_parse_function, num_parallel_calls=1)#tf.data.experimental.AUTOTUNE)
#     return dataset

def main():
    
    dataset_name = 'PNAS'
#     output_dir = f'/media/data/jacob/{dataset_name}'
    output_dir = f'/home/jacob/data/{dataset_name}'
    
    filename_log = check_if_tfrecords_exist(output_dir)
    
    if filename_log == None:
        filename_log = demo_save_tfrecords(dataset_name=dataset_name,
                                           output_dir=output_dir)
        for key, records in filename_log.items():
            filename_log[key] = [os.path.join(output_dir,key,record_fname) for record_fname in sorted(records)]
    else:
        print(f'Found {len(filename_log.keys())} subsets of tfrecords already saved, skipping creation process.')
        
#     train_dataset = build_TFRecordDataset(sorted(filename_log['train']))
#     val_dataset = build_TFRecordDataset(sorted(filename_log['val']))
#     test_dataset = build_TFRecordDataset(sorted(filename_log['test']))
    
#     for example in train_dataset.take(1):
#         img, label = example
#         print(img.numpy().shape, label.numpy())
        
    
#     import matplotlib.pyplot as plt
#     for img, label in train_dataset.take(1):        
#         plt.imshow(img)
#         plt.title(label.numpy())
        
        
        

    
    
    filenames = filename_log['train']    
    TFRecord_timeline = timelined_benchmark(build_TFRecordDataset(filenames))
    
    
    draw_timeline(TFRecord_timeline, "TFRecord", 15)