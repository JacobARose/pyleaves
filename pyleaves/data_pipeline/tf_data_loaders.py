
from functools import partial
import os
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE

from pyleaves.leavesdb.tf_utils.create_tfrecords import decode_example
from pyleaves.utils import ensure_dir_exists

def _parse_function(example_proto, num_classes):
    img, label = decode_example(example_proto)
    label = tf.one_hot(label, num_classes)
    return img, label


def build_train_dataset(filenames, num_classes=None, batch_size=32, buffer_size=1000, seed=17, drop_remainder=True):
    def _parse_function(example_proto, num_classes):
        img, label = decode_example(example_proto)
        label = tf.one_hot(label, num_classes)
        return img, label
    
    dataset_generator_fun = lambda x: tf.data.TFRecordDataset(x)
    __parse_function = partial(_parse_function, num_classes=num_classes)
    
    optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
        .apply(dataset_generator_fun) \
        .map(__parse_function,num_parallel_calls=AUTOTUNE) \
        .shuffle(buffer_size=buffer_size, seed=seed) \
        .repeat() \
        .batch(batch_size,drop_remainder=drop_remainder) \
        .prefetch(AUTOTUNE) \

    return optimized_dataset


def build_test_dataset(filenames, num_classes=None, batch_size=32, num_parallel_calls=None, drop_remainder=True):
    def _parse_function(example_proto, num_classes):
        img, label = decode_example(example_proto)
        label = tf.one_hot(label, num_classes)
        return img, label
    
    dataset_generator_fun = lambda x: tf.data.TFRecordDataset(x)
    __parse_function = partial(_parse_function, num_classes=num_classes)

    optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
        .apply(dataset_generator_fun) \
        .map(__parse_function,num_parallel_calls=num_parallel_calls) \
        .repeat() \
        .batch(batch_size,drop_remainder=drop_remainder) \
        .prefetch(AUTOTUNE) \

    return optimized_dataset

# .apply(dataset_generator_fun) \
# .interleave(dataset_generator_fun, num_parallel_calls=AUTOTUNE)


class DatasetBuilder:
    '''
    Class for implementing a repeatable configuration for preparing tf.data.Datasets

    Currently only compatible with TFRecords.

    train_data = DatasetBuilder()
    '''
    def __init__(self,
                 root_dir,
                 subset='train',
                 num_classes=None,
                 batch_size=32,
                 shuffle_buffer_size=1000,
                 num_parallel_calls_test=None,
                 name=None,
                 seed=17):

        self.root_dir = root_dir
        self.subset = subset
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.buffer_size = shuffle_buffer_size
        self.num_parallel_calls_test = num_parallel_calls_test

        self.name = name
        self.seed = seed

    def list_files(self, records_dir):
        '''
        Arguments:
            records_dir : path to flat directory containing TFRecord shards, usually one level below root_dir and used to indicate 1 specific data split (e.g. train, val, or test)
        Return:
            file_list : Sorted list of TFRecord files contained in flat directory root_dir
        '''

        assert ensure_dir_exists(records_dir)
        file_list = sorted([os.path.join(records_dir,filename) for filename in os.listdir(records_dir) if '.tfrecord' in filename])

        return file_list

    def collect_subsets(self, root_dir):
        '''
        Search root_dir for subdirs corresponding to each data split, where each subdir is a flat directory containing tfrecord shards
        '''
        subsets = {}

        subset_dirs = os.listdir(root_dir)
        for subset in subset_dirs:
            subset_dir = os.path.join(root_dir,subset)
            if os.path.isdir(subset_dir):
                subsets[subset] = self.list_files(subset_dir)

        self.subsets = subsets
        return self.subsets

    def get_dataset(self, subset=None, batch_size=None, num_classes=None):
        if subset is None:
            subset = self.subset
        if batch_size is None:
            batch_size = self.batch_size
        if num_classes is None:
            num_classes = self.num_classes
            
            
        if subset == 'train':
            return build_train_dataset(filenames=self.subsets[subset],
                                       num_classes=num_classes,
                                       batch_size=batch_size,
                                       buffer_size=self.buffer_size,
                                       seed=self.seed)
        elif subset == 'test' or subset == 'val':
            return build_test_dataset(filenames=self.subsets[subset],
                                      num_classes=num_classes,
                                      batch_size=batch_size,
                                      num_parallel_calls=self.num_parallel_calls_test)

        else:
            print('Subset type not recognized, returning None.')
            return None

        
        