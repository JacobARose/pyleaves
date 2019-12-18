
import os
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE

from pyleaves.leavesdb.tf_utils.create_tfrecords import decode_example
from pyleaves.utils import ensure_dir_exists

def _parse_function(example_proto):
    img, label = decode_example(example_proto)
    return img, label


def build_train_dataset(filenames, batch_size=32, buffer_size=1000, seed=17):
    dataset_generator_fun = lambda x: tf.data.TFRecordDataset(x)

    optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
        .apply(dataset_generator_fun) \
        .map(_parse_function,num_parallel_calls=AUTOTUNE) \
        .shuffle(buffer_size=buffer_size, seed=seed) \
        .repeat() \
        .batch(batch_size,drop_remainder=False) \
        .prefetch(AUTOTUNE) \

    return optimized_dataset


def build_test_dataset(filenames, batch_size=32, num_parallel_calls=None):
    dataset_generator_fun = lambda x: tf.data.TFRecordDataset(x)

    optimized_dataset = tf.data.Dataset.from_tensor_slices(filenames) \
        .apply(dataset_generator_fun) \
        .map(_parse_function,num_parallel_calls=num_parallel_calls) \
        .repeat() \
        .batch(batch_size,drop_remainder=False) \
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
                 batch_size=32,
                 shuffle_buffer_size=1000,
                 num_parallel_calls_test=None,
                 name=None,
                 seed=17):

        self.root_dir = root_dir
        self.subset = subset
        self.batch_size = batch_size
        self.buffer_size = shuffle_buffer_size

        self.name = name
        self.seed = seed

    def list_files(self, root_dir):
        '''
        Arguments:
            root_dir : path to directory containing TFRecord shards
        Return:
            self.file_list : Sorted list of TFRecord files contained in flat directory root_dir
        '''

        assert ensure_dir_exists(root_dir)
        file_list = sorted([filename for filename in os.listdir(root_dir) if '.tfrecord' in filename])

        return file_list

    def collect_subsets(self, root_dir):
        subsets = {}

        subset_dirs = os.listdir(root_dir)
        for subset in subset_dirs:
            subset_dir = os.path.join(root_dir,subset)
            if os.path.isdir(subset_dir):
                subsets[subset] = self.list_files(subset_dir)

        self.subsets = subsets
        return self.subsets

    def get_dataset(self, subset=None):
        if subset is None:
            subset = self.subset
        if subset == 'train':
            return build_train_dataset(filenames=self.subsets[subset],
                                       batch_size=self.batch_size,
                                       buffer_size=self.buffer_size,
                                       seed=self.seed)
        elif subset == 'test' or subset == 'val':
            return build_test_dataset(filenames=self.subsets[subset],
                                      batch_size=self.batch_size,
                                      num_parallel_calls=self.num_parallel_calls)

        else:
            print('Subset type not recognized, returning None.')
            return None
