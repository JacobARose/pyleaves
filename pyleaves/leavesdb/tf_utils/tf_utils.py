# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: tf_utils.py


import dataset
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from stuf import stuf
import tensorflow as tf
from tensorflow.keras import backend as K

from pyleaves.data_pipeline.preprocessing import generate_encoding_map, encode_labels, filter_low_count_labels, one_hot_encode_labels
from pyleaves import leavesdb
from pyleaves.utils import ensure_dir_exists

def set_random_seed(seed_value=12321):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__[0] == '2':
        tf.random.set_seed(seed_value)
    else:
        tf.set_random_seed(seed_value)

    print(f'Set seeds [PYTHONHASHSEED, random, np.random, tf.set_random_seed] = {seed_value}')
    return seed_value

def reset_eager_session():
    '''
    Version for eager execution without explicit tf.Session()
    Helper function for resetting Tensorflow session and default graph, mainly for scripts that involve multiple experiments.
    Likely could be simplified or scaled down, written to ensure everything is reset.
    '''
    tf.compat.v1.reset_default_graph()
#     tf.reset_default_graph()
    K.clear_session()


def reset_keras_session():
    '''
    Helper function for resetting Tensorflow session and default graph, mainly for scripts that involve multiple experiments.
    Likely could be simplified or scaled down, written to ensure everything is reset.
    '''
    tf_config=tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
    tf_config.gpu_options.allocator_type = 'BFC'
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)
    K.set_session(sess)


#
#     K.clear_session()
#     K.get_session().close()
#     tf.reset_default_graph()
#
#     tf_config=tf.ConfigProto(log_device_placement=True)
#     tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
#     tf_config.gpu_options.allocator_type = 'BFC'
#     tf_config.gpu_options.allow_growth = True
# #     tf_config.allow_soft_placement = True
#     sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)
#     K.set_session(sess)


def train_val_test_split(x, y, test_size=0.3, val_size=0.3, random_seed=2376, verbose=True):
    #TO DO Refactor in order to get arbitrary k splits
    train_x, train_y = x, y
    val_x, test_x = [],[]
    val_y, test_y = [],[]

    if test_size>0.0:
        train_x, test_x, train_y, test_y  = train_test_split(x, y, test_size=test_size, random_state=random_seed, shuffle=True, stratify=y)
    if val_size>0.0:
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_size, random_state=random_seed, shuffle=True, stratify=train_y)

    data_splits = {
                    'train':
                            {'path': train_x,
                             'label': train_y},
                    'val':
                            {'path': val_x,
                             'label': val_y},
                    'test':
                            {'path': test_x,
                             'label': test_y}
                  }

    return data_splits


def get_data_splits_metadata(data_splits, data_df=None, encoder=None, class_mode='encoder', int_label_col='label', verbose=True):
    '''

    Must provide either data_df or an encoder

    class_mode, str: {'encoder', max', 'min'}
        if 'encoder':
            Set num_classes equal to the length of the provided encoder
        if 'max':
            Set num_classes equal to the total number of unique classes expected to be seen in all splits
        if 'min':
            Set num_classes equal to the mininum per split, allowing different values between splits
            WARNING: Must account for this when performing one_hot_encoding and building models.

    '''


    metadata_splits = {}
    if (encoder is None) and (data_df is not None):
        metadata_splits['label_map'] = generate_encoding_map(data_df, text_label_col='family', int_label_col=int_label_col)
    else:
        metadata_splits['label_map'] = encoder.get_encodings()

    if class_mode=='encoder':
        if encoder is None:
            class_mode='max'
        else:
            num_classes=len(encoder)

    if class_mode == 'max':
        num_classes=0
        for subset, data in data_splits.items():
            nclasses = len(np.unique(data_splits[subset][int_label_col]))
            num_classes = max(num_classes, nclasses)

    for subset, data in data_splits.items():
        if class_mode == 'min':
            num_classes = len(np.unique(data_splits[subset][int_label_col]))

        metadata_splits.update({
                                subset:{
                                        'num_samples':len(data_splits[subset][int_label_col]),
                                        'num_classes':num_classes
                                       }
                               }
                              )
        if verbose:
            print(f'{subset} : {metadata_splits[subset]}')

    return metadata_splits


def load_from_db(dataset_name='PNAS'):
    local_db = leavesdb.init_local_db()
    print(local_db)
    db = dataset.connect(f'sqlite:///{local_db}', row_type=stuf)
    data = leavesdb.db_query.load_data(db, dataset=dataset_name)
    return data

def load_and_format_dataset_from_db(dataset_name='PNAS', low_count_threshold=10, val_size=0.3, test_size=0.3, verbose=True):

    data = load_from_db(dataset_name=dataset_name)
    data_df = encode_labels(data)
    data_df = filter_low_count_labels(data_df, threshold=low_count_threshold, verbose = verbose)
    data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)
    image_paths = data_df['path'].values.reshape((-1,1))
    labels = data_df['label'].values
#     one_hot_labels = one_hot_encode_labels(data_df['label'].values)
    data_splits = train_val_test_split(image_paths, labels, val_size=val_size, test_size=test_size, verbose=verbose)

    metadata_splits = get_data_splits_metadata(data_splits, data_df, verbose=True)

    return data_splits, metadata_splits


def check_if_tfrecords_exist(output_dir):
    '''if tfrecords already exist, return dictionary with mappings to their paths. Otherwise return None.'''
    tfrecords = None
    ensure_dir_exists(output_dir)

    subset_dirs = os.listdir(output_dir)
    if len(subset_dirs) > 0:
        tfrecords = {}
        for subset in subset_dirs:
            subset_path = os.path.join(output_dir,
                                      subset)
            subset_filenames = os.listdir(subset_path)
            if len(subset_filenames)==0:
                return None
            tfrecords[subset] = sorted([os.path.join(subset_path,filename) for filename in subset_filenames])
    return tfrecords



def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        value = value.tostring()
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    """Returns a float_list from a float / double."""

    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
