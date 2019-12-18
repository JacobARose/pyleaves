import dataset
import os
from sklearn.model_selection import train_test_split
from stuf import stuf
import tensorflow as tf
from tensorflow.keras import backend as K

from pyleaves.data_pipeline.preprocessing import generate_encoding_map, encode_labels, filter_low_count_labels, one_hot_encode_labels #, one_hot_decode_labels
from pyleaves import leavesdb
from pyleaves.utils import ensure_dir_exists

def reset_keras_session():
    '''
    Helper function for resetting Tensorflow session and default graph, mainly for scripts that involve multiple experiments.
    Likely could be simplified or scaled down, written to ensure everything is reset.
    '''
    K.clear_session()
    K.get_session().close()
    tf.reset_default_graph()

    tf_config=tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
    tf_config.gpu_options.allocator_type = 'BFC'
    tf_config.gpu_options.allow_growth = True
#     tf_config.allow_soft_placement = True
    sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)
    K.set_session(sess)


def train_val_test_split(image_paths, labels, test_size=0.3, val_size=0.3, random_seed=2376, verbose=True):

    train_paths, test_paths, train_labels, test_labels  = train_test_split(image_paths, labels, test_size=test_size, random_state=random_seed, shuffle=True, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=val_size, random_state=random_seed, shuffle=True, stratify=train_labels)

    if verbose:
        print(f'train samples: {len(train_labels)}')
        print(f'val samples: {len(val_labels)}')
        print(f'test samples: {len(test_labels)}')

    train_data = {'path': train_paths, 'label': train_labels}
    val_data = {'path': val_paths, 'label': val_labels}
    test_data = {'path': test_paths, 'label': test_labels}

    data_splits = {'train': train_data,
                  'val': val_data,
                  'test': test_data}
    return data_splits

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

    data_splits['label_map'] = generate_encoding_map(data_df, text_label_col='family', int_label_col='label')

    return data_splits


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
