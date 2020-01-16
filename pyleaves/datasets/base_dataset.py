'''
Script for defining base class BaseDataset for managing information about a particular subset or collection of datasets during preparation for a particular experiment.

'''

import cv2
import dataset
from itertools import starmap
from more_itertools import chunked, collapse, unzip
from functools import partial
import os
from stuf import stuf
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import dummy

import tensorflow as tf

from pyleaves.analysis.img_utils import load_image
from pyleaves.config import Config
from pyleaves.data_pipeline.tf_data_loaders import DatasetBuilder
from pyleaves import leavesdb
from pyleaves.leavesdb.db_query import load_from_db
from pyleaves.leavesdb.tf_utils.tf_utils import bytes_feature, float_feature, int64_feature, train_val_test_split, get_data_splits_metadata
from pyleaves.data_pipeline.preprocessing import generate_encoding_map, encode_labels, filter_low_count_labels, one_hot_encode_labels
from pyleaves.utils import ensure_dir_exists


def encode_image(img):
    '''
    JPEG COMPRESS BEFORE SAVING IN TFRECORD
    Encode image array as jpg prior to constructing Examples for TFRecords for compressed file size.
    '''
    return cv2.imencode('.jpg', img)[1].tostring()




class BaseDataset:
    
    def __init__(self,
                 name = 'Fossil', 
                 img_size = (224,224),
                 loss_function = 'categorical_crossentropy',
                 preprocess = [None],
                 batch_size = 64,
                 low_count_threshold=0,
                 local_db=None,
                 verbose=False):
        self.name = name
        self.img_size = img_size
        self.loss_function = loss_function
        self.preprocess = preprocess # preprocessing before conversion to TFRecords
        self.config = Config(dataset_name=name)
        self.seed = self.config.seed
        self.dataset_root_dir = os.path.join(self.config.tfrecords, self.name)
        ensure_dir_exists(self.dataset_root_dir)
        self.batch_size = batch_size
        self.low_count_threshold = low_count_threshold
        self.verbose= verbose
        
        if local_db == None:
            self.local_db = os.path.abspath(os.path.join('..','leavesdb','resources','leavesdb.db'))
        else:
            self.local_db = local_db
            
#         self.dataset_builder = DatasetBuilder(root_dir=self.dataset_root_dir,
#                                               num_classes=self.num_classes,
#                                               batch_size=self.batch_size,
#                                               seed=self.seed)
        
        self.feature_encoders = {
        'image/height': int64_feature,
        'image/width': int64_feature,
        'image/channels': int64_feature,
        'image/bytes': bytes_feature,
        'label': int64_feature
                                }
        self.feature_decoders = {
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/bytes': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
                                }


    def extract(self, name):
        '''
        Query filenames and labels from SQLiteDb
        '''
        self.db = dataset.connect(f'sqlite:///{self.local_db}', row_type=stuf)
        data = leavesdb.db_query.load_data(self.db, dataset=name)
        return data
        
    def encode(self, data, low_count_threshold):
        data_df = encode_labels(data)
        data_df = filter_low_count_labels(data_df, threshold=low_count_threshold, verbose=self.verbose)
        data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)
        self.data_df = data_df
        image_paths = data_df['path'].values.reshape((-1,1))
        labels = data_df['label'].values
        
        return image_paths, labels
    
    def create_splits(self, image_paths, labels, val_size, test_size):
        data_splits = train_val_test_split(image_paths, labels, val_size=val_size, test_size=test_size)
        metadata_splits = get_data_splits_metadata(data_splits, self.data_df)
        return data_splits, metadata_splits
        
    def load(self, name, low_count_threshold, val_size, test_size):
        
        data = self.extract(name)
        image_paths, labels = self.encode(data, low_count_threshold)
        data_splits, metadata_splits = self.create_splits(image_paths, labels, val_size, test_size)
        return data_splits, metadata_splits
    
    
class BaseTFRecordDataset(BaseDataset):
    
    def __init__(self,
                 name = 'Fossil', 
                 img_size = (224,224),
                 loss_function = 'categorical_crossentropy',
                 preprocess = [None],
                 batch_size = 64,
                 low_count_threshold = 3,
                 val_size = 0.2, 
                 test_size = 0.2,
                 num_shards = 10,
                 verbose=False):
        super().__init__(name = name, 
                         img_size = img_size,
                         loss_function = loss_function,
                         preprocess = preprocess,
                         batch_size = batch_size,
                         low_count_threshold = low_count_threshold,
                         verbose=verbose)
        self.val_size = val_size
        self.test_size = test_size
        self.num_shards = num_shards
        
        self.load_image = partial(load_image, target_size=self.img_size)
        
#         self.data_splits, self.metadata_splits = self.load(name, low_count_threshold, val_size, test_size)
        
    def shard_data(self, file_paths, labels, num_shards, all_equal=True):
        '''
        all_equal, bool : default = True
            If True:
                discard the remaining samples that don't fit neatly into the desired set of num_shards shards. Ensures total number of samples divides evenly by num_shards.
        '''
        if all_equal:
            num_samples = len(labels)
            remainder = num_samples%num_shards
            if remainder > 0:
                file_paths = file_paths[:-remainder]
                labels = labels[:-remainder]
                print(f'Removing last {remainder} samples before sharding')
        num_samples = len(labels)
        
        zipped_data = zip(file_paths, labels)
        sharded_data = chunked(zipped_data, num_samples//num_shards)
#         shard_dict = {shard_i : shard_data for shard_i, shard_data in enumerate(sharded_data)}
#         return shard_dict
        return sharded_data
    
    ##################################################################
    def encode_example(self, img_w_int_label):
        '''
        Arguments:
            img_w_int_label, tuple(np.ndarray, int) :
                Contains input image x with integer label y for args = ((x, y),)
                
        Return:
            Serialized_example_proto, string or something :
                example ready to be written to TFRecord
        '''
        img, label = img_w_int_label
        shape = img.shape
        img_buffer = encode_image(img)

        features = {
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/bytes': bytes_feature(img_buffer),
            'label': int64_feature(label)
        }
 
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()
    
    def load_example(self, sample):
        img_filepath, label, sample_id = sample
        
        print(f'Loading img # : {sample_id}')
        print(f'current thread : {dummy.threading.get_ident()}, total # : {dummy.threading.active_count()}')
        
        img = self.load_image(img_filepath)
        return self.encode_example((img, label))
        
    def create_tfrecord_shard(self, 
                              shard_filepath,
                              shard,
                              shard_id=0,
                              target_size = (224,224),
                              verbose=True):
        '''
        Function for passing a list of image filpaths and labels to be saved in a single TFRecord file
        located at shard_filepath.
        '''
        
        img_filepaths, labels = shard
        
        load_example = self.load_example
        writer = tf.io.TFRecordWriter(shard_filepath)
        img_filepaths = list(img_filepaths)
        labels = list(labels)
        num_samples = len(labels)
        
        sample_ids = list(range(shard_id*num_samples,(shard_id+1)*num_samples))
        samples = list(zip(img_filepaths, labels, sample_ids))

        thread_start_time = time.process_time()
        with ThreadPool(os.cpu_count()//4) as pool:
#         with Pool(os.cpu_count()) as pool:
            loaded_samples = pool.map(load_example, samples, chunksize=64)
#             loaded_samples = pool.apply(load_example, samples)
        thread_end_time = time.process_time()
        thread_time = thread_end_time-thread_start_time
        print(f'{len(samples)} samples loaded in {thread_time:.4f} sec : {len(samples)/thread_time:.2f} samples/sec')
            
        print(f'Loaded samples for shard {shard_filepath}')
        print(f'Writing to {len(labels)} shard')
        for sample in loaded_samples:
            writer.write(sample)
        writer.close()

        print('Finished saving TFRecord at: ', shard_filepath, '\n')
        
    def export_tfrecords(self):
        output_dir = self.dataset_root_dir
        num_shards = self.num_shards
        self.data_splits, self.metadata_splits = self.load(name=self.name, low_count_threshold=self.low_count_threshold, val_size=self.val_size, test_size=self.test_size)
        
        for split_name, split_data in self.data_splits.items():
#             split_name='train', split_data = self.data_splits['train']

            split_filepaths = list(collapse(split_data['path']))
            split_labels = split_data['label']
            num_samples = len(split_labels)
            print('Splitting',split_name, f'with {num_samples} total samples into {num_shards} shards')
            sharded_data = self.shard_data(split_filepaths, split_labels, num_shards)
#             shards = {}
            multiprocess_data = []
            for i, shard in enumerate(sharded_data):
                shard_fname = f'{split_name}-{str(i).zfill(5)}-of-{str(num_shards).zfill(5)}.tfrecord'
                multiprocess_data.append((i, split_name, num_shards, output_dir, list(shard), self.img_size))
            
            create_tfrecord_shard = self.create_tfrecord_shard
            def shard_worker(data_input):
                i, split_name, num_shards, output_dir, shard, target_size = data_input

                print('Creating shard : ', shard_fname)
                shard_filepath = os.path.join(output_dir,shard_fname)
                unzipped_shard = [list(i) for i in unzip(shard)]
                create_tfrecord_shard(shard_filepath,
                                      shard=unzipped_shard,
                                      shard_id=i,
                                      target_size = target_size)#self.img_size)
                
                
#             print('Initiating multiprocessing')
#             with Pool(os.cpu_count()//3) as pool:
#                 loaded_samples = pool.map(shard_worker, multiprocess_data)#, chunksize=64)
#             print('Finished multiprocessing')
                
#                 shard_fname = f'{split_name}-{str(i).zfill(5)}-of-{str(num_shards).zfill(5)}.tfrecord'
#                 print('Creating shard : ', shard_fname)

#                 shard_filepath = os.path.join(output_dir,shard_fname)
                
#                 shards[i] = list(shard)
#                 unzipped_shard = [list(i) for i in unzip(shards[i])]
#                 self.create_tfrecord_shard(shard_filepath,
#                                       shard=unzipped_shard,
#                                       shard_id=i,
#                                       target_size = self.img_size)
            
            break
                
                
#         from multiprocessing import Pool
#         from multiprocessing.dummy import Pool as ThreadPool
            
            
#         with ThreadPool() as pool:
#             pool.map(func, iterable)
            
            
            
        filename_log = check_if_tfrecords_exist(output_dir)

#         if filename_log == None:
#             filename_log = save_trainvaltest_tfrecords(dataset_name=self.name,
#                                                        output_dir=output_dir,
#                                                        target_size=self.img_size,
#                                                        low_count_threshold=self.low_count_threshold,
#                                                        val_size=self.val_size,
#                                                        test_size=self.test_size,
#                                                        num_shards=self.num_shards)
#             label_map = filename_log.pop('label_map', None)

        for key, records in filename_log.items():
            filename_log[key] = [os.path.join(output_dir,key,record_fname) for record_fname in sorted(records)]
        
        self.filename_log = filename_log
        
        return filename_log
    
    
    
    
    
if __name__ == '__main__':
    '''
    
    
    '''
#     test_dataset = BaseTFRecordDataset()

#     # data = test_dataset.extract(test_dataset.name)
#     # a = next(data)

#     data_splits, metadata_splits = test_dataset.load(test_dataset.name,
#                       test_dataset.low_count_threshold,
#                       test_dataset.val_size, 
#                       test_dataset.test_size)


#     filename_log = test_dataset.export_tfrecords()