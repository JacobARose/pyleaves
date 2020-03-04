'''
Functions for managing images
'''

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from skimage import io
import cv2
from dask.diagnostics import ProgressBar
import dask #.delayed as dd
import dataset
import matplotlib.pyplot as plt
from more_itertools import chunked
import numpy as np
import os
import pandas as pd
from pathos.multiprocessing import ProcessPool
from pathos.threading import ThreadPool
import sys
from stuf import stuf
from threading import Lock
from tqdm import tqdm

import time

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

from tensorflow.keras.applications.vgg16 import preprocess_input

AUTOTUNE = tf.data.experimental.AUTOTUNE

join = os.path.join
splitext = os.path.splitext
basename = os.path.basename

from pyleaves.utils import ensure_dir_exists
from pyleaves.config import DatasetConfig
# from pyleaves import leavesdb
from pyleaves.leavesdb.tf_utils.tf_utils import bytes_feature, int64_feature, float_feature

def get_dataset_from_list(sample_list):
    '''

    '''
    samples = tf.data.Dataset.from_tensor_slices(sample_list)
    samples = samples.prefetch(1)
    return samples

def copy_img2png(src_filepath, target_filepath, label):

    img = tf.io.read_file(src_filepath)
    img = tf.image.decode_image(img, channels=3)

    img = tf.image.encode_png(img)
    tf.io.write_file(target_filepath, img)
    return target_filepath, label

def copy_tiff2png(src_filepath, target_filepath, label):

    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
    try:
        img = cv2.imread(src_filepath.decode('utf-8'), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise
    try:
        cv2.imwrite(target_filepath.decode('utf-8'), img, compression_params)
        return target_filepath, label
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise

##############################################################################

def copy_img2jpg(src_filepath, target_filepath, label):

    img = tf.io.read_file(src_filepath)
    img = tf.image.decode_image(img, channels=3)

    img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)
    tf.io.write_file(target_filepath, img)
    return target_filepath, label

def copy_tiff2jpg(src_filepath, target_filepath, label):

    try:
        img = cv2.imread(src_filepath.decode('utf-8'), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise
    try:
        cv2.imwrite(target_filepath.decode('utf-8'), img)
        return target_filepath, label
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise


##############################################################################
def time_ds(ds):
    start_time = time.perf_counter()
    failed_files =[]
    for i, sample in enumerate(ds):
        print(i)
        if sample==0:
            failed_files.append(i)
    end_time = time.perf_counter()
    run_time = end_time-start_time
    print(f'converted {i} files in {run_time} seconds, at rate {i/run_time} images/sec')
    return failed_files
##############################################################################

##############################################################################
##############################################################################

class Coder:
    def __init__(self, data, output_dir):
        '''
        Class for managing different conversion functions depending on source and target image formats.
        '''
        self.output_ext = 'jpg'

        labels = set(list(data['label']))
        [ensure_dir_exists(join(output_dir,label)) for label in labels]

        self.indices = {
                        'tiff':np.where(data['source_path'].str.endswith('.tif')),
                        'non_tiff':np.where(~(data['source_path'].str.endswith('.tif')))
                       }

        self.data = {
                    'tiff':data[data['source_path'].str.endswith('.tif')],
                    'non_tiff':data[~(data['source_path'].str.endswith('.tif'))]
                    }

    def set_image_reader(self,output_ext='png', from_tiff=False):
        if from_tiff:
            if output_ext=='jpg':
                img_reader = lambda src, target, label: tf.py_func(copy_tiff2jpg, [src, target, label],[tf.string, tf.string])
            elif output_ext=='png':
                img_reader = lambda src, target, label: tf.py_func(copy_tiff2png, [src, target, label],[tf.string, tf.string])
        else:
            if output_ext=='jpg':
                img_reader = lambda src, target, label: copy_img2jpg(src, target, label)
            elif output_ext=='png':
                img_reader = lambda src, target, label: copy_img2png(src, target, label)
        self.img_reader = img_reader
        return img_reader

    def stage_dataset(self, data_df):
        src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
        src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
        target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
        mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels)).cache()
        return mappings_dataset

    def stage_converter(self, data_df, from_tiff=False):
        '''
        Arguments:
            data_df, pd.DataFrame:
                DataFrame containing columns ['path','sourcepath','label'] for image conversion
            from_tiff, bool:
                Indicates whether to use converters for converting from tiff images, or to use generic image converters for converting from jpg or png
            output_ext, bool:
                Default='jpg'
                Must be either 'jpg' or 'png'
        Return:
            converted_dataset, tf.data.Dataset:
                Dataset that, when iterated over, will read and write images referenced in data_df

        '''
        mappings_dataset = self.stage_dataset(data_df)

        img_reader = self.set_image_reader(self,output_ext=self.output_ext, from_tiff=from_tiff)

        converted_dataset = mappings_dataset.map(img_reader, num_parallel_calls=AUTOTUNE)
        return converted_dataset.prefetch(AUTOTUNE)

    def execute_conversion(self, input_dataset):
        '''
        Arguments:
            input_dataset, tf.data.Dataset:
                Dataset that has been output from the stage_converter() method, but not yet iterated through.
        Return:
            output, list:
                List of whatever information is returned by input_dataset, specified in the specific coding/conversion function
        '''
        perf_counter = time.perf_counter
        output=[]
        indices = [0]
        timelog=[perf_counter()]
        j=0
        for i, converted_data in enumerate(input_dataset):
            output.append(converted_data)
            if i%20==0:
                indices.append(i+1)
                timelog.append(perf_counter())
                idx = (indices[j],indices[j+1])
                print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
                j+=1
        return output


class JPGCoder(Coder):

    def __init__(self, data, output_dir):
        super().__init__(data, output_dir)

        self.output_ext='jpg'

    def batch_convert(self):

        outputs = []

        try:
            if self.data['non_tiff'].shape[0]>0:
                print(f"converting {self.data['non_tiff'].shape[0]} non-tiff images to jpg")
                non_tiff_staged = self.stage_converter(data_df=self.data['non_tiff'],from_tiff=False)
                outputs.extend(self.execute_conversion(non_tiff_staged))

            if self.data['tiff'].shape[0]>0:
                print(f"converting {self.data['tiff'].shape[0]} tiff images to jpg")
                tiff_staged = self.stage_converter(data_df=self.data['tiff'],from_tiff=True)
                outputs.extend(self.execute_conversion(tiff_staged))
            return outputs

        except Exception as e:
            print("Unexpected error:", sys.exc_info())
            print(f'[ERROR] {e}')
            raise

##################################################################
##################################################################


def load_and_encode_example(path, label, target_size = (224,224)):
    img = load_image(path, target_size=target_size)
    return encode_example(img,label)


class TFRecordCoder(JPGCoder):

    def __init__(self, data, root_dir, subset='train', target_size=(224,224), num_channels=3, num_shards=10, num_classes=1000):
        '''
        Example usage:
        
            coder = TFRecordCoder(data, output_dir, subset='train', target_size=(224,224), num_shards=10)
            coder.execute_convert()
        
        Arguments:
            data, dict:
                dict with keys ['path', 'label']
            root_dir:
                Root of experiment, contains subdir for each subset containing sequence of TFRecord shards
        '''
        
        
#         ensure_dir_exists(join(output_dir,subset))
        self.root_dir = root_dir
        self.subset = subset
        self.target_size = target_size
        self.num_channels = num_channels
        self.num_shards = num_shards
        self.num_classes = num_classes
        
        self.num_samples = len(data['label'])
        self.indices = np.arange(self.num_samples)
        self.data = data
        self.output_ext='tfrecord'
        self.filepath_log = []
        
        temp = tf.zeros([4, 32, 32, 3])  # Or tf.zeros
        tf.keras.applications.vgg16.preprocess_input(temp)

    @property
    def subset(self):
        return self._subset
    
    @subset.setter
    def subset(self, new_subset):
        self._subset = new_subset
        self.output_dir = join(self.root_dir,self._subset)
        ensure_dir_exists(self.output_dir) 
        
    def gen_shard_filepath(self, shard_key, output_dir):
        '''
        e.g. shard_filepath = self.gen_shard_filepth(shard_key=0, output_dir)
        '''
        shard_fname = f'{self.subset}-{str(shard_key).zfill(5)}-of-{str(self.num_shards).zfill(5)}.tfrecord'
        shard_filepath = os.path.join(output_dir,shard_fname)
        return shard_filepath

    def parse_image(self, src_filepath, label):

        img = tf.io.read_file(src_filepath)
        img = tf.image.decode_image(img, channels=3)
        img = tf.compat.v1.image.resize_image_with_pad(img, *self.target_size)
        return img, label

    def encode_example(self, img, label):
        img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)
        
        features = {
                    'image/bytes': bytes_feature(img),
                    'label': int64_feature(label)
                    }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def decode_example(self, example):
        feature_description = {
                                'image/bytes': tf.io.FixedLenFeature([], tf.string),
                                'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
                                }
        features = tf.io.parse_single_example(example,features=feature_description)

        img = tf.image.decode_jpeg(features['image/bytes'], channels=3) # * 255.0
        img = tf.compat.v1.image.resize_image_with_pad(img, *self.target_size)
        
        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, depth=self.num_classes)

        return img, label       
            
    def _get_sharded_dataset(self, paths, labels, shard_size):
        return tf.data.Dataset.from_tensor_slices((paths, labels)) \
                .map(self.parse_image,num_parallel_calls=AUTOTUNE) \
                .batch(shard_size) \
                .prefetch(AUTOTUNE)

    def stage_dataset(self, data):
        paths = [path[0] for path in data['path']]
        labels = [label for label in data['label']]

        shard_size = self.num_samples//self.num_shards
        print('self.num_shards',self.num_shards)
        return self._get_sharded_dataset(paths, labels, shard_size)
    
    def execute_batch(self, shard_id, images, labels):
        try:
            shard_filepath = self.gen_shard_filepath(shard_key=shard_id, output_dir = self.output_dir)
            self.filepath_log.append(shard_filepath)

            images = images.numpy()
            labels = labels.numpy()
            with tf.io.TFRecordWriter(shard_filepath) as recordfile:
                print(f'Writing {shard_filepath}')
                start_time = time.perf_counter()
                num_samples = len(labels)
                for i in tqdm(range(num_samples)):
                    example = self.encode_example(images[i,...], labels[i,...])
                    recordfile.write(example)
                end_time = time.perf_counter()
                print(f'Finished {shard_filepath}')
                print(f'Wrote {len(labels)} in {end_time-start_time:.2f}')            
        except Exception as e:
            print("Unexpected error:", sys.exc_info())
            print(f'[ERROR] {e}')
            raise

    def execute_convert(self):
        print(f"converting {self.num_samples} images to tfrecord")
        staged_data = self.stage_dataset(data=self.data)

        for shard_id, (images, labels) in enumerate(staged_data):
            self.execute_batch(shard_id, images, labels)
        
    def read_tfrecords(self, tfrecord_paths=None, buffer_size=100, seed=17, batch_size=16, drop_remainder=False):
        if tfrecord_paths is None:
            tfrecord_paths = self.filepath_log
        return tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
            .apply(lambda x: tf.data.TFRecordDataset(x)) \
            .map(self.decode_example,num_parallel_calls=AUTOTUNE) \
            .shuffle(buffer_size=buffer_size, seed=seed) \
            .batch(batch_size,drop_remainder=drop_remainder) \
            .repeat() \
            .prefetch(AUTOTUNE)
            
_R_MEAN = 123.68 / 255.0
_G_MEAN = 116.78 / 255.0
_B_MEAN = 103.94 / 255.0
        
def imagenet_mean_subtraction(image, label):
    """
    Subtracts the given means from each image channel.
    For example:
        means = [123.68, 116.779, 103.939]
        image = imagenet_mean_subtraction(image, means)
    Args:
        image: a tensor of size [height, width, C]. Range between [0,1]
        means: a C-vector of values to subtract from each channel.
        num_channels: number of color channels in the image that will be distorted.
    Returns:
        the centered image.
    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    

    means = tf.reshape(
                        tf.constant([_R_MEAN, _G_MEAN, _B_MEAN]),
                        [1,1,3]
                       )
    
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

#     if len(means) != num_channels:
#         raise ValueError('len(means) must match the number of channels')

    return image - means, label



def get_keras_preprocessing_function(model_name: str, input_format=tuple):
    '''
    if input_dict_format==True:
        Includes value unpacking in preprocess function to accomodate TFDS {'image':...,'label':...} format
    '''
    if model_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif model_name == 'xception':
        from tensorflow.keras.applications.xception import preprocess_input
    elif model_name in ['resnet_50_v2','resnet_101_v2']:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    else:
        preprocess_input = lambda x: x
    
    if input_format==dict:
        def preprocess_func(input_example):
            x = input_example['image']
            y = input_example['label']
            return preprocess_input(x), y
        _temp = {'image':tf.zeros([4, 32, 32, 3]), 'label':tf.zeros(())}
        preprocess_func(_temp)
        
    elif input_format==tuple:
        def preprocess_func(x, y):
            return preprocess_input(x), y
        _temp = ( tf.zeros([4, 32, 32, 3]), tf.zeros(()) )        
        preprocess_func(*_temp)
    else:
        print('''input_format must be either dict or tuple, corresponding to data organized as:
              tuple: (x, y)
              or
              dict: {'image':x, 'label':y}
              ''')
        return None
    
    return preprocess_func






# def get_keras_preprocessing_function(model_name: str, input_format=tuple):
#     '''
#     if input_dict_format==True:
#         Includes value unpacking in preprocess function to accomodate TFDS {'image':...,'label':...} format
#     '''
#     if model_name == 'vgg16':
#         from tensorflow.keras.applications.vgg16 import preprocess_input
#     elif model_name == 'xception':
#         from tensorflow.keras.applications.xception import preprocess_input
#     elif model_name in ['resnet_50_v2','resnet_101_v2']:
#         from tensorflow.keras.applications.resnet_v2 import preprocess_input
#     else:
#         preprocess_input = lambda x: x
    
#     if input_format==dict:
#         def preprocess_func(input_example):

#             x = tf.cast(input_example['image'],tf.float32)
#             y = input_example['label']
#             return preprocess_input(x), y
#         _temp = {'image':tf.zeros([4, 32, 32, 3]), 'label':tf.zeros(())}
#         preprocess_func(_temp)
        
#     elif input_format==tuple:
#         def preprocess_func(x, y):
#             x = tf.cast(x, tf.float32)
#             return preprocess_input(x), y
#         _temp = ( tf.zeros([4, 32, 32, 3]), tf.zeros(()) )        
#         preprocess_func(*_temp)
#     else:
#         print('''input_format must be either dict or tuple, corresponding to data organized as:
#               tuple: (x, y)
#               or
#               dict: {'image':x, 'label':y}
#               ''')
#         return None
    
#     return preprocess_func

    
    
class ImageAugmentor:
    
    def __init__(self,
                 augmentations=['rotate',
                                'flip',
                                'color'],
                 seed=12):
        self.augmentations = augmentations        
        self.seed = seed
    def rotate(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Rotation augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        # Rotate 0, 90, 180, 270 degrees
        return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32,seed=self.seed)), label


    def flip(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """
        x = tf.image.random_flip_left_right(x, seed=self.seed)
        x = tf.image.random_flip_up_down(x, seed=self.seed)

        return x, label

    def color(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Color augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """
        x = tf.image.random_hue(x, 0.08, seed=self.seed)
        x = tf.image.random_saturation(x, 0.6, 1.6, seed=self.seed)
        x = tf.image.random_brightness(x, 0.05, seed=self.seed)
        x = tf.image.random_contrast(x, 0.7, 1.3, seed=self.seed)
        return x, label

    
    
def rgb2gray_3channel(img, label):
    '''
    Convert rgb image to grayscale, but keep num_channels=3
    '''
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.grayscale_to_rgb(img)
    return img, label

def rgb2gray_1channel(img, label):
    '''
    Convert rgb image to grayscale, num_channels from 3 to 1
    '''
    img = tf.image.rgb_to_grayscale(img)
    return img, label    

    
    

    
# def augment(x: tf.Tensor) -> tf.Tensor:
#     """Some augmentation

#     Args:
#         x: Image

#     Returns:
#         Augmented image
#     """
#     x = .... # augmentation here
#     return x

        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
##############################################################################
##############################################################################        
        

class CorruptJPEGError(Exception):
    corrupted_files = []
    def __init__(self, *args):
        '''
        Custom exception 
        '''
        print(f'FOUND Corrupt JPEG: {args}')
        if args:
            type(self).corrupted_files.append(*args)
        


##############################################################################
##############################################################################
#DASK

# @dask.delayed
def load(filepath):
    try:
        img = io.imread(filepath)
        return img
    except:
        raise CorruptJPEGError(filepath)
# @dask.delayed
def save(filepath, img):
    try:
        io.imsave(filepath, img, quality=95)
    except:
        raise CorruptJPEGError(filepath)
    return filepath

@dask.delayed
def process_file(src_fpath, target_fpath):
    try:
        img = load(src_fpath)
        result = save(target_fpath, img)
        return result
    except:
        print('source image path:', src_fpath)
        CorruptJPEGError(src_fpath)
        return src_fpath


class DaskCoder:
    def __init__(self, data, output_dir):
        '''
        Dask version
        Class for managing different conversion functions depending on source and target image formats.
        '''
        self.output_ext = 'jpg'

        labels = set(list(data['label']))
        [ensure_dir_exists(join(output_dir,label)) for label in labels]

        self.indices = len(data['source_path'])
        self.data = data

        self.input_dataset = self.stage_converter(data)


    def stage_dataset(self, data_df):
        src_paths = list(data_df['source_path'])
#         src_labels = list(data_df['label'])
        target_paths = list(data_df['path'])
        mappings_dataset = zip(src_paths, target_paths) #, src_labels)
        return mappings_dataset

    def stage_converter(self, data_df):
        '''
        Arguments:
            data_df, pd.DataFrame:
                DataFrame containing columns ['path','sourcepath','label'] for image conversion
        Return:
            converted_dataset, list(dask.Delayed):
                List of delayed functions to be computed at a later time

        '''
        output_ext = self.output_ext
        mappings_dataset = self.stage_dataset(data_df)

        converted_dataset = [process_file(src, target) for src, target in mappings_dataset]
        return converted_dataset

    def execute_conversion(self, input_dataset):
        '''
        Arguments:
            input_dataset, list(dask.Delayed):
                List of dask.Delayed functions that has been output from the stage_converter() method, but not yet iterated through.
        Return:
            output, list:
                List of whatever information is returned by input_dataset, specified in the specific coding/conversion function
        '''
        with ProgressBar():
            computed_results = dask.compute(*input_dataset)

        return computed_results






##############################################################################

def convert_from_nontiff2png(data_df):
    '''
    data_df must only contain filenames referring to non-TIFF formatted images (e.g. PNG, JPG, GIF) in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    converted_dataset = mappings_dataset.map(lambda src, target, label: copy_img2png(src, target, label), num_parallel_calls=AUTOTUNE)

    perf_counter = time.perf_counter

    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1

    return output


def convert_from_tiff2png(data_df):
    '''
    data_df must only contain filenames referring to TIFF formatted images in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    tiff_reader = lambda src_filepath, target_filepath, label: tf.py_func(copy_tiff2png, [src_filepath, target_filepath, label],[tf.string, tf.string])
    converted_dataset = mappings_dataset.map(tiff_reader, num_parallel_calls=AUTOTUNE)
    perf_counter = time.perf_counter

    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1
    return output


def convert_from_nontiff2jpg(data_df):
    '''
    data_df must only contain filenames referring to non-TIFF formatted images (e.g. PNG, JPG, GIF) in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    converted_dataset = mappings_dataset.map(lambda src, target, label: copy_img2jpg(src, target, label), num_parallel_calls=AUTOTUNE)

    perf_counter = time.perf_counter

    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1

    return output


def convert_from_tiff2jpg(data_df):
    '''
    data_df must only contain filenames referring to TIFF formatted images in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    tiff_reader = lambda src_filepath, target_filepath, label: tf.py_func(copy_tiff2jpg, [src_filepath, target_filepath, label],[tf.string, tf.string])
    converted_dataset = mappings_dataset.map(tiff_reader, num_parallel_calls=AUTOTUNE)
    perf_counter = time.perf_counter

    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1
    return output


##############################################################################

def convert_to_png(data_df, output_dir):
    '''
    Function to load a list of image files, convert to png format if necessary, and save to specified target dir.

    Arguments:
        dataset_name, str:
            Name of source dataset from which images are sourced, to be name of subdir in target root dir
        target_dir, str:
            Root directory for converted images, which will be saved in hierarchy:
            root/
                |dataset_1/
                          |class_1/
                                  |image_1
                                  |image_2
                                  ...

    Return:

    '''

    labels = set(list(data_df['label']))
    [ensure_dir_exists(join(output_dir,label)) for label in labels]
    indices = list(range(len(data_df)))

    tiff_data = data_df[data_df['source_path'].str.endswith('.tif')]
    non_tiff_data = data_df[~(data_df['source_path'].str.endswith('.tif'))]

    outputs = []
    try:
        if non_tiff_data.shape[0]>0:
            print(f'converting {non_tiff_data.shape[0]} non-tiff images to png')
            outputs.extend(convert_from_nontiff2png(non_tiff_data))
        if tiff_data.shape[0]>0:
            print(f'converting {tiff_data.shape[0]} tiff images to png')
            outputs.extend(convert_from_tiff2png(tiff_data))
        return outputs

    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'[ERROR] {e}')
        raise


def convert_to_jpg(data_df, output_dir):
    '''
    Function to load a list of image files, convert to jpg format if necessary, and save to specified target dir.

    Arguments:
        dataset_name, str:
            Name of source dataset from which images are sourced, to be name of subdir in target root dir
        target_dir, str:
            Root directory for converted images, which will be saved in hierarchy:
            root/
                |dataset_1/
                          |class_1/
                                  |image_1
                                  |image_2
                                  ...

    Return:

    '''

    labels = set(list(data_df['label']))
    [ensure_dir_exists(join(output_dir,label)) for label in labels]
    indices = list(range(len(data_df)))

    tiff_data = data_df[data_df['source_path'].str.endswith('.tif')]
    non_tiff_data = data_df[~(data_df['source_path'].str.endswith('.tif'))]

    outputs = []
    try:
        if non_tiff_data.shape[0]>0:
            print(f'converting {non_tiff_data.shape[0]} non-tiff images to jpg')
            outputs.extend(convert_from_non_tiff2jpg(non_tiff_data))
        if tiff_data.shape[0]>0:
            print(f'converting {tiff_data.shape[0]} tiff images to jpg')
            outputs.extend(convert_from_tiff2jpg(tiff_data))
        return outputs

    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'[ERROR] {e}')
        raise





##############################################################################

def plot_image_grid(imgs, labels = np.array([]), x_plots = 4, y_plots = 4, figsize=(15,15)):
	fig, axes = plt.subplots(y_plots, x_plots, figsize=figsize)
	axes = axes.flatten()

	num_imgs = len(imgs)

	if len(axes) > num_imgs:
		axes = axes[:num_imgs]
	for i, ax in enumerate(axes):
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)

		ax.imshow(imgs[i,...])
		if len(labels) >= i:
			ax.set_title(labels[i])
	plt.tight_layout()


def pad_image(img, target_size, interpolation=cv2.INTER_CUBIC):
    old_size = img.shape[:2]
    ratio = np.min(np.array(target_size)/old_size)
    new_size = tuple(np.int16(np.array(old_size)*ratio))

    img = cv2.resize(img,tuple(new_size)[::-1],interpolation=cv2.INTER_CUBIC)

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=[0, 0, 0])
    return img

def resize(img, target_size, padding = True, interpolation=cv2.INTER_CUBIC):
    '''
    Resize function with option to pad borders to avoid warping image aspect ratio
    '''
    if padding == True:
        img = pad_image(img, tuple(target_size), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, tuple(target_size), interpolation=cv2.INTER_CUBIC)
    return img


def load_image(filepath, target_size=(224,224)):
    '''
    Read an image stored at filepath, and resize to target_size.
    Written to default resize with 0 padding in order to conserve aspect ratio. If alternative
    resizing or padding is desired the function can be easily refactored.

    '''
    try:
#         print(filepath)
        img = cv2.imread(filepath)
    except:
        print('[error:] ', filepath)
        return None
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    return img
