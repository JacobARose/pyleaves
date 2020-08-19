# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:34 am
# @Email:  jacobrose@brown.edu
# @Filename: img_utils.py


'''
Functions for managing images
'''
from functools import partial
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
import pyleaves

from pyleaves.leavesdb.tf_utils.tf_utils import bytes_feature, int64_feature, float_feature
from pyleaves.leavesdb.experiments_db import DataBase, Table, TFRecordsTable, EXPERIMENTS_DB, EXPERIMENTS_SCHEMA, TFRecordItem



def _random_integer(minval, maxval, seed):
  """Returns a random 0-D tensor between minval and maxval.
  Args:
    minval: minimum value of the random tensor.
    maxval: maximum value of the random tensor.
    seed: random seed.
  Returns:
    A random 0-D tensor between minval and maxval.
  """
  return tf.random_uniform(
      [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)


def random_pad_image(image,
                     min_image_size=None,
                     max_image_size=None,
                     pad_color=None,
                     seed=None):
  """Randomly pads the image.
  This function randomly pads the image with zeros. The final size of the
  padded image will be between min_image_size and max_image_size.
  if min_image_size is smaller than the input image size, min_image_size will
  be set to the input image size. The same for max_image_size. The input image
  will be located at a uniformly random location inside the padded image.
  The relative location of the boxes to the original image will remain the same.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_image_size: a tensor of size [min_height, min_width], type tf.int32.
                    If passed as None, will be set to image size
                    [height, width].
    max_image_size: a tensor of size [max_height, max_width], type tf.int32.
                    If passed as None, will be set to twice the
                    image [height * 2, width * 2].
    pad_color: padding color. A rank 1 tensor of [3] with dtype=tf.float32.
               if set as None, it will be set to average color of the input
               image.
    seed: random seed.
  Returns:
    image: Image shape will be [new_height, new_width, channels].
  """
  if pad_color is None:
    pad_color = tf.reduce_mean(image, axis=[0, 1])

  image_shape = tf.shape(image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  if max_image_size is None:
    max_image_size = tf.stack([image_height * 2, image_width * 2])
  max_image_size = tf.maximum(max_image_size,
                              tf.stack([image_height, image_width]))

  if min_image_size is None:
    min_image_size = tf.stack([image_height, image_width])
  min_image_size = tf.maximum(min_image_size,
                              tf.stack([image_height, image_width]))

  target_height = tf.cond(
      max_image_size[0] > min_image_size[0],
      lambda: _random_integer(min_image_size[0], max_image_size[0], seed),
      lambda: max_image_size[0])

  target_width = tf.cond(
      max_image_size[1] > min_image_size[1],
      lambda: _random_integer(min_image_size[1], max_image_size[1], seed),
      lambda: max_image_size[1])

  offset_height = tf.cond(
      target_height > image_height,
      lambda: _random_integer(0, target_height - image_height, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  offset_width = tf.cond(
      target_width > image_width,
      lambda: _random_integer(0, target_width - image_width, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  new_image = tf.image.pad_to_bounding_box(
      image,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)

  # Setting color of the padded pixels
  image_ones = tf.ones_like(image)
  image_ones_padded = tf.image.pad_to_bounding_box(
      image_ones,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)
  image_color_padded = (1.0 - image_ones_padded) * pad_color
  new_image += image_color_padded

  return new_image












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
    def __init__(self, data, output_dir, columns={'source_path':'source_path','target_path':'path', 'label':'family'}):
        '''
        Class for managing different conversion functions depending on source and target image formats.
        '''
        self.output_ext = 'jpg'
        self.columns = columns

        self.labels = set(list(data[columns['label']]))
        [ensure_dir_exists(join(output_dir,label)) for label in self.labels]

        is_tiff = data[columns['source_path']].str.endswith('.tif')

        self.indices = {
                        'tiff':np.where(is_tiff)[0].tolist(),
                        'non_tiff':np.where(~is_tiff)[0].tolist()
                       }

        self.data = {
                    'tiff':data.iloc[self.indices['tiff'],:], #data['source_path'].str.endswith('.tif')],
                    'non_tiff':data.iloc[self.indices['non_tiff']] #~(data['source_path'].str.endswith('.tif'))]
                    }

    @property
    def subset(self):
        return self._subset

    @subset.setter
    def subset(self, new_subset):
        self._subset = new_subset
        self.output_dir = self.root_dir #join(self.root_dir,self._subset)
        ensure_dir_exists(self.output_dir)

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
        src_paths = get_dataset_from_list(sample_list=list(data_df[self.columns['source_path']]))
        src_labels = get_dataset_from_list(sample_list=list(data_df[self.columns['label']]))
        target_paths = get_dataset_from_list(sample_list=list(data_df[self.columns['target_path']]))
        mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels)).cache()
        return mappings_dataset

    def stage_converter(self, data_df, from_tiff=False):
        '''
        Arguments:
            data_df, pd.DataFrame:
                DataFrame containing columns ['path','source_path','label'] for image conversion
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

        img_reader = self.set_image_reader(output_ext=self.output_ext, from_tiff=from_tiff)

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

    def __init__(self, data, output_dir, columns={'source_path':'source_path','target_path':'path', 'label':'family'}):
        super().__init__(data, output_dir, columns)
        self.columns = columns

        self.output_ext='jpg'
        # self.input_dataset = self.stage_converter(data, columns=columns)

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

    def __init__(self,
                 data,
                 root_dir,
                 record_subdirs=[],
                 subset='train',
                 target_size=(224,224),
                 num_channels=3,
                 num_shards=10,
                 num_classes=1000,
                 TFRecordItem_factory=None,
                 tfrecords_table=TFRecordsTable(db_path=EXPERIMENTS_DB)):
        '''
        Example usage:

            coder = TFRecordCoder(data, output_dir, subset='train', target_size=(224,224), num_shards=10)
            coder.execute_convert()

        Arguments:
            data, dict:
                dict with keys ['path', 'label']
            root_dir:
                Root of experiment, contains subdir for each subset containing sequence of TFRecord shards
            record_subdirs:
                List of strings representing each level of subdirectory beneath root_dir to search before expecting train/val/test dirs.
        '''

        self.root_dir = os.path.join(root_dir, *record_subdirs)
        self.subset = subset
        self.target_size = target_size
        self.num_channels = num_channels
        self.num_shards = num_shards
        self.num_classes = num_classes

        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)
        self.data = data
        self.output_ext='tfrecord'
        self.filepath_log = []
        self.TFRecordItem_factory = TFRecordItem_factory
        self.tfrecords_table = tfrecords_table
        self.logs = []

#         temp = tf.zeros([4, 32, 32, 3])  # Or tf.zeros
#         tf.keras.applications.vgg16.preprocess_input(temp)

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

    def _get_shards(self, paths, labels, shard_size):
        return tf.data.Dataset.from_tensor_slices((paths, labels)) \
                .map(self.parse_image,num_parallel_calls=AUTOTUNE) \
                .batch(shard_size) \
                .prefetch(AUTOTUNE)

    def stage_dataset(self, data):
        paths = [path[0] for path in data]
        labels = [label[1] for label in data]
        # paths = [path[0] for path in data[self.x_col]]
        # labels = [label for label in data[self.y_col]]

        shard_size = self.num_samples//self.num_shards
        print('self.num_shards',self.num_shards)
        return self._get_shards(paths, labels, shard_size)

    def execute_batch(self, shard_id, images, labels):
        try:
            shard_filepath = self.gen_shard_filepath(shard_key=shard_id, output_dir = self.output_dir)
            self.filepath_log.append(shard_filepath)
            num_samples = labels.shape.as_list()[0]
            log_item = self.TFRecordItem_factory(file_path=shard_filepath, num_samples=num_samples)
            if self.tfrecords_table.check_if_logged([log_item])[0]:
                print('Found pre-written tfrecord',shard_filepath)
                return

            # import pdb; pdb.set_trace()
            images = images.numpy()
            labels = labels.numpy()
            with tf.io.TFRecordWriter(shard_filepath) as recordfile:
                start_time = time.perf_counter()
                # num_samples = len(labels)
                print(f'Writing {shard_filepath}')
                for i in tqdm(range(num_samples)):
                    example = self.encode_example(images[i,...], labels[i,...])
                    recordfile.write(example)
                end_time = time.perf_counter()
                self.tfrecords_table.log_tfrecord(log_item)
                print(f'Finished {shard_filepath}')
                print(f'Wrote {num_samples} in {end_time-start_time:.2f}')
        except Exception as e:
            print("Unexpected error:", sys.exc_info())
            print(f'[ERROR] {e}')
            raise

    def execute_convert(self):
        print(f"converting {self.num_samples} images to tfrecord")
        staged_data = self.stage_dataset(data=self.data)
        for shard_id, (images, labels) in enumerate(staged_data):
            self.execute_batch(shard_id, images, labels)

    # def read_tfrecords(self, tfrecord_paths=None, buffer_size=100, seed=17, batch_size=16, drop_remainder=False):
    #     if tfrecord_paths is None:
    #         tfrecord_paths = self.filepath_log
    #     return tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
    #         .apply(lambda x: tf.data.TFRecordDataset(x)) \
    #         .map(self.decode_example,num_parallel_calls=AUTOTUNE) \
    #         .shuffle(buffer_size=buffer_size, seed=seed) \
    #         .batch(batch_size,drop_remainder=drop_remainder) \
    #         .repeat() \
    #         .prefetch(AUTOTUNE)

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

    return image - means, label



def get_keras_preprocessing_function(model_name: str, input_format=tuple, x_col='path', y_col='label'):
    '''
    #TODO REFACTOR, Functionality currently moved to config_v2.py (4/18/20) [Also duplicated and set for deprecation in base.base_model.py]
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

    if input_format=='dict':
        def preprocess_func(input_example):
            x = input_example[x_col]
            y = input_example[y_col]
            return preprocess_input(x), y
        _temp = {x_col:tf.zeros([4, 32, 32, 3]), y_col:tf.zeros(())}
        preprocess_func(_temp)

    elif input_format=='tuple':
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




class ImageAugmentor:

    def __init__(self,
                 augmentations=['rotate',
                                'flip',
                                'color'],
                 probability=1.0,
                 seed=12):

        self.augmentations = augmentations
        self.p = probability
        self.seed = seed

        self.maps = {'rotate':self.rotate,
                      'flip':self.flip,
                      'color':self.color}

    def rotate(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Rotation augmentation

        Args:
            x,     tf.Tensor: Image
            label, tf.Tensor: arbitrary tensor, passes through unchanged

        Returns:
            Augmented image, label
        """

        # Rotate 0, 90, 180, 270 degrees
        return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32,seed=self.seed)), label

    def flip(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Flip augmentation

        Args:
            x,     tf.Tensor: Image to flip
            label, tf.Tensor: arbitrary tensor, passes through unchanged
        Returns:
            Augmented image, label
        """
        x = tf.image.random_flip_left_right(x, seed=self.seed)
        x = tf.image.random_flip_up_down(x, seed=self.seed)

        return x, label

    def color(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Color augmentation

        Args:
            x,     tf.Tensor: Image
            label, tf.Tensor: arbitrary tensor, passes through unchanged

        Returns:
            Augmented image, label
        """
        x = tf.image.random_hue(x, 0.08, seed=self.seed)
        x = tf.image.random_saturation(x, 0.6, 1.6, seed=self.seed)
        x = tf.image.random_brightness(x, 0.05, seed=self.seed)
        x = tf.image.random_contrast(x, 0.7, 1.3, seed=self.seed)
        return x, label

    def apply_augmentations(self, dataset: tf.data.Dataset) -> tf.data.Dataset: #x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """
        Call this function to apply all of the augmentation in the order of specification
        provided to the constructor __init__() of ImageAugmentor.

        Args:
            dataset, tf.data.Dataset: must yield individual examples of form (x, y)
        Returns:
            Augmented dataset
        """

        for f in self.augmentations:
            # Apply an augmentation only in 25% of the cases.000000
            aug = self.maps[f]
            dataset = dataset.map(lambda x,y: tf.cond(tf.random_uniform([], 0, 1) > (1 - self.p)), lambda: (aug(x),y), lambda: (x,y), num_parallel_calls=4)

        return dataset



## TODO: Integrate color conversions rgb2gray and gray2rgb into ImageAugmentor

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
    false_alarms = []

    def __init__(self, error_packet : tuple):
        '''
        Custom exception
        error_packet should contain
        1. A string or integer identifying the exact error
        2. A dict containing key info on the data that caused the error (e.g. One element from the output of a pd.DataFrame.to_dict('records'))
        '''

        assert len(error_packet)==2
        type(self).corrupted_files.append(error_packet)

        print(f'FOUND Corrupt JPEG: {error_packet}')
        print(f'total {len(self.__class__.corrupted_files)} corrupted files so far')


    # @classmethod
    # def get_failed_rows(cls):
    #     '''Return list of dicts representing each row that failed test'''
    #     return [failed[1] for failed in cls.get_failed_events()]
    #     return failed

    @classmethod
<<<<<<< HEAD
<<<<<<< HEAD
    def reset(cls):
        cls.corrupted_files = []
        cls.false_alarms = []

    @classmethod
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
    def get_failed_events(cls):
        '''Return list of tuples, containing (path,error) for each failed path'''
        failed = [c for c in cls.corrupted_files if not os.path.isfile(c[1]['source_path'])]
        print(len(failed),'total paths point to files that dont exist')
        existing_files = [c[1] for c in cls.corrupted_files if os.path.isfile(c[1]['source_path'])]

        successful=[]
        for i in range(len(existing_files)):
            try:
                img = io.imread(existing_files[i]['source_path'])
                successful.append(existing_files[i])
            except Exception as e:
                print(e)
                # print(existing_files[i], 'failed')
                failed.append(('Failed to read source_path',existing_files[i]))

        print(len(successful), 'total successes after false alarm error')
        print(len(failed), 'total failures')
        cls.false_alarms = successful

        return failed

    @classmethod
    def export_log(cls, filepath=None, log_type='events', ext_log=None):
        if ext_log:
            log_data = ext_log
        elif log_type=='events':
            log_data = cls.get_failed_events()

        if len(log_data)==0:
            return []
        records = []
        for i, (reason, row) in enumerate(log_data):
            records.append({**row, 'reason_omitted':reason})
        log_data = pd.DataFrame(records)

        if filepath:
            if filepath.endswith('.xlsx'):
                log_data.to_excel(filepath)
            elif filepath.endswith('.csv'):
                log_data.to_csv(filepath)
        return log_data


##############################################################################
##############################################################################
#DASK
import pdb;
isfile = os.path.isfile

<<<<<<< HEAD
<<<<<<< HEAD
import warnings
from PIL import Image
warnings.simplefilter('error', Image.DecompressionBombWarning)

=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
def uint16_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    img = 255*(img - img.min())/(img.max()-img.min())
    return img.astype(np.uint8)

def load(sample : dict, column='source_path'):
    try:
        img = io.imread(sample[column])
        return uint16_to_uint8(img)
    except OSError:
        print(str(OSError))
        raise CorruptJPEGError(('OSError: failed loading source',sample))
    except:
        raise CorruptJPEGError(('Unknown: failed loading source',sample))
# @dask.delayed
def save(sample : dict, img : np.ndarray, column='path'):
    try:
        io.imsave(sample[column], img, quality=100)#95)
        assert isfile(sample[column])
    except:
        raise CorruptJPEGError(('failed saving target',sample))
    return sample

@dask.delayed
def process_file(sample : dict, columns={'source_path':'source_path','target_path':'path'}):
    try:
        if isfile(sample[columns['target_path']]):
            return sample
        img = load(sample, column=columns['source_path'])
        result = save(sample, img, column=columns['target_path'])
        return sample
    except CorruptJPEGError:
        return None
    except OSError:
        CorruptJPEGError((str(OSError), sample))
        print(str(OSError))
        return None
<<<<<<< HEAD
<<<<<<< HEAD
    except Image.DecompressionBombWarning:
        CorruptJPEGError((str(Image.DecompressionBombWarning), sample))
        print(str(Image.DecompressionBombWarning))
        return None
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
    except:
        print('Unknown conversion error for source image:', sample[columns['source_path']])
        CorruptJPEGError(('Unknown file processing failure', sample))
        return None
    # return sample


class DaskCoder:
    def __init__(self, data, output_dir=None, columns={'source_path':'source_path','target_path':'path', 'label':'family'}):
        '''
        Dask version
        Class for managing different conversion functions depending on source and target image formats.
        '''
        self.output_ext = 'jpg'

        # if output_dir:
        #     labels = set(list(data[columns['label']]))
        #     [ensure_dir_exists(join(output_dir,label)) for label in labels]

        # self.indices = len(data[columns['source_path']])
        self.data = data
        self.input_dataset = self._stage_converter(data, columns=columns)


    def _stage_dataset(self, data_df, columns={'source_path':'source_path','target_path':'path'}):
        '''Prepare dataset to be passed to converter '''
        return data_df.to_dict('records')
        # return data_df.rename(columns=columns).to_dict('records')

    def _stage_converter(self, data_df, columns={'source_path':'source_path','target_path':'path'}):
        '''Set up Dask graph for lazy execution, to be computed at a later time.

        Arguments:
            data_df, pd.DataFrame:
                DataFrame containing columns ['path','sourcepath','label'] for image conversion
        Return:
            converted_dataset, list(dask.Delayed):
                List of delayed functions to be computed at a later time
        '''
        mappings_dataset = self._stage_dataset(data_df, columns=columns)
        converted_dataset = [process_file(row,columns=columns) for row in mappings_dataset]
        # import pdb; pdb.set_trace()
        # converted_dataset = [item for item in converted_dataset if item is not None]
        return converted_dataset

    def execute_conversion(self, input_dataset):
        '''
        Arguments:
            input_dataset, list(dask.Delayed):
                List of dask.Delayed functions that has been output from the _stage_converter() method, but not yet iterated through.
        Return:
            output, list:
                List of whatever information is returned by input_dataset, specified in the specific coding/conversion function
        '''
        # print('BEGINNING THE DAMN CONVERSION')
        with ProgressBar():
            computed_results = dask.compute(*input_dataset)

        computed_results = pd.DataFrame([item for item in computed_results if item is not None])
        return computed_results























# def load(filepath):
#     try:
#         img = io.imread(filepath)
#         return img.astype(np.uint8)
#     except OSError:
#         print(str(OSError))
#         raise CorruptJPEGError(('failed loading source',filepath))
#     except:
#         raise CorruptJPEGError(('failed loading source',filepath))
# # @dask.delayed
# def save(filepath, img):
#     try:
#         io.imsave(filepath, img, quality=95)
#         assert isfile(filepath)
#     except:
#         raise CorruptJPEGError(('failed saving target',filepath))
#     return filepath
#
# @dask.delayed
# def process_file(src_fpath, target_fpath):
#     try:
#         if isfile(target_fpath):
#             return (2, target_fpath)
#         img = load(src_fpath)
#         result = save(target_fpath, img)
#         return (1, result)
#     except CorruptJPEGError:
#         pass
#     except OSError:
#         print(str(OSError))
#     except:
#         print('failed source image path:', src_fpath)
#         raise CorruptJPEGError(('failed file', src_fpath))
#     return (0, src_fpath)
#
#
# class DaskCoder:
#     def __init__(self, data, output_dir=None, columns={'source_path':'source_path','target_path':'path', 'label':'family'}):
#         '''
#         Dask version
#         Class for managing different conversion functions depending on source and target image formats.
#         '''
#         self.output_ext = 'jpg'
#
#         labels = set(list(data[columns['label']]))
#
#         if output_dir:
#             [ensure_dir_exists(join(output_dir,label)) for label in labels]
#
#         self.indices = len(data[columns['source_path']])
#         self.data = data
#
#         self.input_dataset = self.stage_converter(data, columns=columns)
#
#
#     def stage_dataset(self, data_df, columns={'source_path':'source_path','target_path':'path', 'label':'family'}):
#         src_paths = list(data_df[columns['source_path']])
# #         src_labels = list(data_df['label'])
#         target_paths = list(data_df[columns['target_path']])
#         mappings_dataset = zip(src_paths, target_paths) #, src_labels)
#         return mappings_dataset
#
#     def stage_converter(self, data_df, columns={'source_path':'source_path','target_path':'path'}):
#         '''
#         Arguments:
#             data_df, pd.DataFrame:
#                 DataFrame containing columns ['path','sourcepath','label'] for image conversion
#         Return:
#             converted_dataset, list(dask.Delayed):
#                 List of delayed functions to be computed at a later time
#
#         '''
#         output_ext = self.output_ext
#         mappings_dataset = self.stage_dataset(data_df, columns=columns)
#
#         converted_dataset = [process_file(src, target) for src, target in mappings_dataset]
#         return converted_dataset
#
#     def execute_conversion(self, input_dataset):
#         '''
#         Arguments:
#             input_dataset, list(dask.Delayed):
#                 List of dask.Delayed functions that has been output from the stage_converter() method, but not yet iterated through.
#         Return:
#             output, list:
#                 List of whatever information is returned by input_dataset, specified in the specific coding/conversion function
#         '''
#         with ProgressBar():
#             computed_results = dask.compute(*input_dataset)
#
#         return computed_results






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
