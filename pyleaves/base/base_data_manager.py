# @Author: Jacob A Rose
# @Date:   Wed, April 1st 2020, 6:23 pm
# @Email:  jacobrose@brown.edu
# @Filename: base_data_manager.py

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
import importlib
from stuf import stuf
from pyleaves.utils.img_utils import rgb2gray_3channel, rgb2gray_1channel, ImageAugmentor
from pyleaves.leavesdb import experiments_db
from pyleaves.leavesdb.experiments_db import get_db_table, select_by_col, select_by_multicol

# data_manager_db = r'/media/data/jacob/Fossil_Project/data/data_manager.db'
# r'/home/jacob/Fossil_Project/data_manager.db'

# config = stuf({
#                 'name':'single_domain_experiment',
#
#                 'csv_dir':r'/media/data/jacob/Fossil_Project/data/csv_data',
#                 'tfrecord_dir':r'/media/data/jacob/Fossil_Project/data/tfrecord_data'
#               })


def get_preprocessing_func(model_name):
    if model_name == 'resnet_50_v2':
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif model_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input

    preprocess_input(tf.zeros([4, 32, 32, 3]))

    return lambda x,y: (preprocess_input(x),y)




class BaseDataManager:

    def __init__(self, config):
        """For managing data organized in experiment_db.db

        Parameters
        ----------
        config : type
            Description of parameter `config`.

        Returns
        -------
        type
            Description of returned object.

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """

        self.config = config
        self.tables = experiments_db.get_db_contents()
        self.run_id = config.run_id

    def get_all_tfrecord_rows(self):
        records = select_by_multicol(self.tables['tfrecords'], kwargs={'run_id':self.run_id})
        return records
        # return select_by_col(self.tables['tfrecords'],'run_id',self.run_id)

    def get_num_classes(self):
        return self.get_all_tfrecord_rows().iloc[0,:].num_classes

    def get_num_samples_by_file_group(self, file_group, dataset_stage='dataset_A'):
        records = select_by_multicol(self.tables['tfrecords'], kwargs={'run_id':self.run_id,
                                                                    'file_group':file_group,
                                                                    'dataset_stage':dataset_stage
                                                                    })
        return sum(select_by_col(records,'num_samples'))

    def get_tfrecord_paths_by_file_group(self, file_group, dataset_stage='dataset_A'):
        # import pdb; pdb.set_trace()
        records = select_by_multicol(self.tables['tfrecords'], kwargs={'run_id':self.run_id,
                                                                    'file_group':file_group,
                                                                    'dataset_stage':dataset_stage
                                                                    })
        return list(select_by_col(records,'file_path'))




class DataManager(BaseDataManager):

    def __init__(self, config):
        super().__init__(config)

        # preprocess_input = importlib.import_module(config.model_config.preprocessing_module).preprocess_input
        # lambda x,y: (x,y) #
        self.preprocess_input = get_preprocessing_func(model_name=config.model) #lambda x,y: (preprocess_input(x), y)
        self.augmentors = ImageAugmentor()

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
        img = tf.image.decode_jpeg(features['image/bytes'], channels=self.num_channels) # * 255.0
        img = tf.compat.v1.image.resize_image_with_pad(img, *self.target_size)

        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, depth=self.num_classes)

        return img, label


    def get_data_loader(self, file_group='train', dataset_stage='dataset_A'):
        # import pdb; pdb.set_trace()
        self.num_classes = self.config.num_classes = self.get_num_classes()
        # self.config.num_samples = self.get_num_samples_by_file_group(file_group,dataset_stage)
        tfrecord_paths = self.get_tfrecord_paths_by_file_group(file_group=file_group, dataset_stage=dataset_stage)
        config = self.config

        self.num_classes = self.get_num_classes()
        self.target_size = config.model_config.input_shape[:-1]
        self.num_channels = config.model_config.input_shape[-1]

        data = tf.data.Dataset.from_tensor_slices(tfrecord_paths) \
                    .apply(lambda x: tf.data.TFRecordDataset(x)) \
                    .map(self.decode_example, num_parallel_calls=AUTOTUNE) \
                    .map(self.preprocess_input, num_parallel_calls=AUTOTUNE)

        if config.grayscale == True:
            if self.num_channels==3:
                data = data.map(rgb2gray_3channel, num_parallel_calls=AUTOTUNE)
            elif self.num_channels==1:
                data = data.map(rgb2gray_1channel, num_parallel_calls=AUTOTUNE)

        if file_group == 'train':
            if config.augment_images:
                data = data.map(self.augmentors.rotate, num_parallel_calls=AUTOTUNE) \
                           .map(self.augmentors.flip, num_parallel_calls=AUTOTUNE)

            data = data.shuffle(buffer_size=1000, seed=config.seed)

        data = data.batch(config.batch_size, drop_remainder=True) \
                   .repeat() \
                   .prefetch(AUTOTUNE)

        return data
