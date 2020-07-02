# @Author: Jacob A Rose
# @Date:   Wed, May 27th 2020, 4:35 am
# @Email:  jacobrose@brown.edu
# @Filename: tf_parameterized_train_main.py


#!/usr/bin/env python
# coding: utf-8


'''
python /home/jacob/projects/pyleaves/pyleaves/train/tf_parameterized_train_main.py --neptune_project_name 'jacobarose/sandbox' --experiment_name pnas_minimal_example --config_path '/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/pnas_vgg16_config.json' --gpu_id 5

python /home/jacob/projects/pyleaves/pyleaves/train/tf_parameterized_train_main.py --neptune_project_name 'jacobarose/sandbox' --experiment_name pnas_minimal_example --config_path '/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/pnas_shallow_config.json' --gpu_id 6

python /home/jacob/projects/pyleaves/pyleaves/train/tf_parameterized_train_main.py --neptune_project_name 'jacobarose/sandbox' --experiment_name pnas_minimal_example --config_path '/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/pnas_vgg16_config_50-50.json' --gpu_id 5 -tags '50-50_split'


python /home/jacob/projects/pyleaves/pyleaves/train/tf_parameterized_train_main.py --neptune_project_name 'jacobarose/sandbox' --experiment_name pnas_minimal_example --config_path '/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/pnas_resnet_config.json' --gpu_id 6

python /home/jacob/projects/pyleaves/pyleaves/train/tf_parameterized_train_main.py --neptune_project_name 'jacobarose/sandbox' --experiment_name leaves_minimal_example --config_path '/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/leaves_resnet_config.json' --gpu_id 7

python /home/jacob/projects/pyleaves/pyleaves/train/tf_parameterized_train_main.py --neptune_project_name 'jacobarose/sandbox' --experiment_name leaves_minimal_example --config_path '/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/leaves_vgg16_config.json' --gpu_id 7

python /home/jacob/projects/pyleaves/pyleaves/train/tf_parameterized_train_main.py --neptune_project_name 'jacobarose/sandbox' --experiment_name 'tf_flowers_minimal_example' --config_path '/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/tf_flowers_resnet_config.json' --gpu_id 6



'''

def main(**kwargs):

    import sys

    for k, v in kwargs.items():
        sys.argv += [k, v]

    from pprint import pprint
    import argparse
    import datetime
    import json
    import os


    parser = argparse.ArgumentParser()
    parser.add_argument('--neptune_project_name', default='jacobarose/sandbox', type=str, help='Neptune.ai project name to log under')
    parser.add_argument('--experiment_name', default='pnas_minimal_example', type=str, help='Neptune.ai experiment name to log under')
    parser.add_argument('--config_path', default=r'/home/jacob/projects/pyleaves/pyleaves/configs/example_configs/pnas_resnet_config.json', type=str, help='JSON config file')
    parser.add_argument('-gpu', '--gpu_id', default='1', type=str, help='integer number of gpu to train on', dest='gpu_id')
    parser.add_argument('-tags', '--add-tags', default=[], type=str, nargs='*', help='Add arbitrary list of tags to apply to this run in neptune', dest='tags')
    parser.add_argument('-f', default=None)
    args = parser.parse_args()

    with open(args.config_path, 'r') as config_file:
        PARAMS = json.load(config_file)

    # print(gpu)
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    pprint(PARAMS)
    import tensorflow as tf
    import neptune
    # tf.debugging.set_log_device_placement(True)
    print(tf.__version__)





    import arrow
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    from stuf import stuf
    from more_itertools import unzip
    from functools import partial
    # import tensorflow as tf
    # tf.compat.v1.enable_eager_execution()
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    from pyleaves.leavesdb.tf_utils.tf_utils import set_random_seed, reset_keras_session
    import pyleaves
    from pyleaves.utils.img_utils import random_pad_image
    from pyleaves.utils.utils import ensure_dir_exists
    from pyleaves.datasets import leaves_dataset, fossil_dataset, pnas_dataset, base_dataset
    from pyleaves.models.vgg16 import VGG16, VGG16GrayScale
    from pyleaves.models import resnet, vgg16
    from tensorflow.compat.v1.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
    from tensorflow.keras import metrics
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras import layers
    from tensorflow.keras import backend as K
    import tensorflow_datasets as tfds
    import neptune_tensorboard as neptune_tb

    seed = 346
    # set_random_seed(seed)
    # reset_keras_session()
    def get_preprocessing_func(model_name):
        if model_name.startswith('resnet'):
            from tensorflow.keras.applications.resnet_v2 import preprocess_input
        elif model_name == 'vgg16':
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif model_name=='shallow':
            def preprocess_input(x):
                return x/255.0 # ((x/255.0)-0.5)*2.0

        return preprocess_input #lambda x,y: (preprocess_input(x),y)

    def _load_img(image_path):#, img_size=(224,224)):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
        # return tf.compat.v1.image.resize_image_with_pad(img, *img_size)

    def _encode_label(label, num_classes=19):
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, depth=num_classes)
        return label

    def _load_example(image_path, label, num_classes=19):
        img = _load_img(image_path)
        one_hot_label = _encode_label(label, num_classes=num_classes)
        return img, one_hot_label

    def _load_uint8_example(image_path, label, num_classes=19):
        img = tf.image.convert_image_dtype(_load_img(image_path)*255.0, dtype=tf.uint8)
        one_hot_label = _encode_label(label, num_classes=num_classes)
        return img, one_hot_label

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

    def log_data(logs):
        for k, v in logs.items():
            neptune.log_metric(k, v)

    neptune_logger = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs))

    def focal_loss(gamma=2.0, alpha=4.0):

        gamma = float(gamma)
        alpha = float(alpha)

        def focal_loss_fixed(y_true, y_pred):
            """Focal loss for multi-classification
            FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
            Notice: y_pred is probability after softmax
            gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
            d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
            Focal Loss for Dense Object Detection
            https://arxiv.org/abs/1708.02002

            Arguments:
                y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
                y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

            Keyword Arguments:
                gamma {float} -- (default: {2.0})
                alpha {float} -- (default: {4.0})

            Returns:
                [tensor] -- loss.
            """
            epsilon = 1.e-9
            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)

            model_out = tf.add(y_pred, epsilon)
            ce = tf.multiply(y_true, -tf.log(model_out))
            weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
            fl = tf.multiply(alpha, tf.multiply(weight, ce))
            reduced_fl = tf.reduce_max(fl, axis=1)
            return tf.reduce_mean(reduced_fl)
        return focal_loss_fixed


    def per_class_accuracy(y_true, y_pred):
        return tf.metrics.mean_per_class_accuracy(y_true, y_pred, num_classes=PARAMS['num_classes'])

    def build_model(model_params,
                    optimizer,
                    loss,
                    METRICS):

        if model_params['name']=='vgg16':
            model_builder = vgg16.VGG16GrayScale(model_params)
        elif model_params['name'].startswith('resnet'):
            model_builder = resnet.ResNet(model_params)

        base = model_builder.build_base()
        model = model_builder.build_head(base)

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=METRICS)

        return model

    def build_shallow(input_shape=(224,224,3),
                      num_classes=10,
                      optimizer=None,
                      loss=None,
                      METRICS=None):

        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(64, (7, 7), activation='relu', input_shape=input_shape, kernel_initializer=tf.initializers.GlorotNormal()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (7, 7), activation='relu', kernel_initializer=tf.initializers.GlorotNormal()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (7, 7), activation='relu', kernel_initializer=tf.initializers.GlorotNormal()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64*2, activation='relu', kernel_initializer=tf.initializers.GlorotNormal()))
        model.add(layers.Dense(num_classes,activation='softmax', kernel_initializer=tf.initializers.GlorotNormal()))

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=METRICS)

        return model


    class ImageLogger:
        '''Tensorflow 2.0 version'''
        def __init__(self, log_dir: str, max_images: int, name: str):
            self.file_writer = tf.summary.create_file_writer(log_dir)
            self.log_dir = log_dir
            self.max_images = max_images
            self.name = name
            self._counter = tf.Variable(0, dtype=tf.int64)

            self.filepaths = []

        def add_log(self, img, counter=None, name=None):
            '''
            Intention is to generalize this to an abstract class for logging to any experiment management platform (e.g. neptune, mlflow, etc)

            Currently takes a filepath pointing to an image file and logs to current neptune experiment.
            '''

            # scaled_images = (img - tf.math.reduce_min(img))/(tf.math.reduce_max(img) - tf.math.reduce_min(img))
            # keep = 0
            # scaled_images = tf.image.convert_image_dtype(tf.squeeze(scaled_images[keep,:,:,:]), dtype=tf.uint8)
            # scaled_images = tf.expand_dims(scaled_images, 0)
            # tf.summary.image(name=self.name, data=scaled_images, step=self._counter, max_outputs=self.max_images)


            scaled_img = (img - np.min(img))/(np.max(img) - np.min(img)) * 255.0
            scaled_img = scaled_img.astype(np.uint32)

            neptune.log_image(log_name= name or self.name,
                              x=counter,
                              y=scaled_img)
            return scaled_img

        def __call__(self, images, labels):

            with self.file_writer.as_default():
                scaled_images = (images - tf.math.reduce_min(images))/(tf.math.reduce_max(images) - tf.math.reduce_min(images))
                keep = 0

                scaled_images = tf.image.convert_image_dtype(tf.squeeze(scaled_images[keep,:,:,:]), dtype=tf.uint8)
                scaled_images = tf.expand_dims(scaled_images, 0)

                labels = tf.argmax(labels[[keep], :],axis=1)
                tf.summary.image(name=self.name, data=scaled_images, step=self._counter, max_outputs=self.max_images)

                filepath = os.path.join(self.log_dir,'sample_images',f'{self.name}-{self._counter}.jpg')

                scaled_images = tf.image.encode_jpeg(tf.squeeze(scaled_images))
                tf.io.write_file(filename=tf.constant(filepath),
                                 contents=scaled_images)

            # self.add_log(scaled_images)
            self._counter.assign_add(1)
            return images, labels

    def _cond_apply(x, y, func, prob):
        """Conditionally apply func to x and y with probability prob.

        Parameters
        ----------
        x : type
            Input to conditionally pass through func
        y : type
            Label
        func : type
            Function to conditionally be applied to x and y
        prob : type
            Probability of applying function, within range [0.0,1.0]

        Returns
        -------
        x, y
        """
        return tf.cond((tf.random.uniform([], 0, 1) >= (1.0 - prob)), lambda: func(x,y), lambda: (x,y))


    class ImageAugmentor:
        """Short summary.

        Parameters
        ----------
        augmentations : dict
            Maps a sequence of named augmentations to a scalar probability,
             according to which they'll be conditionally applied in order.
        resize_w_pad : tuple, default=None
            Description of parameter `resize_w_pad`.
        random_crop :  tuple, default=None
            Description of parameter `random_crop`.
        random_jitter : dict
            First applies resize_w_pad, then random_crop. If user desires only 1 of these, set this to None.
            Should be a dict with 2 keys:
                'resize':(height, width)
                'crop_size':(crop_height,crop_width, channels)

        Only 1 of these 3 kwargs should be provided to any given augmentor:
        {'resize_w_pad', 'random_crop', 'random_jitter'}
        Example values for each:
            resize_w_pad=(224,224)
            random_crop=(224,224,3)
            random_jitter={'resize':(338,338),
                           'crop_size':(224,224, 3)}



        seed : int, default=None
            Random seed to apply to all augmentations

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        Attributes
        ----------
        augmentations

        """

        def __init__(self,
                     name='',
                     augmentations={'rotate':1.0,
                                    'flip':1.0,
                                    'color':1.0,
                                    'rgb2gray_3channel':1.0},
                     resize_w_pad=None,
                     random_crop=None,
                     random_jitter={'resize':(338,338),
                                    'crop_size':(224,224,3)},
                     log_dir=None,
                     seed=None):

            self.name = name
            self.augmentations = augmentations
            self.seed = seed

            if resize_w_pad:
                self.target_h = resize_w_pad[0]
                self.target_w = resize_w_pad[1]
                # self.resize = self.resize_w_pad
            elif random_crop:
                self.crop_size = random_crop
                self.target_h = self.crop_size[0]
                self.target_w = self.crop_size[1]
                # self.resize = self.random_crop
            elif random_jitter:
                # self.target_h = tf.random.uniform([], random_jitter['crop_size'][0], random_jitter['resize'][0], dtype=tf.int32, seed=self.seed)
                # self.target_w = tf.random.uniform([], random_jitter['crop_size'][1], random_jitter['resize'][1], dtype=tf.int32, seed=self.seed)
                self.crop_size = random_jitter['crop_size']
                # self.resize = self.random_jitter
                self.target_h = random_jitter['crop_size'][0]
                self.target_w = random_jitter['crop_size'][1]
            self.resize = self.resize_w_pad



            self.maps = {'rotate':self.rotate,
                          'flip':self.flip,
                          'color':self.color,
                          'rgb2gray_3channel':self.rgb2gray_3channel,
                          'rgb2gray_1channel':self.rgb2gray_1channel}

            self.log_dir = log_dir

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

        def rgb2gray_3channel(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
            """Convert RGB image -> grayscale image, maintain number of channels = 3

            Args:
                x,     tf.Tensor: Image
                label, tf.Tensor: arbitrary tensor, passes through unchanged

            Returns:
                Augmented image, label
            """
            x = tf.image.rgb_to_grayscale(x)
            x = tf.image.grayscale_to_rgb(x)
            return x, label

        def rgb2gray_1channel(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
            """Convert RGB image -> grayscale image, reduce number of channels from 3 -> 1

            Args:
                x,     tf.Tensor: Image
                label, tf.Tensor: arbitrary tensor, passes through unchanged

            Returns:
                Augmented image, label
            """
            x = tf.image.rgb_to_grayscale(x)
            return x, label

        def resize_w_pad(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
            # TODO Finish this
            # random_pad_image(x,min_image_size=None,max_image_size=None,pad_color=None,seed=self.seed)
            return tf.image.resize_with_pad(x, target_height=self.target_h, target_width=self.target_w), label

        def random_crop(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
            return tf.image.random_crop(x, size=self.crop_size), label

        @tf.function
        def random_jitter(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
            x, label = self.resize_w_pad(x, label)
            x, label = self.random_crop(x, label)
            return x, label

        def apply_augmentations(self, dataset: tf.data.Dataset):
            """
            Call this function to apply all of the augmentation in the order of specification
            provided to the constructor __init__() of ImageAugmentor.

            Args:
                dataset, tf.data.Dataset: must yield individual examples of form (x, y)
            Returns:
                Augmented dataset
            """

            dataset = dataset.map(self.resize, num_parallel_calls=AUTOTUNE)

            for aug_name, aug_p in self.augmentations.items():
                aug = self.maps[aug_name]
                dataset = dataset.map(lambda x,y: _cond_apply(x, y, aug, prob=aug_p), num_parallel_calls=AUTOTUNE)
                # dataset = dataset.map(lambda x,y: _cond_apply(x, y, func=aug, prob=aug_p), num_parallel_calls=AUTOTUNE)

            return dataset


    class ImageLoggerCallback(Callback):
        '''Tensorflow 2.0 version

        Callback that keeps track of a tf.data.Dataset and logs the correct batch to neptune based on the current batch.
        '''
        def __init__(self, data :tf.data.Dataset, freq=1, max_images=-1, name='', encoder=None):

            self.data = data
            self.freq = freq
            self.max_images = max_images
            self.name = name
            self.encoder=encoder
            self.init_iterator()

        def init_iterator(self):
            self.data_iter = iter(self.data)
            self._batch = 0
            self._count = 0
            self.finished = False

        def yield_batch(self):
            batch_data = next(self.data_iter)
            self._batch += 1
            self._count += batch_data[0].shape[0]
            return batch_data

        def add_log(self, img, counter=None, name=None):
            '''
            Intention is to generalize this to an abstract class for logging to any experiment management platform (e.g. neptune, mlflow, etc)

            Currently takes a filepath pointing to an image file and logs to current neptune experiment.
            '''
            scaled_img = (img - np.min(img))/(np.max(img) - np.min(img)) * 255.0
            scaled_img = scaled_img.astype(np.uint32)

            neptune.log_image(log_name= name or self.name,
                              x=counter,
                              y=scaled_img)
            return scaled_img

        def on_train_batch_begin(self, batch, logs=None):
            if batch % self.freq or self.finished:
                return
            while batch >= self._batch:
                x, y = self.yield_batch()

            if self.max_images==-1:
                self.max_images=x.shape[0]

            if x.ndim==3:
                np.newaxis(x, axis=0)
            if x.shape[0]>self.max_images:
                x = x[:self.max_images,...]
                y = y[:self.max_images,...]

            x = x.numpy()
            y = np.argmax(y.numpy(),axis=1)
            if self.encoder:
                y = self.encoder.decode(y)
            for i in range(x.shape[0]):
                # self.add_log(x[i,...], counter=i, name = f'{self.name}-{y[i]}-batch_{str(self._batch).zfill(3)}')
                self.add_log(x[i,...], counter=self._count+i, name = f'{self.name}-{y[i]}')
            print(f'Batch {self._batch}: Logged {np.max([x.shape[0],self.max_images])} {self.name} images to neptune')

        def on_epoch_end(self, epoch, logs={}):
            self.finished = True


    class ConfusionMatrixCallback(Callback):
        '''Tensorflow 2.0 version'''
        def __init__(self, log_dir, imgs : dict, labels : dict, classes, freq=1, include_train=False, seed=None):
            self.file_writer = tf.summary.create_file_writer(log_dir)
            self.log_dir = log_dir
            self.seed = seed
            self._counter = 0
            assert np.all(np.array(imgs.keys()) == np.array(labels.keys()))
            self.imgs = imgs

            for k,v in labels.items():
                if v.ndim==2:
                    labels[k] = tf.argmax(v,axis=-1)
            self.labels = labels
            self.num_samples = {k:l.numpy().shape[0] for k,l in labels.items()}
            self.classes = classes
            self.freq = freq
            self.include_train = include_train

        def log_confusion_matrix(self, model, imgs, labels, epoch, name='', norm_cm=False):

            pred_labels = model.predict_classes(imgs)
            # pred_labels = tf.argmax(pred_labels,axis=-1)
            pred_labels = pred_labels[:,None]

            con_mat = tf.math.confusion_matrix(labels=labels, predictions=pred_labels, num_classes=len(self.classes)).numpy()
            if norm_cm:
                con_mat = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
            con_mat_df = pd.DataFrame(con_mat,
                             index = self.classes,
                             columns = self.classes)

            figure = plt.figure(figsize=(12, 12))
            sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)

            with self.file_writer.as_default():
                tf.summary.image(name=name+'_confusion_matrix', data=image, step=self._counter)

            neptune.log_image(log_name=name+'_confusion_matrix',
                              x=self._counter,
                              y=figure)
            plt.close(figure)
            self._counter += 1

            return image

        def on_epoch_end(self, epoch, logs={}):

            if (not self.freq) or (epoch%self.freq != 0):
                return

            if self.include_train:
                cm_summary_image = self.log_confusion_matrix(self.model, self.imgs['train'], self.labels['train'], epoch=epoch, name='train')
            cm_summary_image = self.log_confusion_matrix(self.model, self.imgs['val'], self.labels['val'], epoch=epoch, name='val')

####################################################################################
####################################################################################
####################################################################################



    neptune.init(project_qualified_name=args.neptune_project_name)
    # neptune_tb.integrate_with_tensorflow()


    experiment_dir = '/media/data/jacob/sandbox_logs'
    experiment_name = args.experiment_name

    experiment_start_time = arrow.utcnow().format('YYYY-MM-DD_HH-mm-ss')
    log_dir =os.path.join(experiment_dir, experiment_name, 'log_dir',PARAMS['loss'], experiment_start_time)
    ensure_dir_exists(log_dir)
    print('Tensorboard log_dir: ', log_dir)
    # os.system(f'neptune tensorboard {log_dir} --project {args.neptune_project_name}')

    weights_best = os.path.join(log_dir, 'model_ckpt.h5')
    restore_best_weights=False
    histogram_freq=0
    patience=25
    num_epochs = PARAMS['num_epochs']
    initial_epoch=0

    src_db = pyleaves.DATABASE_PATH
    datasets = {
                'PNAS': pnas_dataset.PNASDataset(src_db=src_db),
                'Leaves': leaves_dataset.LeavesDataset(src_db=src_db),
                'Fossil': fossil_dataset.FossilDataset(src_db=src_db)
                }
    # data = datasets[PARAMS['dataset_name']]
    data_config = stuf(threshold=PARAMS['data_threshold'],
                       num_classes=PARAMS['num_classes']    ,
                       data_splits_meta={
                                         'train':PARAMS['train_size'],
                                         'val':PARAMS['val_size'],
                                         'test':PARAMS['test_size']
                                        }
                       )

    preprocess_input = get_preprocessing_func(PARAMS['model_name'])
    preprocess_input(tf.zeros([4, 224, 224, 3]))
    
    load_example = partial(_load_uint8_example, num_classes=data_config.num_classes)
    # load_example = partial(_load_example, num_classes=data_config.num_classes)


    if PARAMS['num_channels']==3:
        color_aug = {'rgb2gray_3channel':1.0}
    elif PARAMS['num_channels']==1:
        color_aug = {'rgb2gray_1channel':1.0}

    resize_w_pad=None
    random_jitter=None
    if not PARAMS['random_jitter']['resize']:
        resize_w_pad = PARAMS['image_size']
    else:
        random_jitter=PARAMS['random_jitter']

    TRAIN_image_augmentor = ImageAugmentor(name='train',
                                           augmentations={**PARAMS["augmentations"],
                                                          **color_aug},#'rotate':1.0,'flip':1.0,**color_aug},
                                           resize_w_pad=resize_w_pad,
                                           random_crop=None,
                                           random_jitter=random_jitter,
                                           log_dir=log_dir,
                                           seed=None)
    VAL_image_augmentor = ImageAugmentor(name='val',
                                         augmentations={**color_aug},
                                         resize_w_pad=PARAMS['image_size'],
                                         random_crop=None,
                                         random_jitter=None,
                                         log_dir=log_dir,
                                         seed=None)
    TEST_image_augmentor = ImageAugmentor(name='test',
                                          augmentations={**color_aug},
                                          resize_w_pad=PARAMS['image_size'],
                                          random_crop=None,
                                          random_jitter=None,
                                          log_dir=log_dir,
                                          seed=None)


    def neptune_log_augmented_images(split_data, num_demo_samples=40, PARAMS=PARAMS):
        num_demo_samples = 40
        cm_data_x = {'train':[],'val':[]}
        cm_data_y = {'train':[],'val':[]}
        cm_data_x['train'], cm_data_y['train'] = next(iter(get_data_loader(data=split_data['train'], data_subset_mode='train', batch_size=num_demo_samples, infinite=True, augment=False,seed=2836)))
        cm_data_x['val'], cm_data_y['val'] = next(iter(get_data_loader(data=split_data['val'], data_subset_mode='val', batch_size=num_demo_samples, infinite=True, augment=False, seed=2836)))

        for (k_x,v_x), (k_y, v_y) in zip(cm_data_x.items(), cm_data_y.items()):
            x = tf.data.Dataset.from_tensor_slices(v_x)
            y = tf.data.Dataset.from_tensor_slices(v_y)
            xy_data = tf.data.Dataset.zip((x, y))
            v = xy_data.map(VAL_image_augmentor.resize, num_parallel_calls=AUTOTUNE)
            v_aug = TRAIN_image_augmentor.apply_augmentations(xy_data)
            v_x, v_y = [i.numpy() for i in next(iter(v.batch(10*num_demo_samples)))]
            v_x_aug, v_y_aug = [i.numpy() for i in next(iter(v_aug.batch(10*num_demo_samples)))]
            k = k_x
            for i in range(num_demo_samples):
                print(f'Neptune: logging {k}_{i}')
                print(f'{v_x[i].shape}, {v_x_aug[i].shape}')
                idx = np.random.randint(0,len(v_x))
                if True: #'train' in k:
                    TRAIN_image_augmentor.logger.add_log(v_x[idx],counter=i, name=k)
                    TRAIN_image_augmentor.logger.add_log(v_x_aug[idx],counter=i, name=k+'_aug')


    def get_data_loader(data : tuple, data_subset_mode='train', batch_size=32, num_classes=None, infinite=True, augment=True, seed=2836):

        num_samples = len(data[0])
        x = tf.data.Dataset.from_tensor_slices(data[0])
        labels = tf.data.Dataset.from_tensor_slices(data[1])
        data = tf.data.Dataset.zip((x, labels))

        data = data.cache()
        if data_subset_mode == 'train':
            data = data.shuffle(buffer_size=num_samples)

        # data = data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=tf.uint8),y), num_parallel_calls=-1)
        # data = data.map(load_example, num_parallel_calls=AUTOTUNE)
        data = data.map(load_example, num_parallel_calls=AUTOTUNE)


        data = data.map(lambda x,y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

        if infinite:
            data = data.repeat()

        if data_subset_mode == 'train':
            data = data.shuffle(buffer_size=200, seed=seed)
            augmentor = TRAIN_image_augmentor
        elif data_subset_mode == 'val':
            augmentor = VAL_image_augmentor
        elif data_subset_mode == 'test':
            augmentor = TEST_image_augmentor

        if augment:
            data = augmentor.apply_augmentations(data)

        data = data.batch(batch_size, drop_remainder=True)

        return data.prefetch(AUTOTUNE)

    def get_tfds_data_loader(data : tf.data.Dataset, data_subset_mode='train', batch_size=32, num_samples=100, num_classes=19, infinite=True, augment=True, seed=2836):


        def encode_example(x, y):
            x = tf.image.convert_image_dtype(x, tf.float32) * 255.0
            y = _encode_label(y, num_classes=num_classes)
            return x, y

        test_d = next(iter(data))
        print(test_d[0].numpy().min())
        print(test_d[0].numpy().max())

        data = data.shuffle(buffer_size=num_samples) \
                   .cache() \
                   .map(encode_example, num_parallel_calls=AUTOTUNE)

        test_d = next(iter(data))
        print(test_d[0].numpy().min())
        print(test_d[0].numpy().max())

        data = data.map(preprocess_input, num_parallel_calls=AUTOTUNE)

        test_d = next(iter(data))
        print(test_d[0].numpy().min())
        print(test_d[0].numpy().max())

        if data_subset_mode == 'train':
            data = data.shuffle(buffer_size=100, seed=seed)
            augmentor = TRAIN_image_augmentor
        elif data_subset_mode == 'val':
            augmentor = VAL_image_augmentor
        elif data_subset_mode == 'test':
            augmentor = TEST_image_augmentor

        if augment:
            data = augmentor.apply_augmentations(data)

        test_d = next(iter(data))
        print(test_d[0].numpy().min())
        print(test_d[0].numpy().max())

        data = data.batch(batch_size, drop_remainder=True)
        if infinite:
            data = data.repeat()

        return data.prefetch(AUTOTUNE)






    # y_true = [[0, 1, 0], [0, 0, 1]]
    # y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

    def accuracy(y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)

        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


    def true_pos(y_true, y_pred):
        # y_true = K.ones_like(y_true)
        return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    def false_pos(y_true, y_pred):
        # y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return all_positives - true_positives

    def true_neg(y_true, y_pred):
        # y_true = K.ones_like(y_true)
        return K.sum(1-K.round(K.clip(y_true * y_pred, 0, 1)))

    def recall(y_true, y_pred):
        # y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (all_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        y_true = K.ones_like(y_true)

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        # tf.print(y_true, y_pred)
        return precision

    def f1_score(y_true, y_pred):
        m_precision = precision(y_true, y_pred)
        m_recall = recall(y_true, y_pred)
        # pdb.set_trace()
        return 2*((m_precision*m_recall)/(m_precision+m_recall+K.epsilon()))

    # def false_neg(y_true, y_pred):
    #     y_true = K.ones_like(~y_true)
    #     true_neg = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     all_negative = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     return all_negatives - true_

        # return K.mean(K.argmax(y_true,axis=1)*K.argmax(y_pred,axis=1))

        # 'accuracy',
        # metrics.TrueNegatives(name='tn'),
        # metrics.FalseNegatives(name='fn'),
    METRICS = [
        f1_score,
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.CategoricalAccuracy(name='accuracy'),
        metrics.TopKCategoricalAccuracy(name='top_3_categorical_accuracy', k=3),
        metrics.TopKCategoricalAccuracy(name='top_5_categorical_accuracy', k=5)
    ]
    PARAMS['sys.argv'] = ' '.join(sys.argv)

    with neptune.create_experiment(name=experiment_name, params=PARAMS, upload_source_files=[__file__]):


        print('Logging experiment tags:')
        for tag in args.tags:
            print(tag)
            neptune.append_tag(tag)

        neptune.append_tag(PARAMS['dataset_name'])
        neptune.append_tag(PARAMS['model_name'])
        neptune.log_artifact(args.config_path)
        cm_data_x = {'train':[],'val':[]}
        cm_data_y = {'train':[],'val':[]}

        if PARAMS['dataset_name'] in tfds.list_builders():
            num_demo_samples=40

            tfds_builder = tfds.builder(PARAMS['dataset_name'])
            tfds_builder.download_and_prepare()

            num_samples = tfds_builder.info.splits['train'].num_examples
            num_samples_dict = {'train':int(num_samples*PARAMS['train_size']),
                            'val':int(num_samples*PARAMS['val_size']),
                            'test':int(num_samples*PARAMS['test_size'])}

            classes = tfds_builder.info.features['label'].names
            num_classes = len(classes)

            train_slice = [0,int(PARAMS['train_size']*100)]
            val_slice = [int(PARAMS['train_size']*100), int((PARAMS['train_size']+PARAMS['val_size'])*100)]
            test_slice = [100 - int(PARAMS['test_size']*100), 100]

            tfds_train_data = tfds.load(PARAMS['dataset_name'], split=f"train[{train_slice[0]}%:{train_slice[1]}%]", shuffle_files=True, as_supervised=True)
            tfds_validation_data = tfds.load(PARAMS['dataset_name'], split=f"train[{val_slice[0]}%:{val_slice[1]}%]", shuffle_files=True, as_supervised=True)
            tfds_test_data = tfds.load(PARAMS['dataset_name'], split=f"train[{test_slice[0]}%:{test_slice[1]}%]", shuffle_files=True, as_supervised=True)

            # PARAMS['batch_size']=1
            train_data = get_tfds_data_loader(data = tfds_train_data, data_subset_mode='train', batch_size=PARAMS['batch_size'], num_samples=num_samples_dict['train'], num_classes=num_classes, infinite=True, augment=True, seed=2836)
            validation_data = get_tfds_data_loader(data = tfds_validation_data, data_subset_mode='val', batch_size=PARAMS['batch_size'], num_samples=num_samples_dict['val'], num_classes=num_classes, infinite=True, augment=True, seed=2837)
            test_data = get_tfds_data_loader(data = tfds_test_data, data_subset_mode='test', batch_size=PARAMS['batch_size'], num_samples=num_samples_dict['test'], num_classes=num_classes, infinite=True, augment=True, seed=2838)

            # tfds_train_data = tfds.load(PARAMS['dataset_name'], split=f"train[{train_slice[0]}%:{train_slice[1]}%]", shuffle_files=True, as_supervised=True)
            # tfds_validation_data = tfds.load(PARAMS['dataset_name'], split=f"train[{val_slice[0]}%:{val_slice[1]}%]", shuffle_files=True, as_supervised=True)
            # tfds_test_data = tfds.load(PARAMS['dataset_name'], split=f"train[{test_slice[0]}%:{test_slice[1]}%]", shuffle_files=True, as_supervised=True)

            split_data = {'train':get_tfds_data_loader(data = tfds_train_data, data_subset_mode='train', batch_size=num_demo_samples, num_samples=num_samples_dict['train'], num_classes=num_classes, infinite=True, augment=True, seed=2836),
                          'val':get_tfds_data_loader(data = tfds_validation_data, data_subset_mode='val', batch_size=num_demo_samples, num_samples=num_samples_dict['val'], num_classes=num_classes, infinite=True, augment=True, seed=2837),
                          'test':get_tfds_data_loader(data = tfds_test_data, data_subset_mode='test', batch_size=num_demo_samples, num_samples=num_samples_dict['test'], num_classes=num_classes, infinite=True, augment=True, seed=2838)
                          }

            steps_per_epoch=num_samples_dict['train']//PARAMS['batch_size']
            validation_steps=num_samples_dict['val']//PARAMS['batch_size']

            cm_data_x['train'], cm_data_y['train'] = next(iter(split_data['train']))
            cm_data_x['val'], cm_data_y['val'] = next(iter(split_data['val']))

        else:
            data = datasets[PARAMS['dataset_name']]
            neptune.set_property('num_classes',data.num_classes)
            neptune.set_property('class_distribution',data.metadata.class_distribution)

            encoder = base_dataset.LabelEncoder(data.data.family)
            split_data = base_dataset.preprocess_data(data, encoder, data_config)
            # import pdb;pdb.set_trace()
            for subset, subset_data in split_data.items():
                split_data[subset] = [list(i) for i in unzip(subset_data)]

            PARAMS['batch_size'] = 32

            steps_per_epoch=len(split_data['train'][0])//PARAMS['batch_size']#//10
            validation_steps=len(split_data['val'][0])//PARAMS['batch_size']#//10

            split_datasets = {
                              k:base_dataset.BaseDataset.from_dataframe(
                                pd.DataFrame({
                                            'path':v[0],
                                            'family':v[1]
                                            })) \
                              for k,v in split_data.items()
                             }

            for k,v in split_datasets.items():
                print(k, v.num_classes)

            classes = split_datasets['train'].classes

            train_data=get_data_loader(data=split_data['train'], data_subset_mode='train', batch_size=PARAMS['batch_size'], infinite=True, augment=True, seed=2836)
            validation_data=get_data_loader(data=split_data['val'], data_subset_mode='val', batch_size=PARAMS['batch_size'], infinite=True, augment=True, seed=2837)
            if 'test' in split_data.keys():
                test_data=get_data_loader(data=split_data['test'], data_subset_mode='test', batch_size=PARAMS['batch_size'], infinite=True, augment=True, seed=2838)

            num_demo_samples=150
            # neptune_log_augmented_images(split_data, num_demo_samples=num_demo_samples, PARAMS=PARAMS)
            cm_data_x['train'], cm_data_y['train'] = next(iter(get_data_loader(data=split_data['train'], data_subset_mode='train', batch_size=num_demo_samples, infinite=True, augment=True, seed=2836)))
            cm_data_x['val'], cm_data_y['val'] = next(iter(get_data_loader(data=split_data['val'], data_subset_mode='val', batch_size=num_demo_samples, infinite=True, augment=True,  seed=2836)))


        ########################################################################################
        train_image_logger_cb = ImageLoggerCallback(data=train_data, freq=20, max_images=-1, name='train', encoder=encoder)
        val_image_logger_cb = ImageLoggerCallback(data=validation_data, freq=20, max_images=-1, name='val', encoder=encoder)
        ########################################################################################

        cm_callback = ConfusionMatrixCallback(log_dir, cm_data_x, cm_data_y, classes=classes, seed=PARAMS['seed'], include_train=True)
        checkpoint = ModelCheckpoint(weights_best, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min',restore_best_weights=restore_best_weights)
        tfboard = TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, write_images=True)
        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
        callbacks = [checkpoint,tfboard,early, cm_callback, neptune_logger, train_image_logger_cb, val_image_logger_cb]
    ##########################
        if PARAMS['optimizer'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=PARAMS['lr']
            )
        elif PARAMS['optimizer'] == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate=PARAMS['lr']
            )
        elif PARAMS['optimizer'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=PARAMS['lr']
            )
    ##########################
        if PARAMS['loss']=='focal_loss':
            loss = focal_loss(gamma=2.0, alpha=4.0)
        elif PARAMS['loss']=='categorical_crossentropy':
            loss = 'categorical_crossentropy'
    ##########################
        model_params = stuf(name=PARAMS['model_name'],
                            model_dir=os.path.join(experiment_dir, experiment_name, 'models'),
                            num_classes=PARAMS['num_classes'],
                            frozen_layers = PARAMS['frozen_layers'],
                            input_shape = (*PARAMS['image_size'],PARAMS['num_channels']),
                            base_learning_rate = PARAMS['lr'],
                            regularization = PARAMS['regularization'])
    ####
        if PARAMS['model_name']=='shallow':
            model = build_shallow(input_shape=model_params.input_shape,
                                  num_classes=PARAMS['num_classes'],
                                  optimizer=optimizer,
                                  loss=loss,
                                  METRICS=METRICS)

        else:
            model = build_model(model_params,
                                optimizer,
                                loss,
                                METRICS)
        print(f"TRAINING {PARAMS['model_name']}")

        model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

        history = model.fit(train_data,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            validation_data=validation_data,
                            shuffle=True,
                            initial_epoch=initial_epoch,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps)


        if 'test' in split_data:
            results = model.evaluate(test_data,
                                    steps=len(split_data['test'][0]))
        else:
            results = model.evaluate(validation_data,
                                    steps=validation_steps)


if __name__=='__main__':
    main()

# print(x_train.min(),x_train.max())
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test, y_test)
