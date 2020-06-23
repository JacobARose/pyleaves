# @Author: Jacob A Rose
# @Date:   Wed, May 27th 2020, 1:39 am
# @Email:  jacobrose@brown.edu
# @Filename: triplet_train.py
'''

'''

import os
# gpu = 7
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
import tensorflow as tf

import arrow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from stuf import stuf
from more_itertools import unzip
from functools import partial

# tf.compat.v1.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

from pyleaves.train.callbacks import ConfusionMatrixCallback
from pyleaves.leavesdb.tf_utils.tf_utils import set_random_seed, reset_keras_session
import pyleaves
from pyleaves.datasets import leaves_dataset, fossil_dataset, pnas_dataset, base_dataset
from pyleaves.models.vgg16 import VGG16, VGG16GrayScale

from tensorflow.compat.v1.keras.callbacks import (Callback,
                                                  ModelCheckpoint,
                                                  TensorBoard,
                                                  LearningRateScheduler,
                                                  EarlyStopping)
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from pyleaves.models import resnet, vgg16

import neptune
import neptune_tensorboard as neptune_tb

seed = 345
set_random_seed(seed)
# reset_keras_session()

def get_preprocessing_func(model_name):
    if model_name.startswith('resnet'):
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif model_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif model_name == 'shallow':
        def preprocess_input(x):
            return ((x/255.0)-0.5)*2.0

    return lambda x, y: (preprocess_input(x),y)

def _load_img(image_path, img_size=(224,224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.compat.v1.image.resize_image_with_pad(img, *img_size)

def _encode_label(label, num_classes=19):
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=num_classes)
    return label

def _load_example(image_path, label, img_size=(224,224), num_classes=19):
    img = _load_img(image_path, img_size=img_size)
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



if False
    # set project and start integration with keras
    neptune.init(project_qualified_name='jacobarose/sandbox')#minimal_working_examples')
    neptune_tb.integrate_with_tensorflow()
    # neptune_tb.integrate_with_keras()
    experiment_dir = '/media/data/jacob/sandbox_logs'
    experiment_name = 'pnas_minimal_example'

    # parameters
    PARAMS = {'gpu':7,
              'num_epochs': 20,
              'batch_size': 32,
              'optimizer':'Adam',
              'loss':'categorical_crossentropy', #'focal_loss',
              'lr': 0.0001,
              'model_name':'resnet_50_v2', #'vgg16', #'shallow',
              'dataset_name':'PNAS',
              'regularization':{'l1':1e-4}, #None
              'num_classes':19,
              'image_size': (224,224),
              'num_channels':3,
              'train_size':0.4,
              'val_size':0.1,
              'test_size':0.5,
              'data_threshold':0,
              'seed':29384
             }

    #CALLBACKS
    experiment_start_time = arrow.utcnow().format('YYYY-MM-DD_HH-mm-ss')
    log_dir =os.path.join(experiment_dir, experiment_name, 'log_dir',PARAMS['loss'], experiment_start_time)
    weights_best = os.path.join(log_dir, 'model_ckpt.h5')
    restore_best_weights=False
    histogram_freq=0
    patience=25

    num_epochs = PARAMS['num_epochs']
    shuffle=True
    initial_epoch=0

    src_db = pyleaves.DATABASE_PATH
    datasets = {
                'PNAS': pnas_dataset.PNASDataset(src_db=src_db),
                'Leaves': leaves_dataset.LeavesDataset(src_db=src_db),
                'Fossil': fossil_dataset.FossilDataset(src_db=src_db)
                }
    data = datasets[PARAMS['dataset_name']]
    data_config = stuf(threshold=0,
                       data_splits_meta={
                                         'train':PARAMS['train_size'],
                                         'val':PARAMS['val_size'],
                                         'test':PARAMS['test_size']
                                        }
                       )

    preprocess_input = get_preprocessing_func(PARAMS['model_name'])
    preprocess_input(tf.zeros([4, 32, 32, 3]), tf.zeros([4, 32]))
    load_example = partial(_load_example, img_size=PARAMS['image_size'], num_classes=data.num_classes)



    # class ConfusionMatrixCallback(Callback):
    #
    #     def __init__(self, log_dir, val_imgs, val_labels, classes, freq=1, seed=None):
    #         self.file_writer = tf.contrib.summary.create_file_writer(log_dir)
    #         self.log_dir = log_dir
    #         self.seed = seed
    #         self._counter = 0
    #         self.val_imgs = val_imgs
    #
    #         if val_labels.ndim==2:
    #             val_labels = tf.argmax(val_labels,axis=1)
    #         self.val_labels = val_labels
    #         self.num_samples = val_labels.numpy().shape[0]
    #         self.classes = classes
    #         self.freq = freq
    #
    #     def log_confusion_matrix(self, model, imgs, labels, epoch, norm_cm=False):
    #
    #         pred_labels = model.predict_classes(imgs)# = tf.reshape(imgs, (-1,PARAMS['image_size'], PARAMS['num_channels'])))
    #         pred_labels = pred_labels[:,None]
    #
    #         con_mat = tf.math.confusion_matrix(labels=labels, predictions=pred_labels, num_classes=len(self.classes)).numpy()
    #         if norm_cm:
    #             con_mat = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    #         con_mat_df = pd.DataFrame(con_mat,
    #                          index = self.classes,
    #                          columns = self.classes)
    #
    #         figure = plt.figure(figsize=(16, 16))
    #         sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    #         plt.tight_layout()
    #         plt.ylabel('True label')
    #         plt.xlabel('Predicted label')
    #
    #         buf = io.BytesIO()
    #         plt.savefig(buf, format='png')
    #         buf.seek(0)
    #
    #         image = tf.image.decode_png(buf.getvalue(), channels=4)
    #         image = tf.expand_dims(image, 0)
    #
    #         with self.file_writer.as_default(), tf.contrib.summary.always_record_summaries():
    #             tf.contrib.summary.image(name='val_confusion_matrix',
    #                                      tensor=image,
    #                                      step=self._counter)
    #
    #         neptune.log_image(log_name='val_confusion_matrix',
    #                           x=self._counter,
    #                           y=figure)
    #         plt.close(figure)
    #
    #         self._counter += 1
    #
    #         return image
    #
    #     def on_epoch_end(self, epoch, logs={}):
    #
    #         if (not self.freq) or (epoch%self.freq != 0):
    #             return
    #
    #         cm_summary_image = self.log_confusion_matrix(self.model, self.val_imgs, self.val_labels, epoch=epoch)

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
        model.add(layers.Conv2D(64, (7, 7), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (7, 7), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (7, 7), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64*2, activation='relu'))
        model.add(layers.Dense(num_classes,activation='softmax'))

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=METRICS)

        return model


    # In[4]:


    class ImageDataLogger:
        def __init__(self, log_dir: str, max_images: int, name: str, augment=True):
            self.file_writer = tf.contrib.summary.create_file_writer(log_dir)
            self.max_images = max_images
            self.name = name
            self.augment = augment
            self._counter = 0

        def __call__(self, images, labels):
            if self.augment:
                images = tf.image.random_flip_left_right(images)
                images = tf.image.random_flip_up_down(images)

            with self.file_writer.as_default(), tf.contrib.summary.always_record_summaries():
                scaled_images = (images - tf.math.reduce_min(images))/(tf.math.reduce_max(images) - tf.math.reduce_min(images))
                tf.contrib.summary.image(name=self.name,tensor=scaled_images,step=self._counter,max_images=self.max_images)
            self._counter += 1
            return images, labels

    TRAIN_image_augmentor = ImageDataLogger(log_dir=log_dir, max_images=4, name='train', augment=True)#False)
    VAL_image_augmentor = ImageDataLogger(log_dir=log_dir, max_images=4, name='val', augment=False)
    TEST_image_augmentor = ImageDataLogger(log_dir=log_dir, max_images=4, name='test', augment=False)

    def get_data_loader(data : tuple, data_subset_mode='train', batch_size=32, num_channels=1, infinite=True, seed=2836):

        paths = tf.data.Dataset.from_tensor_slices(data[0])
        labels = tf.data.Dataset.from_tensor_slices(data[1])
        data = tf.data.Dataset.zip((paths, labels)) \
                              .map(load_example, num_parallel_calls=AUTOTUNE) \
                              .map(preprocess_input, num_parallel_calls=AUTOTUNE)

        if num_channels==3:
            data = data.map(rgb2gray_3channel, num_parallel_calls=AUTOTUNE)
        elif num_channels==1:
            data = data.map(rgb2gray_1channel, num_parallel_calls=AUTOTUNE)

        if data_subset_mode == 'train':
            data = data.shuffle(buffer_size=2000, seed=seed) \
                       .batch(batch_size, drop_remainder=True) \
                       .map(TRAIN_image_augmentor)
        elif data_subset_mode == 'val':
            data = data.batch(batch_size, drop_remainder=True) \
                       .map(VAL_image_augmentor)
        elif data_subset_mode == 'test':
            data = data.batch(batch_size, drop_remainder=True) \
                       .map(TEST_image_augmentor)

        if infinite:
            data = data.repeat()

        return data.prefetch(AUTOTUNE)

    METRICS = [
    #     per_class_accuracy,
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.CategoricalAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.TopKCategoricalAccuracy(name='top_3_categorical_accuracy', k=3),
        metrics.TopKCategoricalAccuracy(name='top_5_categorical_accuracy', k=5)
    ]
    ###########################################################################
    ###########################################################################


    encoder = base_dataset.LabelEncoder(data.data.family)
    split_data = base_dataset.preprocess_data(data, encoder, data_config)
    for subset, subset_data in split_data.items():
        split_data[subset] = [list(i) for i in unzip(subset_data)]

    steps_per_epoch=len(split_data['train'][0])//PARAMS['batch_size']
    validation_steps=len(split_data['val'][0])//PARAMS['batch_size']

    # split_datasets = {
    #                   k:base_dataset.BaseDataset \
    #                                 .from_dataframe(
    #                                                 pd.DataFrame({
    #                                                             'path':v[0],
    #                                                             'family':v[1]
    #                                                             })) \
    #                   for k,v in split_data.items()
    #                  }


    with neptune.create_experiment(name=experiment_name, params=PARAMS):

        neptune.set_property('num_classes',data.num_classes)
        neptune.set_property('class_distribution',data.metadata.class_distribution)

    ##########################
        train_data=get_data_loader(data=split_data['train'], data_subset_mode='train', batch_size=PARAMS['batch_size'], num_channels=PARAMS['num_channels'], infinite=True, seed=2836)
        validation_data=get_data_loader(data=split_data['val'], data_subset_mode='val', batch_size=PARAMS['batch_size'], num_channels=PARAMS['num_channels'], infinite=True, seed=2836)
        train_batch = next(iter(train_data))
        train_images, train_labels = train_batch[0].numpy(), train_batch[1].numpy()
        print(train_images.min(), train_images.max())
        plt.imshow(train_images[5,:,:,:].squeeze())

    ##########################
        num_val_samples = len(split_data['val'][0])
        cm_val_data_loader = iter(get_data_loader(data=split_data['val'], data_subset_mode='val', batch_size=num_val_samples, num_channels=PARAMS['num_channels'], infinite=True, seed=2836))# \
        cm_val_imgs, cm_val_labels = next(cm_val_data_loader)
        cm_callback = ConfusionMatrixCallback(log_dir, cm_val_imgs, cm_val_labels, classes=data.classes, seed=PARAMS['seed'])
    ####
        checkpoint = ModelCheckpoint(weights_best, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min',restore_best_weights=restore_best_weights)
        tfboard = TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, write_images=True)
        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
        callbacks = [checkpoint,tfboard,early, cm_callback, neptune_logger]
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
                            frozen_layers = None,
                            input_shape = (*PARAMS['image_size'],PARAMS['num_channels']),
                            base_learning_rate = PARAMS['lr'],
                            regularization = PARAMS['regularization'])
    ####
        if PARAMS['model_name']=='shallow':
            model = build_shallow(input_shape=(224,224,1),
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


        history = model.fit(train_data,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            validation_data=validation_data,
                            shuffle=True,
                            initial_epoch=initial_epoch,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps)





    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
























































    #coding=utf-8
    # import os
    # import sys
    # import tensorflow as tf
    # from keras.optimizers import *
    # from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
    # from keras.losses import categorical_crossentropy
    #
    # from utils import *
    # from data_loader import *
    # from model import *
    # from torchvision.transforms import *
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #
    #
    # train_samples = './txt/train.txt'
    # test_samples = './txt/test.txt'
    #
    #
    # def main(save_path, date):
    #
    #     ############################################################
    #     # tripletnet,need_save_model = TripletNet(margin=0.5, dis='cos')
    #     tripletnet,need_save_model = TripletNet(margin=1, dis='euclidean')
    #     tripletnet.summary()
    #
    #     adam = Adam(lr=1e-3, decay=0.7)
    #     sgd = SGD(lr=0.0001, momentum=0.9, decay=0.7)
    #     adroms = RMSprop(lr=0.001)
    #     tripletnet.compile(optimizer=adam, loss=None)
    #
    #
    #
    #     trainTrans = Compose([Resize(size=(224,224))])
    #     train = dataLoader(train_samples, batch_size=24, tag='train', transforms=trainTrans)
    #
    #     testTrans = Compose([Resize(size=(224,224))])
    #     test = dataLoader(test_samples, batch_size=24, tag='test', transforms=testTrans)
    #     modelcheck = MyModelCheckpoint('{}/{}.hdf5'.format(save_path, date),
    #                                     need_save_model,
    #                                     save_best_only=True,
    #                                     save_weights_only=False)
    #
    #
    #
    #     # rdshechudle = ReduceLROnPlateau(factor=0.75,
    #     #                                 patience=3,
    #     #                                 min_lr=1e-6)
    #     rdshechudle = LearningRateScheduler(lambda epoch, lr: lr / (epoch + 1)**2)
    #
    #     tripletnet.fit_generator(train,
    #             steps_per_epoch=train.__len__(),
    #             epochs=100,
    #             callbacks=[modelcheck, rdshechudle],
    #             validation_data=test,
    #             validation_steps=test.__len__(),
    #             use_multiprocessing=True,
    #             workers=8,
    #             max_queue_size=20
    #             )
    #
    # if __name__ == '__main__':
    #     save_path = sys.argv[1]
    #     date = sys.argv[2]
    #     main(save_path, date)
