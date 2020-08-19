# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: callbacks.py

<<<<<<< HEAD
<<<<<<< HEAD
'''
DEPRECATED (7/22/2020) - Jacob Rose
Action: Moved to pyleaves.utils.callback_utils.py
Reason: Cleaning up repo organization
'''


=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
=======
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
import io
import os

# import datetime
import random
# from tensorflow import keras
import tensorflow as tf
# import tfmpl
from tensorflow.compat.v1.keras.callbacks import (Callback,
                                                  CSVLogger,
                                                  ModelCheckpoint,
                                                  TensorBoard,
                                                  LearningRateScheduler,
                                                  EarlyStopping)

# from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc

import neptune
import cv2
from mlxtend import evaluate, plotting
from pyleaves.models.vgg16 import visualize_activations#, get_layers_by_index

<<<<<<< HEAD
<<<<<<< HEAD



=======



>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
class BaseCallback(Callback):

=======



class BaseCallback(Callback):

>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e
    def __init__(self, log_dir, seed = None):
        super().__init__()
        self.log_dir = log_dir
        self.seed = seed
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)


    def write_image_summary(self, input_image, title='images', epoch=0):
        image_summary = tf.summary.image(title, input_image)
        img_sum = self.sess.run(image_summary)
        self.writer.add_summary(img_sum, global_step=epoch)
        print('Finished image summary writing')

    def get_batch(self):
        batch = self.sess.run(self.validation_data)
        return batch

    def get_random_batch(self, max=40):
        i = random.randint(1,max)
        for _ in range(i):
            batch = self.get_batch()
        return batch

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.compat.v1.name_scope('summaries'):
            mean = tf.reduce_mean(input_tensor=var)
            tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed=None):
        self._seed = seed or random.randint(-1e5,1e5)
        random.seed(self.seed)






class NeptuneVisualizationCallback(Callback):
    """Callback for logging to Neptune.

    1. on_batch_end: Logs all performance metrics to neptune
    2. on_epoch_end: Logs all performance metrics + confusion matrix + roc plot to neptune

    Parameters
    ----------
    model : type
        Description of parameter `model`.
    validation_data : list
        len==2 list containing [images, labels]
    image_dir : type
        Description of parameter `image_dir`.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    Attributes
    ----------
    model
    validation_data

    """
    def __init__(self, validation_data):
        super().__init__()
        # self.model = model
        self.validation_data = validation_data

    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'batch_{log_name}', log_value)

<<<<<<< HEAD
    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'epoch_{log_name}', log_value)

        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]

        y_pred_class = np.argmax(y_pred, axis=1)

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        neptune.log_image('confusion_matrix', fig)

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        neptune.log_image('roc_curve', fig)










class ConfusionMatrixCallback(Callback):

    def __init__(self, log_dir, val_imgs, val_labels, classes, freq=1, seed=None):
        super().__init__()
        self.file_writer = tf.contrib.summary.create_file_writer(log_dir)
        self.log_dir = log_dir
        self.seed = seed
        self._counter = 0
        self.val_imgs = val_imgs

        if val_labels.ndim==2:
            val_labels = tf.argmax(val_labels,axis=1)
        self.val_labels = val_labels
        self.num_samples = val_labels.numpy().shape[0]
        self.classes = classes
        self.freq = freq

    def log_confusion_matrix(self, model, imgs, labels, epoch, norm_cm=False):

        pred_labels = model.predict_classes(imgs)# = tf.reshape(imgs, (-1,PARAMS['image_size'], PARAMS['num_channels'])))
        pred_labels = pred_labels[:,None]

        con_mat = tf.math.confusion_matrix(labels=labels, predictions=pred_labels, num_classes=len(self.classes)).numpy()
        if norm_cm:
            con_mat = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        con_mat_df = pd.DataFrame(con_mat,
                         index = self.classes,
                         columns = self.classes)

        figure = plt.figure(figsize=(16, 16))
        sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self.file_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image(name='val_confusion_matrix',
                                     tensor=image,
                                     step=self._counter)

        neptune.log_image(log_name='val_confusion_matrix',
                          x=self._counter,
                          y=figure)
        plt.close(figure)

        self._counter += 1

        return image

    def on_epoch_end(self, epoch, logs={}):

        if (not self.freq) or (epoch%self.freq != 0):
            return
        self.log_confusion_matrix(self.model, self.val_imgs, self.val_labels, epoch=epoch)


################################################

class VisualizeActivationsCallback(BaseCallback):

    def __init__(self, val_data, log_dir, freq=10, seed=None, sess=None):
        super().__init__(log_dir=log_dir, seed=seed)
        self.graph = tf.get_default_graph() #self.sess.graph #
        self.sess = sess or tf.Session()
        self.validation_iterator = val_data.make_one_shot_iterator()
        self.validation_data = self.validation_iterator.get_next()
        self.freq = freq

    def write_activation_summaries(self, model, input_images, input_labels, epoch=0):
        if type(model.layers[0])==tf.python.keras.engine.training.Model:
            #In case model was constructed from a base model
            model = model.layers[0]

        activation_grids_list = visualize_activations(input_images, model, self.sess, self.graph, group='vgg16_conv_block_outputs')[:1]
        i=0
        for activation_grid in activation_grids_list:
            # layer_name =  get_layers_by_index(model,[i])[0].name
            layer_name, grid_summary = self.sess.run(*activation_grid)
            self.write_image_summary(grid_summary, title=f'Layer : {layer_name}', epoch=epoch)

    def on_epoch_end(self, epoch, logs={}):
=======
    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'epoch_{log_name}', log_value)

        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]

        y_pred_class = np.argmax(y_pred, axis=1)

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        neptune.log_image('confusion_matrix', fig)

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        neptune.log_image('roc_curve', fig)










class ConfusionMatrixCallback(Callback):

    def __init__(self, log_dir, val_imgs, val_labels, classes, freq=1, seed=None):
        super().__init__()
        self.file_writer = tf.contrib.summary.create_file_writer(log_dir)
        self.log_dir = log_dir
        self.seed = seed
        self._counter = 0
        self.val_imgs = val_imgs

        if val_labels.ndim==2:
            val_labels = tf.argmax(val_labels,axis=1)
        self.val_labels = val_labels
        self.num_samples = val_labels.numpy().shape[0]
        self.classes = classes
        self.freq = freq

    def log_confusion_matrix(self, model, imgs, labels, epoch, norm_cm=False):

        pred_labels = model.predict_classes(imgs)# = tf.reshape(imgs, (-1,PARAMS['image_size'], PARAMS['num_channels'])))
        pred_labels = pred_labels[:,None]

        con_mat = tf.math.confusion_matrix(labels=labels, predictions=pred_labels, num_classes=len(self.classes)).numpy()
        if norm_cm:
            con_mat = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        con_mat_df = pd.DataFrame(con_mat,
                         index = self.classes,
                         columns = self.classes)

        figure = plt.figure(figsize=(16, 16))
        sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self.file_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image(name='val_confusion_matrix',
                                     tensor=image,
                                     step=self._counter)

        neptune.log_image(log_name='val_confusion_matrix',
                          x=self._counter,
                          y=figure)
        plt.close(figure)

        self._counter += 1

        return image

    def on_epoch_end(self, epoch, logs={}):

        if (not self.freq) or (epoch%self.freq != 0):
            return
        self.log_confusion_matrix(self.model, self.val_imgs, self.val_labels, epoch=epoch)


################################################

class VisualizeActivationsCallback(BaseCallback):

    def __init__(self, val_data, log_dir, freq=10, seed=None, sess=None):
        super().__init__(log_dir=log_dir, seed=seed)
        self.graph = tf.get_default_graph() #self.sess.graph #
        self.sess = sess or tf.Session()
        self.validation_iterator = val_data.make_one_shot_iterator()
        self.validation_data = self.validation_iterator.get_next()
        self.freq = freq

    def write_activation_summaries(self, model, input_images, input_labels, epoch=0):
        if type(model.layers[0])==tf.python.keras.engine.training.Model:
            #In case model was constructed from a base model
            model = model.layers[0]

        activation_grids_list = visualize_activations(input_images, model, self.sess, self.graph, group='vgg16_conv_block_outputs')[:1]
        i=0
        for activation_grid in activation_grids_list:
            # layer_name =  get_layers_by_index(model,[i])[0].name
            layer_name, grid_summary = self.sess.run(*activation_grid)
            self.write_image_summary(grid_summary, title=f'Layer : {layer_name}', epoch=epoch)

    def on_epoch_end(self, epoch, logs={}):
>>>>>>> 1179b95c98968c8d47c7e3ebfdac6146574ae95e

        if self.freq==0:
            return
        if epoch % self.freq != 0:
            return

        input_images, input_labels = self.get_random_batch()
        self.seed += 1

        # with self.sess.as_default():
        self.write_activation_summaries(self.model, input_images, input_labels, epoch=epoch)

#############################################################


# class ConfusionMatrixCallback(BaseCallback):
#
#     def __init__(self, val_data, batches_per_epoch, log_dir, freq=10, sess=None):
#         super().__init__(log_dir)
#         # self.sess = tf.Session()
#         self.graph = tf.get_default_graph() #self.sess.graph #
#         self.sess = sess or tf.Session()
#         self.validation_iterator = val_data.make_one_shot_iterator()
#         self.validation_data = self.validation_iterator.get_next()
#         self.batches_per_epoch = batches_per_epoch
#         self.freq = freq
#         print('EXECUTING EAGERLY: ', tf.executing_eagerly())
#         # self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)
#
#     def confusion_matrix(self, y_target, y_predicted, binary=False, positive_label=1):
#         # import pdb; pdb.set_trace()
#         cm = evaluate.confusion_matrix(y_target, y_predicted, binary=binary, positive_label=positive_label)
#         fig, ax = plotting.plot_confusion_matrix(conf_mat=cm)
#         buffer = io.BytesIO()
#         fig.savefig(buffer, format='png')
#         buffer.seek(0)
#         img = tf.image.decode_png(buffer.getvalue(), channels=4)
#         img = tf.expand_dims(img, 0)
#         return img
#
#     def on_epoch_end(self, epoch, logs={}):
#         # import pdb;pdb.set_trace()
#         image_summary = []
#
#         if self.freq==0:
#             return
#         if epoch % self.freq != 0:
#             return
#
#         val_data = (self.get_batch() for _ in range(self.batches_per_epoch))
#         real_fake = {'y_pred':[],'labels':[]}
#         for imgs, labels in val_data:
#             y_logits = self.model.predict(imgs)
#             y_pred = np.argmax(y_logits, axis=1).tolist()
#             y_true = np.argmax(labels, axis=1).tolist()
#             real_fake['y_pred'].extend(y_pred)
#             real_fake['labels'].extend(y_true)
#         cm = self.confusion_matrix(real_fake['labels'], real_fake['y_pred'])
#         self.write_image_summary(cm, 'confusion matrix', epoch=epoch)
#         if 'confusion_matrix' not in logs:
#             logs['confusion_matrix'] = []
#
#         logs['confusion_matrix'].append(cm)
        # return epoch, logs









# class TensorflowImageLogger(Callback):
#
#     def __init__(self, val_data, log_dir, max_images=25, freq=10, sess=None):
#         super().__init__()
#         self.log_dir = log_dir
#         self.graph = tf.get_default_graph() #self.sess.graph #
#         self.sess = sess or tf.Session()
#         # self.sess = tf.Session()
#         self.validation_iterator = val_data.make_one_shot_iterator()
#         self.validation_data = self.validation_iterator.get_next()
#
#         self.max_images = max_images
#         self.freq = freq
#         print('EXECUTING EAGERLY: ', tf.executing_eagerly())
#
#         self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)
#
#     def get_batch(self):
#         batch = self.sess.run(self.validation_data)
#         return batch
#
#     @tfmpl.figure_tensor
#     def image_grid(self, input_images=[], titles=[]):
#         """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
#         max_images = self.max_images
#
#         r, c = 3, 5
# #         figure = plt.figure(figsize=(10,10))
#         figs = tfmpl.create_figures(1, figsize=(20,30))
#         cnt = 0
#         for idx, f in enumerate(figs):
#             for i in range(r):
#                 for j in range(c):
#                     ax = f.add_subplot(r,c, cnt + 1)#, title=input_labels[cnt])
#                     ax.set_xticklabels([])
#                     ax.set_yticklabels([])
#                     img = input_images[cnt]
#                     ax.set_title(titles[cnt])
#                     ax.imshow(img)
#                     cnt+=1
#             f.tight_layout()
#         return figs
#
#     def write_image_summary(self, input_images, input_labels, titles):
# #         with K.get_session() as sess:
#         if True:
#             image_summary = tf.summary.image(f'images', self.image_grid(input_images, titles))
#             img_sum = self.sess.run(image_summary)
#             self.writer.add_summary(img_sum)
#             print('Finished image summary writing')
#
#     def on_epoch_end(self, epoch, logs={}):
#         image_summary = []
# #         batch_size = self.max_images
#         input_images, input_labels = self.get_batch()
#
#
#         print(input_images.shape)
#         titles = []
#         for idx in range(15):
#             img = input_images[idx]
#
#             img_min, img_max = np.min(img), np.max(img)
#             if (img_max <= 1.0) and (img_min >= -1.0):
#                 #Scale from [-1.0,1.0] to [0.0,1.0] for visualization
#                 titles.append(f'(min,max) pixels = ({img_min:0.1f},{img_max:0.1f})|rescaled->[0.0,1.0]')
#                 img += 1
#                 img /= 2
#             else:
#                 titles.append(f'(min,max) pixels = ({img_min:0.1f},{img_max:0.1f})')
#
#         self.write_image_summary(input_images, input_labels, titles)


def get_callbacks(weights_best=r'./model_ckpt.h5',
                  logs_dir=r'/media/data/jacob',
                  restore_best_weights=False,
                  val_data=None,
                  batches_per_epoch=20,
                  histogram_freq=10,
                  freq=10,
                  seed=None,
                  patience=25,
                  sess=None):


    checkpoint = ModelCheckpoint(weights_best, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min',restore_best_weights=restore_best_weights)

    tfboard = TensorBoard(log_dir=logs_dir, histogram_freq=histogram_freq)#, write_images=True)
    csv = CSVLogger(os.path.join(logs_dir,'training_log.csv'))
    early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    callback_list = [checkpoint,tfboard,early,csv]

    if val_data is not None:
        # callback_list.append(VisualizeActivationsCallback(val_data, logs_dir, freq=freq, seed=seed, sess=sess))
        callback_list.append(ConfusionMatrixCallback(val_data, batches_per_epoch, logs_dir, freq=freq, sess=sess))
        # callback_list.append(TensorflowImageLogger(val_data, log_dir = logs_dir, freq=freq))
        if sess is not None:
            sess.run(tf.initialize_all_variables())
    return callback_list
