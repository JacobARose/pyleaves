# @Author: Jacob A Rose
# @Date:   Wed, July 22nd 2020, 9:51 pm
# @Email:  jacobrose@brown.edu
# @Filename: callback_utils.py


'''
Created (7/22/2020) by Jacob A Rose
-This script was originally located at pyleaves.train.callbacks
-All scripts that reference original location should be transitioned to here

'''


import io
import os
import random
import signal
import sys
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.framework import ops
# from tensorflow.compat.v1.keras.callbacks import 
from tensorflow.keras.callbacks import (Callback,
										CSVLogger,
										ModelCheckpoint,
										TensorBoard,
										LearningRateScheduler,
										ReduceLROnPlateau,
										EarlyStopping)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import neptune
import cv2
from typing import List
# from mlxtend import evaluate, plotting
from pyleaves.models.vgg16 import visualize_activations#, get_layers_by_index
from pyleaves.utils.model_utils import WorkerTrainingState
from pyleaves.utils.neptune_utils import neptune

class BackupAndRestore(Callback):
	"""
	Based on official tf.keras.callbacks.BackupAndRestore

	Callback to back up and restore the training state.
	`BackupAndRestore` callback is intended to recover from interruptions that
	happened in the middle of a model.fit execution by backing up the
	training states in a temporary checkpoint file (based on TF CheckpointManager)
	at the end of each epoch. If training restarted before completion, the
	training state and model are restored to the most recently saved state at the
	beginning of a new model.fit() run.

	Note:
	1. This callback is not compatible with disabling eager execution.
	2. A checkpoint is saved at the end of each epoch, when restoring we'll redo
	any partial work from an unfinished epoch in which the training got restarted
	(so the work done before a interruption doesn't affect the final model state).
	3. This works for both single worker and multi-worker mode, only
	MirroredStrategy and MultiWorkerMirroredStrategy are supported for now.
	Example:
	>>> class InterruptingCallback(tf.keras.callbacks.Callback):
	...   def on_epoch_begin(self, epoch, logs=None):
	...     if epoch == 4:
	...       raise RuntimeError('Interrupting!')
	>>> callback = tf.keras.callbacks.experimental.BackupAndRestore(
	... backup_dir="/tmp")
	>>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
	>>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
	>>> try:
	...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
	...             batch_size=1, callbacks=[callback, InterruptingCallback()],
	...             verbose=0)
	... except:
	...   pass
	>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
	...             batch_size=1, callbacks=[callback], verbose=0)
	>>> # Only 6 more epochs are run, since first trainning got interrupted at
	>>> # zero-indexed epoch 4, second training will continue from 4 to 9.
	>>> len(history.history['loss'])

	Arguments:
	  backup_dir: String,
		path to save the model file. This is the directory in
		which the system stores temporary files to recover the model from jobs
		terminated unexpectedly. The directory cannot be reused elsewhere to
		store other checkpoints, e.g. by BackupAndRestore callback of another
		training, or by another callback (ModelCheckpoint) of the same training.
	"""

	def __init__(self, backup_dir):
		super(BackupAndRestore, self).__init__()
		self.backup_dir = backup_dir
		self._supports_tf_logs = True
		# self._supported_strategies = (
		# 	distribute_lib._DefaultDistributionStrategy,
		# 	mirrored_strategy.MirroredStrategy,
		# 	collective_all_reduce_strategy.CollectiveAllReduceStrategy)

		if not context.executing_eagerly():
			if ops.inside_function():
				raise ValueError('This Callback\'s method contains Python state and '
							 'should be called outside of `tf.function`s.')
			else:  # Legacy graph mode:
				raise ValueError(
					'BackupAndRestore only supports eager mode. In graph '
					'mode, consider using ModelCheckpoint to manually save '
					'and restore weights with `model.load_weights()` and by '
					'providing `initial_epoch` in `model.fit()` for fault tolerance.')

		# Only the chief worker writes model checkpoints, but all workers
		# restore checkpoint at on_train_begin().
		self._chief_worker_only = False

	def set_model(self, model):
		self.model = model

	def on_train_begin(self, logs=None):
		# TrainingState is used to manage the training state needed for
		# failure-recovery of a worker in training.
		# pylint: disable=protected-access
		# if not isinstance(self.model.distribute_strategy,self._supported_strategies):
		# 	raise NotImplementedError(
		# 							  'Currently only support empty strategy, MirroredStrategy and '
		# 							  'MultiWorkerMirroredStrategy.')
		self.model._training_state = (WorkerTrainingState(self.model, self.backup_dir))
		self._training_state = self.model._training_state
		self._training_state.restore()
		# signal.signal(signal.SIGINT, self._delete_backup_signal)

	def on_train_end(self, logs=None):
		# pylint: disable=protected-access
		# On exit of training, delete the training state backup file that was saved
		# for the purpose of worker recovery.
		if self.model._in_multi_worker_mode():
			if self.model.stop_training or getattr(self.model, '_successful_loop_finish', False):
				self._training_state.delete_backup()
				# del self._training_state
				# del self.model._training_state

	def on_epoch_end(self, epoch, logs=None):
		# Back up the model and current epoch for possible future recovery.
		self._training_state.back_up(epoch)













class BaseCallback(Callback):

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




from tqdm import trange

def tf_data2np(data: tf.data.Dataset, num_batches: int=None):
    
	num_batches = num_batches or 4

	x_val, y_val = [], []
	data_iter = iter(data)
	print(f'Loading {num_batches} batches into memory for confusion matrix callback')
	for _ in trange(num_batches):
		x, y = next(data_iter)
		x_val.append(x)
		y_val.append(y)
	return np.vstack(x_val), np.vstack(y_val)







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
	def __init__(self, 
				 validation_data,
				 num_classes: int=None,
				 labels: List[int]=None,
				 text_labels: List[str]=None,
				 steps: int=None,
				 subset_prefix: str=None,
				 experiment=None):
		super().__init__()

		self.experiment = experiment or neptune
		if isinstance(validation_data, tf.data.Dataset):
			x_true, y_true = tf_data2np(data=validation_data, num_batches=steps)
			# if steps:
			# 	x_true,y_true=[],[]
			# 	print(f'Instantiating {steps} batches into memory for use in NeptuneVisualizationcallback')
			# 	for i, (x,y) in enumerate(validation_data):
			# 		x_true.append(x.numpy())
			# 		y_true.append(y.numpy())
			# 		print('Finished batch',i)
			# 		if i >= steps:
			# 			break
			# 	x_true = np.vstack(x_true)
			# 	y_true = np.vstack(y_true)
			# else:
			# 	x_true, y_true = next(iter(validation_data))
			# 	x_true = x_true.numpy()
			# 	y_true = y_true.numpy()
		elif type(validation_data)==tuple:
			x_true, y_true = validation_data
		else:
			print('Invalid data type for input parameter validation_data in NeptuneVisualizationCallback constructor')
			raise Exception
		if y_true.ndim > 1:
			y_true = np.argmax(y_true, axis=1)

		self.x_true = x_true
		self.y_true = y_true
			
		self.num_classes = num_classes or len(set(y_true))
		self.labels = labels
		self.text_labels = text_labels

		# if subset_prefix is None:
		# 	self.prefix = ''
		# else:
		self.prefix = subset_prefix or ''

		self.validation_data = (x_true, y_true)
		print('Finished initializing NeptuneVisualizationCallback')

	def get_predictions(self, epoch, logs={}):
		x_true, y_true = self.validation_data
		y_prob = np.asarray(self.model.predict(x_true))
		y_pred = np.argmax(y_prob, axis=1)
		
		if self.text_labels:
			y_true = [self.text_labels[y] for y in y_true]
			y_pred = [self.text_labels[y] for y in y_pred]
			labels = self.text_labels
		else:
			labels = self.labels
		return x_true, y_true, y_prob, y_pred, labels

	def on_batch_end(self, batch, logs={}):
		for log_name, log_value in logs.items():
			self.experiment.log_metric(f'{self.prefix}_batch_{log_name}', log_value)

	def on_epoch_end(self, epoch, logs={}):
		for log_name, log_value in logs.items():
			self.experiment.log_metric(log_name, log_value)

		_, y_true, y_prob, y_pred, labels = self.get_predictions(epoch, logs={})

		fig, ax = plt.subplots(figsize=(16, 12))
		plot_confusion_matrix(y_true, y_pred, labels=labels, ax=ax)
		self.experiment.log_image(f'{self.prefix}_confusion_matrix', fig)
		

		if self.num_classes == 2:
			fig, ax = plt.subplots(figsize=(16, 12))
			plot_roc(y_true, y_prob, ax=ax)
			self.experiment.log_image(f'{self.prefix}_roc_curve', fig)

		plt.close('all')



	def on_test_end(self, epoch, logs={}):
		for log_name, log_value in logs.items():
			self.experiment.log_metric(log_name, log_value)

		_, y_true, y_prob, y_pred, labels = self.get_predictions(epoch, logs=logs)

		fig, ax = plt.subplots(figsize=(16, 12))
		plot_confusion_matrix(y_true, y_pred, labels=labels, ax=ax)
		self.experiment.log_image(f'{self.prefix}_confusion_matrix', fig)

		if self.num_classes == 2:
			fig, ax = plt.subplots(figsize=(16, 12))
			plot_roc(y_true, y_prob, ax=ax)
			self.experiment.log_image(f'{self.prefix}_roc_curve', fig)
		plt.close('all')

# class NeptuneVisualizationCallback(Callback):
# 	"""Callback for logging to Neptune.

# 	1. on_batch_end: Logs all performance metrics to neptune
# 	2. on_epoch_end: Logs all performance metrics + confusion matrix + roc plot to neptune

# 	Parameters
# 	----------
# 	model : type
# 		Description of parameter `model`.
# 	validation_data : list
# 		len==2 list containing [images, labels]
# 	image_dir : type
# 		Description of parameter `image_dir`.

# 	Examples
# 	-------
# 	Examples should be written in doctest format, and
# 	should illustrate how to use the function/class.
# 	>>>

# 	Attributes
# 	----------
# 	model
# 	validation_data

# 	"""
# 	def __init__(self, validation_data):
# 		super().__init__()
# 		# self.model = model
# 		self.validation_data = validation_data

# 	def on_batch_end(self, batch, logs={}):
# 		for log_name, log_value in logs.items():
# 			neptune.log_metric(f'batch_{log_name}', log_value)

# 	def on_epoch_end(self, epoch, logs={}):
# 		for log_name, log_value in logs.items():
# 			neptune.log_metric(f'epoch_{log_name}', log_value)

# 		y_pred = np.asarray(self.model.predict(self.validation_data[0]))
# 		y_true = self.validation_data[1]

# 		y_pred_class = np.argmax(y_pred, axis=1)

# 		fig, ax = plt.subplots(figsize=(16, 12))
# 		plot_confusion_matrix(y_true, y_pred_class, ax=ax)
# 		neptune.log_image('confusion_matrix', fig)

# 		fig, ax = plt.subplots(figsize=(16, 12))
# 		plot_roc(y_true, y_pred, ax=ax)
# 		neptune.log_image('roc_curve', fig)










class ConfusionMatrixCallback(Callback):

	def __init__(self, log_dir, val_imgs, val_labels, classes, freq=1, seed=None, neptune_experiment=None):
		super().__init__()
		# self.file_writer = tf.contrib.summary.create_file_writer(log_dir)
		self.file_writer = tf.summary.create_file_writer(log_dir)
		self.log_dir = log_dir
		self.seed = seed
		self._counter = 0
		self.val_imgs = val_imgs

		if val_labels.ndim==2:
			val_labels = tf.argmax(val_labels,axis=-1)
		self.val_labels = val_labels
		self.num_samples = val_labels.numpy().shape[0]
		self.classes = classes
		self.freq = freq
		self.experiment = neptune_experiment

	def log_confusion_matrix(self, model, imgs, labels, epoch, norm_cm=False):

		# pred_labels = model.predict_classes(imgs)
		# pred_labels = pred_labels[:,None]
		pred_labels = tf.argmax(model.predict(imgs), axis=-1)
		# pred_labels = pred_labels[:,None]

		con_mat = tf.math.confusion_matrix(labels=labels, predictions=pred_labels, num_classes=len(self.classes)).numpy()
		if norm_cm:
			con_mat = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
		con_mat_df = pd.DataFrame(con_mat,
						 index = self.classes,
						 columns = self.classes)

		figure = plt.figure(figsize=(16, 16))
		sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)

		image = tf.image.decode_png(buf.getvalue(), channels=4)
		image = tf.expand_dims(image, 0)

		# with self.file_writer.as_default(), tf.contrib.summary.always_record_summaries():
		# 	tf.contrib.summary.image(name='val_confusion_matrix',
		# 							 tensor=image,
		# 							 step=self._counter)
		with self.file_writer.as_default(): #, tf.summary.always_record_summaries():
			tf.summary.image(name='val_confusion_matrix',
							 data=image,
							 step=epoch) #self._counter)

		if self.experiment:
			self.experiment.log_image(log_name='val_confusion_matrix',
				   					  x=epoch, #self._counter,
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


# def get_callbacks(weights_best=r'./model_ckpt.h5',
# 				  logs_dir=r'/media/data/jacob',
# 				  restore_best_weights=False,
# 				  val_data=None,
# 				  batches_per_epoch=20,
# 				  histogram_freq=10,
# 				  freq=10,
# 				  seed=None,
# 				  patience=25,
# 				  sess=None):


# 	checkpoint = ModelCheckpoint(weights_best, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min',restore_best_weights=restore_best_weights)

# 	tfboard = TensorBoard(log_dir=logs_dir, histogram_freq=histogram_freq)#, write_images=True)
# 	csv = CSVLogger(os.path.join(logs_dir,'training_log.csv'))
# 	early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

# 	callback_list = [checkpoint,tfboard,early,csv]

# 	if val_data is not None:
# 		# callback_list.append(VisualizeActivationsCallback(val_data, logs_dir, freq=freq, seed=seed, sess=sess))
# 		callback_list.append(ConfusionMatrixCallback(val_data, batches_per_epoch, logs_dir, freq=freq, sess=sess))
# 		# callback_list.append(TensorflowImageLogger(val_data, log_dir = logs_dir, freq=freq))
# 		if sess is not None:
# 			sess.run(tf.initialize_all_variables())
# 	return callback_list
