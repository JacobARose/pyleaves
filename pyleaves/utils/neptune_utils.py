# @Author: Jacob A Rose
# @Date:   Thu, June 25th 2020, 2:23 pm
# @Email:  jacobrose@brown.edu
# @Filename: neptune_utils.py


'''

Logging utils for working with Neptune.ai

'''



import neptune
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


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
            self.add_log(x[i,...], counter=self._count+i, name = f'{y[i]}-{self.name}')
        print(f'Batch {self._batch}: Logged {np.max([x.shape[0],self.max_images])} {self.name} images to neptune')

    def on_epoch_end(self, epoch, logs={}):
        self.finished = True
