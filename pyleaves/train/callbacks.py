
import os
from tensorflow.compat.v1.keras.callbacks import (Callback,
                                                  CSVLogger,
                                                  ModelCheckpoint,
                                                  TensorBoard,
                                                  LearningRateScheduler,
                                                  EarlyStopping)
import datetime

from tensorflow import keras
# import matplotlib.pyplot as plt
import tfmpl
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2

class TensorflowImageLogger(Callback):
        
    def __init__(self, val_data, log_dir, max_images=25, freq=10):
        super().__init__()
        self.log_dir = log_dir
        self.sess = tf.Session()
        self.validation_iterator = val_data.make_one_shot_iterator()
        self.validation_data = self.validation_iterator.get_next()

        self.max_images = max_images
        self.freq = freq
        print('EXECUTING EAGERLY: ', tf.executing_eagerly())

        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)
            
    def get_batch(self):
        batch = self.sess.run(self.validation_data)
        return batch
    
    @tfmpl.figure_tensor
    def image_grid(self, input_images=[], titles=[]):
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
        max_images = self.max_images
        
        r, c = 3, 5
#         figure = plt.figure(figsize=(10,10))
        figs = tfmpl.create_figures(1, figsize=(20,30))
        cnt = 0
        for idx, f in enumerate(figs):
            for i in range(r):
                for j in range(c):
                    ax = f.add_subplot(r,c, cnt + 1)#, title=input_labels[cnt])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    img = input_images[cnt]
                    ax.set_title(titles[cnt])
                    ax.imshow(img)
                    cnt+=1
            f.tight_layout()
        return figs
    
    def write_image_summary(self, input_images, input_labels, titles):
#         with K.get_session() as sess:
        if True:
            image_summary = tf.summary.image(f'images', self.image_grid(input_images, titles))
            img_sum = self.sess.run(image_summary)
            self.writer.add_summary(img_sum)
            print('Finished image summary writing')
        
    def on_epoch_end(self, epoch, logs={}):
        image_summary = []
#         batch_size = self.max_images
        input_images, input_labels = self.get_batch()
    
        
        print(input_images.shape)
        titles = []
        for idx in range(15):
            img = input_images[idx]
            
            img_min, img_max = np.min(img), np.max(img)
            if (img_max <= 1.0) and (img_min >= -1.0):
                #Scale from [-1.0,1.0] to [0.0,1.0] for visualization
                titles.append(f'(min,max) pixels = ({img_min:0.1f},{img_max:0.1f})|rescaled->[0.0,1.0]')
                img += 1
                img /= 2
            else:
                titles.append(f'(min,max) pixels = ({img_min:0.1f},{img_max:0.1f})')
            
        self.write_image_summary(input_images, input_labels, titles)
            

def get_callbacks(weights_best=r'./model_ckpt.h5', 
                  logs_dir=r'/media/data/jacob', 
                  restore_best_weights=False,
                  val_data=None,
                  freq=10):
    
    
    checkpoint = ModelCheckpoint(weights_best, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min',restore_best_weights=restore_best_weights)
    
    tfboard = TensorBoard(log_dir=logs_dir)#, write_images=True)
    csv = CSVLogger(os.path.join(logs_dir,'training_log.csv'))
    early = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    
    callback_list = [checkpoint,tfboard,early,csv]
    
    if val_data is not None:
        callback_list.append(TensorflowImageLogger(val_data, log_dir = logs_dir, freq=freq))
    
    return callback_list

