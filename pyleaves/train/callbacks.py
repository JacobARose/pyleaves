
import os
from tensorflow.compat.v1.keras.callbacks import (CSVLogger,
                                                  ModelCheckpoint,
                                                  TensorBoard,
                                                  LearningRateScheduler,
                                                  EarlyStopping)


def get_callbacks(weights_best=r'./model_ckpt.h5', logs_dir=r'/media/data/jacob', restore_best_weights=False):
    
    checkpoint = ModelCheckpoint(weights_best, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min',restore_best_weights=restore_best_weights)
    tfboard = TensorBoard(log_dir=logs_dir)
    csv = CSVLogger(os.path.join(logs_dir,'training_log.csv'))
    early = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    
    return [checkpoint,tfboard,early,csv]