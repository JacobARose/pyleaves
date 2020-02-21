"""
Created on Mon Feb 10 03:23:32 2019

script: pyleaves/pyleaves/train/example_train.py

@author: JacobARose
"""


def main(dataset_name='PNAS',
         model_name='vgg16',
         experiment_dir=r'/media/data/jacob/Fossil_Project/vgg16/PNAS',
         gpu_ids = [0],
         tfrecord_root_dir=r'/media/data/jacob/Fossil_Project/tfrecord_data',
         batch_size=64,
         target_size=(224,224),
         base_learning_rate=0.001,
         num_epochs=100,
         preprocessing='imagenet',
         augment_images=False):

    reset_eager_session()
#     tf.reset_default_graph()
    dataset_config = DatasetConfig(dataset_name=dataset_name,
                                   label_col='family',
                                   target_size=target_size,
                                   channels=3,
                                   low_class_count_thresh=3,
                                   data_splits={'val_size':0.2,'test_size':0.2},
                                   tfrecord_root_dir=tfrecord_root_dir,
                                   num_shards=10)

    train_config = TrainConfig(model_name=model_name,
                     batch_size=batch_size,
                     frozen_layers=(0,-4),
                     base_learning_rate=base_learning_rate,
                     buffer_size=1000,
                     num_epochs=num_epochs,
                     preprocessing=preprocessing,
                     augment_images=augment_images,
                     seed=3)

    experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                         train_config=train_config)

    ############################################
    #TODO: Move config definitions outside main() for:
    #    1. simplifying overall logic in main & segregating configuration to section marked by if __name__=='__main__'
    #    2. Moving towards defining most or all run parameters in separate config files
    ############################################


    trainer = BaseTrainer(experiment_config=experiment_config)


    train_data = trainer.get_data_loader(subset='train')
    val_data = trainer.get_data_loader(subset='val')
    test_data = trainer.get_data_loader(subset='test')

#     AUTOTUNE = tf.data.experimental.AUTOTUNE
#     train_data = tfds.load("mnist", split='train').shuffle(1000).batch(batch_size).repeat().prefetch(AUTOTUNE)

    model_params = trainer.get_model_params('train')
    fit_params = trainer.get_fit_params()
    callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'weights_best.h5'), logs_dir=os.path.join(experiment_dir,'logdir'), restore_best_weights=False)

    model = build_model(**model_params)  #name='shallow', num_classes=10000, frozen_layers=(0,-4), input_shape=(224,224,3), base_learning_rate=0.0001)


    history = model.fit(train_data,
                 steps_per_epoch = fit_params['steps_per_epoch'],
                 epochs=fit_params['epochs'],
                 validation_data=val_data,
                 validation_steps=fit_params['validation_steps'],
                 callbacks=callbacks
                 )
    return history


if __name__=='__main__':
    '''
    Example:
    python /home/jacob/pyleaves/pyleaves/train/example_train.py -d PNAS -m resnet_50_v2 -gpu 3 -bsz 64


    python example_train.py -d PNAS -m resnet_50_v2 -gpu 3 -bsz 64

    Possible models:
    [
    'shallow',
    'vgg16',
    'xception',
    'resnet_50_v2',
    'resnet_101_v2'
    ]

    '''

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', default='PNAS', type=str, help='Name of dataset of images to use for creating TFRecords')
    parser.add_argument('-m', '--model_name', default='vgg16', type=str, help='Name of model to train')
    parser.add_argument('-gpu', '--gpu_id', default=0, type=int, help='integer number of gpu to train on')

    parser.add_argument('-tfrec', '--tfrecord_dir', default=r'/media/data/jacob/Fossil_Project/tfrecord_data', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    parser.add_argument('-bsz', '--batch_size', default=64, type=int, help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default=0.001, type=float, help='Starting learning rate')
    parser.add_argument('-epochs', '--num_epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-f',default='')
    args = parser.parse_args()

    import numpy as np
    import os
    import tensorflow as tf
    tf.enable_eager_execution()

    from pyleaves.utils import set_visible_gpus
    args.gpu_id=3
    set_visible_gpus([3])#args.gpu_id])
    ####
    from pyleaves.leavesdb.tf_utils.tf_utils import reset_eager_session
    from pyleaves.models.keras_models import build_model
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig
    from pyleaves.train.base_train import BaseTrainer
    from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history

    import mlflow
    import mlflow.tensorflow
    mlflow.set_tracking_uri(r'/media/data/jacob/Fossil_Project/mlflow')
    mlflow.set_artifact_uri(r'./mlruns')
    print(mlflow.tracking.get_tracking_uri())
    print(mlflow.get_artifact_uri())

    mlflow.set_experiment('Base Learning rate hp_search')

    if args.model_name in ['vgg16', 'resnet_50_v2','resnet_101_v2']:
        target_size=(224,224)
    elif args.model_name=='xception':
        target_size=(299,299)
    else:
        target_size=(224,224)

    learning_rates = [0.001,1e-4,1e-5,1e-6]
    histories = []
    for lr in learning_rates:

        with mlflow.start_run(run_name=f'lr={lr}', nested=True):
            experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',args.model_name,args.dataset_name,f'lr-{lr}')

            args.base_learning_rate=lr
            args.target_size=target_size

            mlflow.tensorflow.autolog()
            mlflow.log_params(args.__dict__)

            print('STARTING LEARNING RATE =',lr)
            history = main(args.dataset_name,
                 args.model_name,
                 experiment_dir,
                 [args.gpu_id],
                 args.tfrecord_dir,
                 batch_size=args.batch_size,
                 target_size=target_size,
                 base_learning_rate=lr,
                 num_epochs=args.num_epochs,
                 preprocessing='imagenet',
                 augment_images=True)

            histories.append((lr,history))
        #     mlflow_log_best_history(history)
            mlflow_log_history(history)
