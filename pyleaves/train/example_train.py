"""
Created on Mon Feb 10 03:23:32 2019

script: pyleaves/pyleaves/train/example_train.py

@author: JacobARose
"""


def main(experiment_config, experiment_dir):


    ############################################
    #TODO: Move config definitions outside main() for:
    #    1. simplifying overall logic in main & segregating configuration to section marked by if __name__=='__main__'
    #    2. Moving towards defining most or all run parameters in separate config files
    ############################################


    trainer = BaseTrainer(experiment_config=experiment_config)
    
    for subset, paths in trainer.tfrecord_files.items():
        for path in paths:
            mlflow.log_artifact(path,f'artifacts/{subset}')
    
    

    train_data = trainer.get_data_loader(subset='train')
    val_data = trainer.get_data_loader(subset='val')
    test_data = trainer.get_data_loader(subset='test')

#     AUTOTUNE = tf.data.experimental.AUTOTUNE
#     train_data = tfds.load("mnist", split='train').shuffle(1000).batch(batch_size).repeat().prefetch(AUTOTUNE)

    model_params = trainer.get_model_params('train')
    fit_params = trainer.get_fit_params()
    callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'weights_best.h5'), logs_dir=os.path.join(experiment_dir,'logdir'), restore_best_weights=False)

    print('model_params',model_params)
    
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
    parser.add_argument('-ch', '--num_channels', default=3, type=int, help='Number of input channels, either 1 for grayscale, or 3 for rgb')    
    parser.add_argument('-bsz', '--batch_size', default=64, type=int, help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default=1e-4, type=float, help='Starting learning rate')
    parser.add_argument('-epochs', '--num_epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('-f',default='')
    args = parser.parse_args()

    import datetime
    import numpy as np
    import os
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()

    from pyleaves.utils import set_visible_gpus, ensure_dir_exists
#     args.gpu_id=2
    set_visible_gpus([args.gpu_id])
    ####
    from pyleaves.leavesdb.tf_utils.tf_utils import reset_eager_session
    from pyleaves.models.keras_models import build_model
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig
    from pyleaves.train.base_train import BaseTrainer
    from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history

    import mlflow
    import mlflow.tensorflow
    tracking_dir = r'/media/data/jacob/Fossil_Project/experiments/mlflow'
    ensure_dir_exists(tracking_dir)
    mlflow.set_tracking_uri(tracking_dir)
#     mlflow.set_tracking_uri(r'sqlite:///'+tracking_dir+'/experiment.db')
    print(mlflow.tracking.get_tracking_uri())
    
    mlflow.set_experiment(tracking_dir+r'/baselines')
    print(mlflow.get_artifact_uri())
    
    if args.num_channels==3:
        color_type = 'rgb'
    else:
        color_type = 'grayscale'
    
    model_names = ['vgg16', 'xception', 'resnet_50_v2','resnet_101_v2', 'shallow']
    
    for model_name in model_names:
        args.model_name = model_name
        print('model_name=',args.model_name)
        if args.model_name in ['vgg16', 'resnet_50_v2','resnet_101_v2']:
            target_size=(224,224)
        elif args.model_name=='xception':
            target_size=(299,299)
        else:
            target_size=(224,224)
            
        dataset_names = ['PNAS', 'Fossil', 'Leaves']
        histories = []
        for dataset_name in dataset_names:
            args.dataset_name = dataset_name
            
            if args.model_name=='vgg16' and args.dataset_name != 'Leaves':
                continue
            
            learning_rates = [1e-4,1e-5]            
            for lr in learning_rates:
                args.base_learning_rate = lr

                with mlflow.start_run(run_name=f'{args.model_name}-{args.dataset_name}-{color_type}-lr_{args.base_learning_rate}_baseline', nested=True):
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',args.model_name,args.dataset_name,color_type, current_time)

                    reset_eager_session()

                    dataset_config = DatasetConfig(dataset_name=args.dataset_name,
                                                   label_col='family',
                                                   target_size=target_size,
                                                   num_channels=args.num_channels,
                                                   low_class_count_thresh=3,
                                                   data_splits={'val_size':0.2,'test_size':0.2},
                                                   tfrecord_root_dir=args.tfrecord_dir,
                                                   num_shards=10)

                    train_config = TrainConfig(model_name=args.model_name,
                                               batch_size=args.batch_size,
                                               frozen_layers=(0,-4),
                                               base_learning_rate=args.base_learning_rate,
                                               buffer_size=1000,
                                               num_epochs=args.num_epochs,
                                               preprocessing=True, #'imagenet',
                                               augment_images=True,
                                               augmentations=['rotate','flip'],
                                               seed=4) #3)

                    experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                                         train_config=train_config)            

                    mlflow.tensorflow.autolog()
                    mlflow.log_params(args.__dict__)
                    mlflow.log_params(experiment_config)

                    print(f'BEGINNING: DATASET:{args.dataset_name}|MODEL:{args.model_name}|lr:{args.base_learning_rate}|num_channels:{args.num_channels}')
                    print('-'*30)

                    history = main(experiment_config, experiment_dir)

                    histories.append((dataset_name, model_name, history))
                #     mlflow_log_best_history(history)
                    mlflow_log_history(history)

            
            
            
            
            
            
            
            
            