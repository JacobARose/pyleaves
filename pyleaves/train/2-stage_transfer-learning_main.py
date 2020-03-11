"""
Created on Mon Feb 10 03:23:32 2019

script: pyleaves/pyleaves/train/example_train.py

@author: JacobARose
"""


def main(experiment_configs, experiment_dir):


    ############################################
    #TODO: Moving towards defining most or all run parameters in separate config files
    ############################################


    trainer = TransferTrainer(experiment_configs, src_db=os.path.join(pyleaves.RESOURCES_DIR,'updated_leavesdb.db'))

    trainer.init_model_builder(domain='source')

    source_model_filepath = os.path.join(trainer.model_manager.model_dir,trainer.model_name+'_source_model.h5')
    target_model_filepath = os.path.join(trainer.model_manager.model_dir,trainer.model_name+'_target_model.h5')    
    
    source_train_data = trainer.get_data_loader(domain='source', subset='train')
    source_val_data = trainer.get_data_loader(domain='source', subset= 'val')

    #Get parameters for fitting and callbacks
    fit_params = trainer.get_fit_params(domain='source')
    callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'source_domain_weights_best.h5'), 
                                  logs_dir=os.path.join(experiment_dir,'tensorboard_logs'), 
                                  restore_best_weights=True)

    # TRAIN ON SOURCE DOMAIN

    history = trainer.model.fit(source_train_data,
                     steps_per_epoch = fit_params['steps_per_epoch'],
                     epochs=fit_params['epochs'],
                     validation_data=source_val_data,
                     validation_steps=fit_params['validation_steps'],
                     callbacks=callbacks
                     )
    trainer.histories['source'] = history

    
    trainer.save_model(filepath=source_model_filepath)
    #######################################################################
    # TARGET DOMAIN
    
    #trainer.load_model(filepath=source_model_filepath)

    target_train_data = trainer.get_data_loader(domain='target', subset='train')
    target_val_data = trainer.get_data_loader(domain='target', subset= 'val')
    target_test_data = trainer.get_data_loader(domain='target', subset='test')

    fit_params = trainer.get_fit_params(domain='target')
    callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'target_domain_weights_best.h5'), 
                                  logs_dir=os.path.join(experiment_dir,'tensorboard_logs'), 
                                  restore_best_weights=True)

#     num_test_samples = trainer.domains['target'].metadata_splits['test']['num_samples']
#     num_steps = num_test_samples//batch_size
    #zero_shot_test_results = trainer.model.evaluate(target_test_data, steps=num_steps)

    # FINETUNE ON TARGET DOMAIN

    history = trainer.model.fit(target_train_data,
                     steps_per_epoch = fit_params['steps_per_epoch'],
                     epochs=fit_params['epochs'],
                     validation_data=target_val_data,
                     validation_steps=fit_params['validation_steps'],
                     callbacks=callbacks
                     )
    
    trainer.histories['target'] = history
    
    

    
    
    return trainer

if __name__=='__main__':
    '''
    Example:
    python /home/jacob/pyleaves/pyleaves/train/2-stage_transfer-learning_main.py -d PNAS Fossil -m vgg16 -gpu 0 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 20 -r l2 -r_p 0.001 --experiment TransferBaselines


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
    import datetime
    import json
    import numpy as np
    import os
    
    parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_config', default='', type=str, nargs='?', help='Requires 2 args identifying datasets by name in order of input to the pipeline. Stage 1: train + validate, Stage 2: finetune + validate + test')
#     parser.add_argument('-m', '--model_config', default='vgg16', type=str, help='Name of model to train')

    
    parser.add_argument('-d', '--dataset_names', default=['Leaves2020','PNAS'], type=str, nargs=2, help='Requires 2 args identifying datasets by name in order of input to the pipeline. Stage 1: train + validate, Stage 2: finetune + validate + test')
    parser.add_argument('-m', '--model_name', default='vgg16', type=str, help='Name of model to train')
    parser.add_argument('-gpu', '--gpu_id', default='1', type=str, help='integer number of gpu to train on')
#     parser.add_argument('-ch', '--num_channels', default=3, type=int, help='Number of input channels, either 1 for grayscale, or 3 for rgb')
    parser.add_argument('-c', '--color_type', default='grayscale', type=str, help='grayscale or rgb')
    parser.add_argument('-bsz', '--batch_size', default=64, type=int, help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default=1e-4, nargs='*', type=float, help="Starting learning rate, <float> for a single value or 'all' to loop through a hardcoded range of values")
    parser.add_argument('-thresh', '--low_class_count_thresh', default=10, type=int)
    parser.add_argument('-r', '--regularizations', default='l2', type=str, help='comma separated list of regularizers to search through. Enter combinations of l1 and l2, enter anything else for None.') #3
    parser.add_argument('-r_p', '--r_params', default='0.001', type=str, help='comma separated list of regularizer strengths to search through. Enter combinations of floats.') #3
    parser.add_argument('-epochs', '--num_epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('-exp', '--experiment', default='Baselines', type=str, help=r"Name of new or existing MLFlow experiment to log results into. TODO: Add None option")    
    parser.add_argument('--model_dir', default=r'/media/data_cifs/jacob/Fossil_Project/models', type=str, help='Directory in which to save/load models and/or model weights')
    parser.add_argument('-tfrec', '--tfrecord_dir', default=r'/media/data/jacob/Fossil_Project/tfrecord_data', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    parser.add_argument('-f',default='')
    args = parser.parse_args()

    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) ####SHOULD THIS BE AN INT???
    tf.compat.v1.enable_eager_execution()
    import pyleaves
    from pyleaves.utils import ensure_dir_exists # set_visible_gpus,
    ####
    from pyleaves.leavesdb.tf_utils.tf_utils import reset_eager_session
    
    from pyleaves.models.resnet import ResNet, ResNetGrayScale
    from pyleaves.models.vgg16 import VGG16, VGG16GrayScale    
    from pyleaves.models.keras_models import build_model
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig
    from pyleaves.train.transfer_trainer import TransferTrainer
    from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history

    import mlflow
    import mlflow.tensorflow
    
    tracking_dir = r'/media/data/jacob/Fossil_Project/experiments/mlflow'
    ensure_dir_exists(tracking_dir)
    mlflow.set_tracking_uri(tracking_dir)
    print(mlflow.tracking.get_tracking_uri())
    mlflow.set_experiment(args.experiment)
    
    ############################
    # Spaghetti Code for Assembling Hyperparameter search records to iterate through
    #########################################
    #########################################
    import itertools
    import random
    random.seed(6)
    from collections import OrderedDict    
    
    if args.model_name == 'all':
        model_names = ['resnet_50_v2','resnet_152_v2', 'vgg16', 'xception', 'shallow'][:3]
#         model_names = ['vgg16', 'xception', 'resnet_50_v2','resnet_101_v2', 'shallow']
    else:
        model_names=[args.model_name]
    #########################################
    dataset_names = [args.dataset_names]
    #########################################
    if type(args.base_learning_rate)!=list:
        learning_rates = [args.base_learning_rate]
    else:
        learning_rates = args.base_learning_rate
    #########################################
    if type(args.batch_size)!=list:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = args.batch_size
    #########################################
    regularizer = {args.regularizations:args.r_params}
    
                        
    
    hparams = OrderedDict()
    hparams['model_name_list'] = model_names #['resnet_50_v2','resnet_152_v2', 'vgg16', 'xception', 'shallow']
    hparams['dataset_name_list'] = dataset_names #['PNAS', 'Fossil', 'Leaves']
    hparams['learning_rate_list'] = learning_rates #[1e-3, 1e-4,1e-5]
    hparams['batch_size_list'] = batch_sizes #[64, 128]
#     hparams['regularizer_type_list'] = regularizer_types #['l1','l2']
#     hparams['regularizer_param_list'] = r_params #[0.001, 0.01]
    
    hparams_labeled = OrderedDict()
    for k, v in hparams.items():
        hparams_labeled[k] = list(itertools.product([k.replace('_list','s')],v))
#     hparams_labeled
    hparam_sampler = list(itertools.product(*list(hparams_labeled.values())))
    
    print('BEGINNING HPARAM SEARCH THROUGH A TOTAL OF ',len(hparam_sampler),' INDIVIDUAL HPARAM PERMUTATIONS.')
    print('#'*20)
    print('#'*20)
#     random.shuffle(hparam_sampler)
    #########################################
    #########################################
    for num_finished, hparam in enumerate(hparam_sampler):
        hparam = {k:v for k,v in hparam}
        
        args.model_name = hparam['model_names']
        args.dataset_names = hparam['dataset_names']
        args.base_learning_rate = hparam['learning_rates']
        args.batch_size = hparam['batch_sizes']
#         regularizer = {hparam['regularizer_types']:hparam['regularizer_params']}        
        run_name=f'{args.model_name}-{args.dataset_names}-{args.color_type}-lr_{args.base_learning_rate}-bsz_{args.batch_size}'
        
        with mlflow.start_run(run_name=run_name, nested=True):

            num_channels=3
            if args.model_name=='vgg16':
                target_size=(224,224)
                if args.color_type=='grayscale':
                    num_channels=1
            elif 'resnet' in args.model_name:
                target_size=(224,224)
            elif args.model_name=='xception':
                target_size=(299,299)
            else:
                target_size=(224,224)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            dataset_config_source_domain = DatasetConfig(dataset_name=args.dataset_names[0],
                                            label_col='family',
                                            target_size=target_size,
                                            num_channels=num_channels,
                                            grayscale=(args.color_type=='grayscale'),
                                            low_class_count_thresh=args.low_class_count_thresh,
                                            data_splits={'val_size':0.2,'test_size':0.0},
                                            tfrecord_root_dir=args.tfrecord_dir,
                                            num_shards=10)

            dataset_config_target_domain = DatasetConfig(dataset_name=args.dataset_names[1],
                                            label_col='family',
                                            target_size=target_size,
                                            num_channels=num_channels,
                                            grayscale=(args.color_type=='grayscale'),
                                            low_class_count_thresh=args.low_class_count_thresh,
                                            data_splits={'val_size':0.2,'test_size':0.2},
                                            tfrecord_root_dir=args.tfrecord_dir,
                                            num_shards=10)


            train_config = TrainConfig(model_name=args.model_name,
                                       model_dir=args.model_dir,
                                       batch_size=args.batch_size,
                                       frozen_layers=None,
                                       base_learning_rate=args.base_learning_rate,
                                       buffer_size=500,
                                       num_epochs=args.num_epochs,
                                       preprocessing=True,
                                       augment_images=True,
                                       augmentations=['rotate','flip'],
                                       regularization=regularizer,
                                       seed=5,
                                       verbose=True)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project',
                                                      'experiments',
                                                      'domain_transfer',
                                                      '-'.join([args.model_name,args.color_type]),
                                                      '-'.join(args.dataset_names),
                                                      f'lr-{args.base_learning_rate}-bsz_{args.batch_size}',
                                                      current_time)


            experiment_configs = [ExperimentConfig(dataset_config_source_domain, train_config),
                                  ExperimentConfig(dataset_config_target_domain, train_config)]
                
                    
            reset_eager_session()

            mlflow.tensorflow.autolog()

    #         mlflow.log_params(experiment_config)

            print(f'BEGINNING: \nSOURCE DATASET:{args.dataset_names[0]}|\nTARGET DATASET:{args.dataset_names[1]}|\nMODEL:{args.model_name}|bsz:{args.batch_size}|lr:{args.base_learning_rate}|num_channels:{num_channels}|Color_type={args.color_type}|regularizer={regularizer}')
            print('-'*30)

            trainer = main(experiment_configs, experiment_dir)

            histories = trainer.histories


            mlflow.log_params(args.__dict__)
            
            try:
                for k, v in trainer.configs.items():
                    mlflow.log_params(v)
                    print('logged', k)
            except:
                mlflow.log_params(experiment_configs[0])
                mlflow.log_params(experiment_configs[1])
            
#             for k, v in trainer.configs['source'].items():
#                 mlflow.log_params(v)
#                 print('logged source', k)
                
#             for k, v in trainer.configs['target'].items():
#                 mlflow.log_params(v)
#                 print('logged target', k)
                
            #Log training history for both source and target domains
            for domain, history in histories.items():
                mlflow_log_history(history, history_name=domain)

            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#     #########################################
#     #########################################
#     for model_name in model_names:
#         for dataset_name in dataset_names:
#             for lr in learning_rates:
#                 for bsz in batch_sizes:
#                     with mlflow.start_run(run_name=f'{args.model_name}-{args.dataset_name}-{color_type}-lr_{args.base_learning_rate}-bsz_{args.batch_size}', nested=True):
#                         for regularizer in regularizations:
#                             args.batch_size = bsz            
#                             args.base_learning_rate = lr
#                             args.dataset_name = dataset_name
#                             args.model_name = model_name
#                             print('model_name=',args.model_name)

#                             if args.model_name=='vgg16':
#                                 target_size=(224,224)
#                             elif 'resnet' in args.model_name:
#                                 target_size=(224,224)
#                             elif args.model_name=='xception':
#                                 target_size=(299,299)
#                             else:
#                                 target_size=(224,224)

#                             histories = []

#                             current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#                             experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',args.model_name,args.dataset_name,color_type,f'lr-{args.base_learning_rate}-bsz_{args.batch_size}',current_time)
#                             reset_eager_session()

#                             dataset_config = DatasetConfig(dataset_name=args.dataset_name,
#                                                            label_col='family',
#                                                            target_size=target_size,
#                                                            num_channels=args.num_channels,
#                                                            grayscale=True,
#                                                            low_class_count_thresh=args.low_class_count_thresh,
#                                                            data_splits={'val_size':0.2,'test_size':0.2},
#                                                            tfrecord_root_dir=args.tfrecord_dir,
#                                                            num_shards=10)

#                             train_config = TrainConfig(model_name=args.model_name,
#                                                        model_dir=args.model_dir,
#                                                        batch_size=args.batch_size,
#                                                        frozen_layers=None, #(0,-4),
#                                                        base_learning_rate=args.base_learning_rate,
#                                                        buffer_size=500,
#                                                        num_epochs=args.num_epochs,
#                                                        preprocessing=True,
#                                                        augment_images=True,
#                                                        augmentations=['rotate','flip'],
#                                                        regularization=regularizer,
#                                                        seed=5,
#                                                        verbose=True)

#                             experiment_config = ExperimentConfig(dataset_config=dataset_config,
#                                                                  train_config=train_config)            

#                             mlflow.tensorflow.autolog()

#     #                         mlflow.log_params(experiment_config)

#                             print(f'BEGINNING: DATASET:{args.dataset_name}|MODEL:{args.model_name}|bsz:{args.batch_size}|lr:{args.base_learning_rate}|num_channels:{args.num_channels}|Grayscale={experiment_config.grayscale}')
#                             print('-'*30)

#                             trainer = main(experiment_config, experiment_dir)

#                             history = trainer.history

#                             histories.append((dataset_name, model_name, history))

#                             mlflow.log_params(args.__dict__)
#                             for k, v in trainer.configs.items():
#                                 mlflow.log_params(v)
#                                 print('logged ', k)
#                             mlflow_log_history(history)
            
            
            
            
            
            
            
            