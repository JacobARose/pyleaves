"""
Created on Tue Mar 17 03:23:32 2019

script: /pyleaves/pyleaves/train/csv_datasets_2-stage_transfer-learning_main.py

@author: JacobARose
"""


def main(experiment_configs, experiment_results_dir):


    ############################################
    #TODO: Moving towards defining most or all run parameters in separate config files
    ############################################
    label_encoders = {}
    for i, domain in enumerate(['source','target']):
        label_mapping_filepath = experiment_configs[i]['label_mappings']        
        label_encoders.update({domain:LabelEncoder(filepath=label_mapping_filepath)})
        print(domain, len(label_encoders[domain]))

    trainer = TransferTrainer(experiment_configs,
                              trainer_constructor=CSVTrainer,
                              label_encoders=label_encoders)

    trainer.init_model_builder(domain='source')

    source_model_filepath = os.path.join(trainer.model_manager.model_dir,trainer.model_name+'_source_model.h5')
    target_model_filepath = os.path.join(trainer.model_manager.model_dir,trainer.model_name+'_target_model.h5')    
    
    source_train_data = trainer.get_data_loader(domain='source', subset='train')
    source_val_data = trainer.get_data_loader(domain='source', subset= 'val')

    #Get parameters for fitting and callbacks
    fit_params = trainer.get_fit_params(domain='source')
    callbacks = get_callbacks(weights_best=os.path.join(trainer.model_manager.model_dir,'source_domain_weights_best.h5'), 
                                  logs_dir=os.path.join(experiment_results_dir,'tensorboard_logs'), 
                                  restore_best_weights=True)

    # TRAIN ON SOURCE DOMAIN

    history = trainer.fit(source_train_data,
                     steps_per_epoch = fit_params['steps_per_epoch'],
                     epochs=fit_params['epochs'],
                     validation_data=source_val_data,
                     validation_steps=fit_params['validation_steps'],
                     callbacks=callbacks,
                     history_name='source'
                     )
    trainer.histories['source'] = history

    
    trainer.save_model(filepath=source_model_filepath)
    #######################################################################
    # TARGET DOMAIN
    
    trainer.load_model(filepath=source_model_filepath)

    target_train_data = trainer.get_data_loader(domain='target', subset='train')
    target_val_data = trainer.get_data_loader(domain='target', subset= 'val')
    target_test_data = trainer.get_data_loader(domain='target', subset='test')

    fit_params = trainer.get_fit_params(domain='target')
    callbacks = get_callbacks(weights_best=os.path.join(trainer.model_manager.model_dir,'target_domain_weights_best.h5'), 
                                  logs_dir=os.path.join(experiment_results_dir,'tensorboard_logs'), 
                                  restore_best_weights=True)

    num_test_samples = trainer.domains['target'].metadata_splits['test']['num_samples']
    num_steps = num_test_samples//trainer.domains['target'].config['batch_size']
    test_results = []
    test_results += [
                      trainer.evaluate(target_test_data, steps=num_steps, log_name='0-shot_test')
                    ]
    
    # FINETUNE ON TARGET DOMAIN

    history = trainer.fit(target_train_data,
                          steps_per_epoch = fit_params['steps_per_epoch'],
                          epochs=fit_params['epochs'],
                          validation_data=target_val_data,
                          validation_steps=fit_params['validation_steps'],
                          callbacks=callbacks,
                          history_name='target'
                          )
    
    trainer.histories['target'] = history
    
    test_results += [
                     trainer.evaluate(target_test_data, steps=num_steps, log_name='test_acc')
                    ]
    trainer.test_results = test_results
    
    return trainer

if __name__=='__main__':
    '''
    Example:

    python /home/jacob/pyleaves/pyleaves/train/csv_datasets_2-stage_transfer-learning_main.py -d Leaves PNAS -m resnet_50_v2 --experiment_root_dir /media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments -gpu 1 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 10 -r l2 -r_p 0.001 --experiment CSVTransferBaselines


    python /home/jacob/pyleaves/pyleaves/train/csv_datasets_2-stage_transfer-learning_main.py -d Fossil PNAS -m resnet_50_v2 --experiment_root_dir /media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments -gpu 0 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 10 -r l2 -r_p 0.001 --experiment CSVTransferBaselines

    python /home/jacob/pyleaves/pyleaves/train/csv_datasets_2-stage_transfer-learning_main.py -d Fossil Leaves -m resnet_50_v2 --experiment_root_dir /media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments -gpu 6 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 10 -r l2 -r_p 0.001 --experiment CSVTransferBaselines


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
    import itertools
    import random
    from collections import OrderedDict

    seed_value = 1837
        
    parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_config', default='', type=str, nargs='?', help='Requires 2 args identifying datasets by name in order of input to the pipeline. Stage 1: train + validate, Stage 2: finetune + validate + test')
#     parser.add_argument('-m', '--model_config', default='vgg16', type=str, help='Name of model to train')

#     parser.add_argument('--dataset_name', type=str, default='PNAS')
    parser.add_argument('--experiment_root_dir', type=str, default=r'/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments')
#     parser.add_argument('--run_name', type=str, default='PNAS_Leaves')
    parser.add_argument('-d', '--dataset_names', default=['PNAS','Fossil'], type=str, nargs=2, help='Requires 2 args identifying datasets by name in order of input to the pipeline. Stage 1: train + validate, Stage 2: finetune + validate + test')
    parser.add_argument('-m', '--model_name', default='vgg16', type=str, nargs='*', help='Name of model to train')
    parser.add_argument('-gpu', '--gpu_id', default='1', type=str, help='integer number of gpu to train on')
#     parser.add_argument('-ch', '--num_channels', default=3, type=int, help='Number of input channels, either 1 for grayscale, or 3 for rgb')
    parser.add_argument('-c', '--color_type', default='grayscale', type=str, help='grayscale or rgb')
    parser.add_argument('-bsz', '--batch_size', default=64, type=int, nargs='*', help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default=1e-4, nargs='*', type=float, help="Starting learning rate, <float> for a single value or 'all' to loop through a hardcoded range of values")
    parser.add_argument('-thresh', '--low_class_count_thresh', default=10, type=int)
    parser.add_argument('-r', '--regularizations', default='l2', type=str, help='comma separated list of regularizers to search through. Enter combinations of l1 and l2, enter anything else for None.')
    parser.add_argument('-r_p', '--r_params', default='0.001', type=str, nargs='*', help='comma separated list of regularizer strengths to search through. Enter combinations of floats.') #3
    parser.add_argument('-epochs', '--num_epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('-exp', '--experiment', default='CSVTransferBaselines', type=str, help=r"Name of new or existing MLFlow experiment to log results into. TODO: Add None option")
    parser.add_argument('--data_db_path', default=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db', type=str, help='Directory in which to save/load models and/or model weights')
    parser.add_argument('--model_dir', default=r'/media/data_cifs/jacob/Fossil_Project/models', type=str, help='Directory in which to save/load models and/or model weights')
    parser.add_argument('-tfrec', '--tfrecord_dir', default=r'/media/data/jacob/Fossil_Project/tfrecord_data', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    parser.add_argument('-f',default='')
    args = parser.parse_args()

    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) ####SHOULD THIS BE AN INT???
    tf.compat.v1.enable_eager_execution()

    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
    
    import pyleaves
    from pyleaves.utils import ensure_dir_exists, process_hparam_args
    ####
    from pyleaves.data_pipeline.preprocessing import LabelEncoder    
    from pyleaves.leavesdb.tf_utils.tf_utils import reset_eager_session
    from pyleaves.utils.csv_utils import gather_run_data, load_csv_data
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig
    from pyleaves.train.transfer_trainer import TransferTrainer
    from pyleaves.train.csv_trainer import CSVTrainer
    from pyleaves.analysis.mlflow_utils import mlflow_log_params_dict, mlflow_log_history, mlflow_log_best_history
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
    search_params=['base_learning_rate','batch_size']
    
    if args.model_name == 'all':
        args.model_name = ['resnet_50_v2','resnet_152_v2', 'vgg16']
    elif type(args.model_name)==str:
        search_params.append('model_name')
    #########################################
    for d in args.dataset_names:
        assert d in ['PNAS','Leaves','Fossil']
    args.dataset_names = [args.dataset_names, args.dataset_names[::-1]]
    args.run_names = ['_'.join(names) for names in args.dataset_names]
    #########################################
    regularizer = {args.regularizations:args.r_params}
    
    new_args = process_hparam_args(args, search_params=search_params)
    
    hparams = OrderedDict({
                           'model_names':args.model_name,
                           'dataset_names':args.dataset_names,
#                            'run_names':args.run_names,
                           'learning_rates':args.base_learning_rate,
                           'batch_sizes':args.batch_size
                           }
                          )
    hparams_labeled = OrderedDict()
    for k, v in hparams.items():
        hparams_labeled[k] = list(itertools.product([k],v))

    hparam_sampler = list(
            itertools.product(
                            *list(
                                hparams_labeled.values()
                                )
                            )
                        )
    print('BEGINNING HPARAM SEARCH THROUGH A TOTAL OF ',len(hparam_sampler),' INDIVIDUAL HPARAM PERMUTATIONS.')
    print('#'*20)
    print('#'*20)  #     random.shuffle(hparam_sampler)
    #########################################

    for num_finished, hparam in enumerate(hparam_sampler):
        hparam = {k:v for k,v in hparam}
        
        args.model_name = hparam['model_names']
        args.dataset_names = hparam['dataset_names']
        args.run_name = '_'.join(hparam['dataset_names'])
        args.base_learning_rate = hparam['learning_rates']
        args.batch_size = hparam['batch_sizes']

        mlflow_run_name=f'{args.model_name}-{args.run_name}-{args.color_type}-lr_{args.base_learning_rate}-bsz_{args.batch_size}'        
#         run_name=f'{args.model_name}-{args.dataset_names}-{args.color_type}-lr_{args.base_learning_rate}-bsz_{args.batch_size}'
        
        with mlflow.start_run(run_name=mlflow_run_name, nested=True):


            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            experiment_name = os.path.basename(args.experiment_root_dir)

            experiment_results_dir = os.path.join(args.experiment_root_dir,
                                                  'results',
                                                  '-'.join([args.model_name,args.color_type]),
                                                  '_'.join(args.dataset_names),
                                                  f'lr-{args.base_learning_rate}-bsz_{args.batch_size}',
                                                  current_time)

            run_records = gather_run_data(args.experiment_root_dir, run=args.run_name, return_type='records')
            source_run_record = [rec for rec in run_records if rec['domain']=='source'][0]
            target_run_record = [rec for rec in run_records if rec['domain']=='target'][0]
            
            dataset_config_source_domain = DatasetConfig(experiment_name=experiment_name,
                                            **source_run_record,
                                            experiment_root_dir=args.experiment_root_dir,
                                            label_col='family',
#                                             target_size=target_size,
#                                             num_channels=num_channels,
                                            grayscale=(args.color_type=='grayscale'),
                                            color_type=args.color_type,
                                            low_class_count_thresh=args.low_class_count_thresh,
                                            data_splits={'val_size':0.2,'test_size':0.0},
                                            tfrecord_root_dir=args.tfrecord_dir,
#                                             data_db_path=args.data_db_path,
                                            num_shards=10)

            dataset_config_target_domain = DatasetConfig(experiment_name=experiment_name,
                                            **target_run_record,
                                            experiment_root_dir=args.experiment_root_dir,
                                            label_col='family',
#                                             target_size=target_size,
#                                             num_channels=num_channels,
                                            grayscale=(args.color_type=='grayscale'),
                                            color_type=args.color_type,
                                            low_class_count_thresh=args.low_class_count_thresh,
                                            data_splits={'val_size':0.2,'test_size':0.2},
                                            tfrecord_root_dir=args.tfrecord_dir,
#                                             data_db_path=args.data_db_path,
                                            num_shards=10)


            train_config_source_domain = TrainConfig(model_name=args.model_name,
                                       model_dir=args.model_dir,
                                       batch_size=args.batch_size,
                                       frozen_layers=None,
                                       base_learning_rate=args.base_learning_rate,
                                       buffer_size=500,
                                       num_epochs=args.num_epochs,
                                       preprocessing=True,
                                       x_col='x',
                                       y_col='y',
                                       augment_images=True,
                                       augmentations=['rotate','flip'],
                                       regularization=regularizer,
                                       seed=5,
                                       verbose=True)

            train_config_target_domain = TrainConfig(model_name=args.model_name,
                                       model_dir=args.model_dir,
                                       batch_size=args.batch_size,
                                       frozen_layers=None,
                                       base_learning_rate=args.base_learning_rate*0.5,
                                       buffer_size=500,
                                       num_epochs=args.num_epochs,
                                       preprocessing=True,
                                       x_col='x',
                                       y_col='y',
                                       augment_images=True,
                                       augmentations=['rotate','flip'],
                                       regularization=regularizer,
                                       seed=5,
                                       verbose=True)            
            
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            experiment_configs = [ExperimentConfig(dataset_config_source_domain, train_config_source_domain),
                                  ExperimentConfig(dataset_config_target_domain, train_config_target_domain)]
                                    
            reset_eager_session()

            mlflow.tensorflow.autolog()

    #         mlflow.log_params(experiment_config)

            print(f'BEGINNING: \nSOURCE DATASET:{args.dataset_names[0]}|\nTARGET DATASET:{args.dataset_names[1]}|\nMODEL:{args.model_name}|bsz:{args.batch_size}|lr:{args.base_learning_rate}|Color_type={args.color_type}|regularizer={regularizer}')
            print('-'*30)

            trainer = main(experiment_configs, experiment_results_dir)

            histories = trainer.histories

            mlflow.log_params(args.__dict__)
            
            try:
                mlflow_log_params_dict(trainer.config)
            except:
                mlflow_log_params_dict(experiment_configs[0])
                mlflow_log_params_dict(experiment_configs[1])

                
                
                
#             for k, v in trainer.configs['source'].items():
#                 mlflow.log_params(v)
#                 print('logged source', k)
                
#             for k, v in trainer.configs['target'].items():
#                 mlflow.log_params(v)
#                 print('logged target', k)
                
            #Log training history for both source and target domains

        
#             for domain, history in histories.items():
#                 mlflow_log_history(history, history_name=domain)

            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
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
            
            
            
            
            
            
            
            