"""
Created on Tue Mar 17 03:23:32 2019

script: /pyleaves/pyleaves/train/csv_datasets_train.py

@author: JacobARose
"""


def main(experiment_config, experiment_results_dir):


    ############################################
    #TODO: Moving towards defining most or all run parameters in separate config files
    ############################################
    
    domain = experiment_config.domain
    label_mapping_filepath = experiment_config['label_mappings']
    
    label_encoder = LabelEncoder(filepath=label_mapping_filepath)
    print(label_encoder)
    trainer = CSVTrainer(experiment_config, label_encoder=label_encoder)

    trainer.init_model_builder()

    model_filepath = os.path.join(trainer.model_manager.model_dir,trainer.model_name+'_'+domain+'_model.h5')
    
    train_data = trainer.get_data_loader(subset='train')
    val_data = trainer.get_data_loader(subset= 'val')
    test_data = trainer.get_data_loader(subset= 'test')

    #Get parameters for fitting and callbacks
    fit_params = trainer.get_fit_params()
    callbacks = get_callbacks(weights_best=os.path.join(trainer.model_manager.model_dir,trainer.model_name+'_'+domain+'_model_weights_best.h5'), 
                                  logs_dir=os.path.join(experiment_results_dir,'tensorboard_logs'), 
                                  restore_best_weights=True)

    history = trainer.fit(train_data,
                     steps_per_epoch = fit_params['steps_per_epoch'],
                     epochs=fit_params['epochs'],
                     validation_data=val_data,
                     validation_steps=fit_params['validation_steps'],
                     callbacks=callbacks) #,
#                      history_name=domain
#                      )
    trainer.histories[domain] = history

    
    trainer.save_model(filepath=model_filepath)
    #######################################################################
    # TARGET DOMAIN
    
    #trainer.load_model(filepath=source_model_filepath)
    num_test_samples = trainer.metadata_splits['test']['num_samples']
    num_steps = num_test_samples//trainer.config['batch_size']
    test_results = [trainer.evaluate(test_data, steps=num_steps, log_name='test')]#'trained-on-source_train--evaluated-on-source_test')]
    

    trainer.test_results = test_results
    
    return trainer

if __name__=='__main__':
    '''
    Example:
    python /home/jacob/pyleaves/pyleaves/train/2-stage_transfer-learning_main.py -d PNAS Fossil -m vgg16 -gpu 0 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 20 -r l2 -r_p 0.001 --experiment TransferBaselines


python /home/jacob/pyleaves/pyleaves/train/2-stage_transfer-learning_main.py -d Leaves2020 PNAS -m resnet_50_v2 -gpu 2 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 20 -r l2 -r_p 0.001 --experiment TransferBaselines



     python /home/jacob/pyleaves/pyleaves/train/csv_datasets_train.py --run_name PNAS -m resnet_50_v2 --experiment_root_dir /media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments --experiment BaselinesCSV -gpu 0
     
     python /home/jacob/pyleaves/pyleaves/train/csv_datasets_train.py --run_name Leaves -m resnet_50_v2 --experiment_root_dir /media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments --experiment BaselinesCSV -gpu 5

    python /home/jacob/pyleaves/pyleaves/train/csv_datasets_train.py --run_name Fossil -m resnet_50_v2 --experiment_root_dir /media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments --experiment BaselinesCSV -gpu 6

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
    random.seed(6)
    
    parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_config', default='', type=str, nargs='?', help='Requires 2 args identifying datasets by name in order of input to the pipeline. Stage 1: train + validate, Stage 2: finetune + validate + test')
#     parser.add_argument('-m', '--model_config', default='vgg16', type=str, help='Name of model to train')

    parser.add_argument('--run_name', type=str, default='PNAS')
#     parser.add_argument('--dataset_name', type=str, default='PNAS')
    parser.add_argument('--experiment_root_dir', type=str, default=r'/media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments')
    parser.add_argument('-m', '--model_name', default='vgg16', type=str, nargs='*', help='Name of model to train')
    parser.add_argument('-gpu', '--gpu_id', default='0', type=str, help='integer number of gpu to train on')
#     parser.add_argument('-ch', '--num_channels', default=3, type=int, help='Number of input channels, either 1 for grayscale, or 3 for rgb')
    parser.add_argument('-c', '--color_type', default='grayscale', type=str, help='grayscale or rgb')
    parser.add_argument('-bsz', '--batch_size', default=64, type=int, nargs='*', help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default=1e-4, nargs='*', type=float, help="Starting learning rate, <float> for a single value or 'all' to loop through a hardcoded range of values")
    parser.add_argument('-thresh', '--low_class_count_thresh', default=10, type=int)
    parser.add_argument('-r', '--regularizations', default='l2', type=str, help='comma separated list of regularizers to search through. Enter combinations of l1 and l2, enter anything else for None.')
    parser.add_argument('-r_p', '--r_params', default='0.001', type=str, nargs='*', help='comma separated list of regularizer strengths to search through. Enter combinations of floats.') #3
    parser.add_argument('-epochs', '--num_epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('-exp', '--experiment', default='Baselines', type=str, help=r"Name of new or existing MLFlow experiment to log results into. TODO: Add None option")
    parser.add_argument('-tracking_dir', '--mlflow_tracking_dir', default=r'/media/data/jacob/Fossil_Project/experiments/mlflow', type=str, help=r"Absolute path of MLFlow tracking dir for logging this experiment.")
    parser.add_argument('--data_db_path', default=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db', type=str, help='Directory in which to save/load models and/or model weights')
    parser.add_argument('--model_dir', default=r'/media/data_cifs/jacob/Fossil_Project/models', type=str, help='Directory in which to save/load models and/or model weights')
    parser.add_argument('-tfrec', '--tfrecord_dir', default=r'/media/data/jacob/Fossil_Project/tfrecord_data', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    
    parser.add_argument('-f',default='')
    args = parser.parse_args()

    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) ####SHOULD THIS BE AN INT???
    tf.compat.v1.enable_eager_execution()
    import pyleaves
    from pyleaves.utils import ensure_dir_exists, process_hparam_args
    ####
    from pyleaves.data_pipeline.preprocessing import LabelEncoder
    from pyleaves.leavesdb.tf_utils.tf_utils import reset_eager_session
    from pyleaves.utils.csv_utils import gather_run_data, load_csv_data
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig, CSVDomainDataConfig, CSVFrozenRunDataConfig
    from pyleaves.train.csv_trainer import CSVTrainer
    from pyleaves.analysis.mlflow_utils import mlflow_log_params_dict, mlflow_log_history, mlflow_log_best_history
    import mlflow
    import mlflow.tensorflow
    
    ensure_dir_exists(args.mlflow_tracking_dir)
    mlflow.set_tracking_uri(args.mlflow_tracking_dir)
    mlflow.set_experiment(args.experiment)
    
    #     print(mlflow.tracking.get_tracking_uri())
    
    ############################
    #########################################
    search_params=['run_name','base_learning_rate','batch_size']
    
    if args.model_name == 'all':
        args.model_name = ['resnet_50_v2','resnet_152_v2', 'vgg16']
    elif type(args.model_name)==str:
        search_params.append('model_name')
    #########################################
    #########################################
    regularizer = {args.regularizations:args.r_params}
    
    new_args = process_hparam_args(args, search_params=search_params)
    
    hparams = OrderedDict({
                           'model_names':args.model_name,
                           'run_names':args.run_name,
                           'learning_rates':args.base_learning_rate,
                           'batch_sizes':args.batch_size
                           }
                          )
    
    hparams_labeled = OrderedDict()
    for k, v in hparams.items():
        hparams_labeled[k] = list(itertools.product([k],v))

    hparam_sampler = list(
            itertools.product(*list(hparams_labeled.values()))
                         )
    
    print('BEGINNING HPARAM SEARCH THROUGH A TOTAL OF ',len(hparam_sampler),' INDIVIDUAL HPARAM PERMUTATIONS.')
    print('#'*20)
    print('#'*20)
    #########################################

    for num_finished, hparam in enumerate(hparam_sampler):
        hparam = {k:v for k,v in hparam}
        
        args.model_name = hparam['model_names']
        args.run_name = hparam['run_names']
        args.dataset_name = args.run_name
        args.domain = args.run_name
        args.base_learning_rate = hparam['learning_rates']
        args.batch_size = hparam['batch_sizes']

        mlflow_run_name=f'{args.model_name}-{args.run_name}-{args.color_type}-lr_{args.base_learning_rate}-bsz_{args.batch_size}'
        
        with mlflow.start_run(run_name=mlflow_run_name, nested=True):

            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            experiment_name = os.path.basename(args.experiment_root_dir)

            experiment_results_dir = os.path.join(args.experiment_root_dir,
                                                  'results',
                                                  '-'.join([args.model_name,args.color_type]),
                                                  args.dataset_name,
                                                  f'lr-{args.base_learning_rate}-bsz_{args.batch_size}',
                                                  current_time)
            
#             experiment_records = gather_experiment_data(experiment_root_dir, return_type='records')
#             get_records_attribute(experiment_records, attribute_key='run')
            run_records = gather_run_data(args.experiment_root_dir, run=args.run_name, return_type='records')
            
#             get_records_attribute(run_records, attribute_key='run')
            
#             domain_config_0 = CSVDomainDataConfig(experiment_name=experiment_name,
#                                                         **run_records[0],
#                                                         grayscale=True,
#                                                         color_type='grayscale',
#                                                         num_channels=1,
#                                                         low_class_count_thresh=10,
#                                                         data_splits={'val_size':0.2,'test_size':0.2},
#                                                         num_shards=10)
            
            
#             dataset_config_domain = CSVFrozenRunDataConfig(experiment_name=experiment_name, #"single-domain_experiments",
#                                                          run=args.run_name, #"Leaves",
#                                                          experiment_root_dir=args.experiment_root_dir,
#                                                          tfrecord_root_dir=args.tfrecord_dir,
#                                                          low_class_count_thresh=10,
#                                                          data_configs={
#                                                                       args.domain: domain_config_0
#                                                                      })
#             dataset_config_domain.init_config_file()

            
            
            dataset_config = DatasetConfig(experiment_name=experiment_name,
                                           **run_records[0],
                                           experiment_root_dir=args.experiment_root_dir,
                                            label_col='family',
#                                             target_size=target_size,
#                                             num_channels=num_channels,
                                            grayscale=(args.color_type=='grayscale'),
                                            color_type=args.color_type,
                                            low_class_count_thresh=args.low_class_count_thresh,
                                            data_splits={'val_size':0.0,'test_size':0.5},
                                            tfrecord_root_dir=args.tfrecord_dir,
                                            num_shards=10)
            
            
            
            train_config  = TrainConfig(model_name=args.model_name,
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
            
            experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                                 train_config=train_config)

            reset_eager_session()

            mlflow.tensorflow.autolog()

    #         mlflow.log_params(experiment_config)

            print(f'BEGINNING: DATASET:{args.dataset_name}|MODEL:{args.model_name}|bsz:{args.batch_size}|lr:{args.base_learning_rate}|Color_type={args.color_type}|regularizer={regularizer}')
            print('-'*30)

            trainer = main(experiment_config, experiment_results_dir)

            histories = trainer.histories

            mlflow.log_params(args.__dict__)
            
            try:
                mlflow_log_params_dict(trainer.config)
#                 for k, v in trainer.configs.items():
#                     mlflow.log_params(v)
#                     print('logged', k)
            except:
                mlflow_log_params_dict(experiment_config)

                
                
