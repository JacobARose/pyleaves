"""
Created on Mon Feb 10 03:23:32 2019

script: pyleaves/pyleaves/train/example_train.py

@author: JacobARose
"""


def main(experiment_config, experiment_dir):


    ############################################
    #TODO: Moving towards defining most or all run parameters in separate config files
    ############################################


    trainer = BaseTrainer(experiment_config=experiment_config)
    
#     for subset, paths in trainer.tfrecord_files.items():
#         if experiment_config.verbose: print(subset)
#         for path in paths:
#             if experiment_config.verbose: print('\t',path)
#             mlflow.log_artifact(path,f'artifacts/{subset}')    

    train_data = trainer.get_data_loader(subset='train')
    val_data = trainer.get_data_loader(subset= 'val')
    test_data = trainer.get_data_loader(subset='test')

#     debug=False
#     if debug:
#         if tf.executing_eagerly():
#             batch_imgs, batch_labels = next(iter(val_data))
#         else:
#             validation_iterator = val_data.make_one_shot_iterator()
#             val_data_next = validation_iterator.get_next()
#             sess = tf.compat.v1.Session()
#             batch_imgs, batch_labels = sess.run(val_data_next)

#         from pyleaves.analysis.img_utils import plot_image_grid

#         plot_image_grid(batch_imgs, [np.argmax(l) for l in batch_labels], 8, 8)
#         for i in range(64):
#             img = batch_imgs[i,...]
#             print(i, f'min = {np.min(img):.2f}, max = {np.max(img):.2f}, mean = {np.mean(img):.2f}, std = {np.std(img):.2f}')            
            
#         #From [-1.0,1.0] to [0,255]
#         uint_imgs = np.array(batch_imgs)
#         uint_imgs += 1
#         uint_imgs /= 2
#         uint_imgs *= 255
#         uint_imgs = uint_imgs.astype(np.uint8)

#         print(f'min = {np.min(batch_imgs):.2f}, max = {np.max(batch_imgs):.2f}, mean = {np.mean(batch_imgs):.2f}, std = {np.std(batch_imgs):.2f}')
#         print(f'min = {np.min(uint_imgs)}, max = {np.max(uint_imgs)}, mean = {np.mean(uint_imgs):.2f}, std = {np.std(uint_imgs):.2f}')

#         plot_image_grid(uint_imgs, [np.argmax(l) for l in batch_labels], 8, 8)
    
    trainer.init_model_builder()
    
#     model_config = trainer.get_model_config('train')
    fit_params = trainer.get_fit_params()
    callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'weights_best.h5'), 
                              logs_dir=os.path.join(experiment_dir,'tensorboard_logs'), 
                              restore_best_weights=False,
                              val_data=None)

#     model_name = model_config.model_name
#     print('model_config:\n',json.dumps(model_config,indent=4))
    
#     if model_name is 'vgg16':
#         model_builder = VGG16GrayScale(model_config)
#         model = model_builder.build_model()
    
#     elif model_name.startswith('resnet'):
#         model_builder = ResNet(model_config)
#         model = model_builder.build_model()
    
#     else:
#         model = build_model(**model_config)
    
    
    history = trainer.model.fit(train_data,
                 steps_per_epoch = fit_params['steps_per_epoch'],
                 epochs= fit_params['epochs'],
                 validation_data=val_data,
                 validation_steps=fit_params['validation_steps'],
                 callbacks=callbacks
                 )
    
#     trainer.config['model_config'] = model_config
#     trainer.config.train_config['fit_params'] = fit_params
    trainer.history = history
    
    return trainer

if __name__=='__main__':
    '''
    Example:
    python /home/jacob/pyleaves/pyleaves/train/example_train.py -d all -m all -gpu 3 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 20 -r l2 -r_p 0.001 --experiment BaselinesGrayScale --data_db_path /home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db

    python /home/jacob/pyleaves/pyleaves/train/example_train.py -d Leaves2020 -m resnet_50_v2 -gpu 3 -bsz 64 -lr 1e-4 --color_type grayscale -thresh 20 -r l2 -r_p 0.001 --experiment BaselinesGrayScale --data_db_path /home/jacob/pyleaves/pyleaves/leavesdb/resources/converted_updated_leavesdb.db

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
    parser.add_argument('-gpu', '--gpu_id', default='0', type=str, help='integer number of gpu to train on')
#     parser.add_argument('-ch', '--num_channels', default=3, type=int, help='Number of input channels, either 1 for grayscale, or 3 for rgb')
    parser.add_argument('-c', '--color_type', default='grayscale', type=str, help='grayscale or rgb')
    parser.add_argument('-bsz', '--batch_size', default='64', type=str, help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default='1e-4', type=str, help="Starting learning rate, <float> for a single value or 'all' to loop through a hardcoded range of values")
    parser.add_argument('-thresh', '--low_class_count_thresh', default=10, type=int) #3
    parser.add_argument('-r', '--regularizations', default='l2', type=str, help='comma separated list of regularizers to search through. Enter combinations of l1 and l2, enter anything else for None.') #3
    parser.add_argument('-r_p', '--r_params', default='0.001', type=str, help='comma separated list of regularizer strengths to search through. Enter combinations of floats.') #3
    parser.add_argument('-epochs', '--num_epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('-exp', '--experiment', default='Baselines', type=str, help=r"Name of new or existing MLFlow experiment to log results into. TODO: Add None option")
    parser.add_argument('--data_db_path', default=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db', type=str, help='Directory in which to save/load models and/or model weights')
    parser.add_argument('--model_dir', default=r'/media/data_cifs/jacob/Fossil_Project/models', type=str, help='Directory in which to save/load models and/or model weights')
    parser.add_argument('-tfrec', '--tfrecord_dir', default=r'/media/data/jacob/Fossil_Project/tfrecord_data', type=str, help=r"Parent dir above the location that's intended for saving the TFRecords for this dataset")
    parser.add_argument('-f',default='')
    args = parser.parse_args()

    import datetime
    import json
    import numpy as np
    import os
    import tensorflow as tf
    
    
#     args.base_learning_rate = 'all'
    
#     config = tf.ConfigProto(device_count = {'GPU': args.gpu_id})

#     args.gpu_id=2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) ####SHOULD THIS BE AN INT???
    tf.compat.v1.enable_eager_execution()
    from pyleaves.utils import ensure_dir_exists # set_visible_gpus,
#     set_visible_gpus([args.gpu_id])
    ####

    from pyleaves.leavesdb.tf_utils.tf_utils import reset_eager_session
    
    from pyleaves.models.resnet import ResNet, ResNetGrayScale
    from pyleaves.models.vgg16 import VGG16, VGG16GrayScale    
    from pyleaves.models.keras_models import build_model
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig
    from pyleaves.train.base_trainer import BaseTrainer, BaseTrainer_v1
    from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history

    import mlflow
    import mlflow.tensorflow
    
    tracking_dir = r'/media/data/jacob/Fossil_Project/experiments/mlflow'
    ensure_dir_exists(tracking_dir)
    mlflow.set_tracking_uri(tracking_dir)
    print(mlflow.tracking.get_tracking_uri())
    mlflow.set_experiment(args.experiment)
#     print(mlflow.get_artifact_uri())
    
#     if args.num_channels==3:
#         color_type = 'rgb'
#     else:
#         color_type = 'grayscale'
    
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
    if args.dataset_name == 'all':
        dataset_names = ['PNAS', 'Fossil', 'Leaves2020'] #'Leaves']
    else:
        dataset_names = [args.dataset_name]
        
    #########################################    
    learnrates = args.base_learning_rate.split(',')
    learning_rates = []
    for lr in learnrates:
        try:
            learning_rates.append(float(lr))
        except ValueError:
            if args.base_learning_rate == 'all':
                learning_rates = [1e-3, 1e-4,1e-5]
            break
    if len(learning_rates)==0:
        learning_rates = [1e-4]
        print(f'Undefined Learning Rate option provided. Continuing with default {learning_rates[0]}')
    #########################################
    if args.batch_size == 'all':
        batch_sizes = [64, 128]
    else:
        batch_sizes = [int(args.batch_size)]    
    #########################################
    reg_list = args.regularizations.split(',')
    regularizer_types=[]
    for r in reg_list:
        if r in ['l1','l2']:
            regularizer_types.append(r)
    
    if len(regularizer_types)==0:
        regularizer_types = [None]    
    
    r_params=[]
    for r_param in args.r_params.split(','):
        try:
            r_params.append(float(r_param))
        except ValueError:
            if r_param=='all':
                r_params = [0.001, 0.01]
                break
    if len(r_params)==0:
        r_params = [0.001]
        print(f'Undefined Regularization option provided. Continuing with default {r_params[0]}')
                    
    
    hparams = OrderedDict()
    hparams['model_name_list'] = model_names #['resnet_50_v2','resnet_152_v2', 'vgg16', 'xception', 'shallow']
    hparams['dataset_name_list'] = dataset_names #['PNAS', 'Fossil', 'Leaves']
    hparams['learning_rate_list'] = learning_rates #[1e-3, 1e-4,1e-5]
    hparams['batch_size_list'] = batch_sizes #[64, 128]
    hparams['regularizer_type_list'] = regularizer_types #['l1','l2']
    hparams['regularizer_param_list'] = r_params #[0.001, 0.01]
    
    hparams_labeled = OrderedDict()
    for k, v in hparams.items():
        hparams_labeled[k] = list(itertools.product([k.replace('_list','s')],v))
    hparams_labeled
    hparam_sampler = list(itertools.product(*list(hparams_labeled.values())))
    
    print('BEGINNING HPARAM SEARCH THROUGH A TOTAL OF ',len(hparam_sampler),' INDIVIDUAL HPARAM PERMUTATIONS.')
    print('#'*20,'\n','#'*20)
#     random.shuffle(hparam_sampler)
    #########################################
    #########################################
    for num_finished, hparam in enumerate(hparam_sampler):
        hparam = {k:v for k,v in hparam}
#         break
        args.model_name = hparam['model_names']
        args.dataset_name = hparam['dataset_names']
        args.base_learning_rate = hparam['learning_rates']
        args.batch_size = hparam['batch_sizes']
        regularizer = {hparam['regularizer_types']:hparam['regularizer_params']}        
        run_name=f'{args.model_name}-{args.dataset_name}-{args.color_type}-lr_{args.base_learning_rate}-bsz_{args.batch_size}'
        
        with mlflow.start_run(run_name=run_name, nested=True):

#             num_channels=3
#             if args.model_name=='vgg16':
#                 target_size=(224,224)
#                 if args.color_type=='grayscale':
#                     num_channels=1
#             elif 'resnet' in args.model_name:
#                 target_size=(224,224)
#             elif args.model_name=='xception':
#                 target_size=(299,299)
#             else:
#                 target_size=(224,224)

            histories = []
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project',
                                          'experiments',
                                          args.model_name,
                                          args.dataset_name,
                                          args.color_type,
                                          f'lr-{args.base_learning_rate}-bsz_{args.batch_size}',
                                          current_time)
            reset_eager_session()

            dataset_config = DatasetConfig(dataset_name=args.dataset_name,
                                            label_col='family',
#                                             target_size=target_size,
#                                             num_channels=num_channels,
                                            grayscale=(args.color_type=='grayscale'),
                                            low_class_count_thresh=args.low_class_count_thresh,
                                            data_splits={'val_size':0.2,'test_size':0.2},
                                            tfrecord_root_dir=args.tfrecord_dir,
                                            data_db_path=args.data_db_path,
                                            num_shards=10)

            train_config = TrainConfig(model_name=args.model_name,
                                       model_dir=args.model_dir,
                                       batch_size=args.batch_size,
                                       frozen_layers=None, #(0,-4),
                                       base_learning_rate=args.base_learning_rate,
                                       buffer_size=500,
                                       num_epochs=args.num_epochs,
                                       preprocessing=True,
                                       augment_images=True,
                                       augmentations=['rotate','flip'],
                                       regularization=regularizer,
                                       seed=5,
                                       verbose=True)

            experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                                 train_config=train_config)

            mlflow.tensorflow.autolog()

    #         mlflow.log_params(experiment_config)

            print(f'BEGINNING: DATASET:{args.dataset_name}|MODEL:{args.model_name}|bsz:{args.batch_size}|lr:{args.base_learning_rate}|Color_type={args.color_type}|regularizer={regularizer}')
            print('-'*30)

            trainer = main(experiment_config, experiment_dir)

            history = trainer.history

            histories.append((args.dataset_name, args.model_name, history))

            mlflow.log_params(args.__dict__)
            for k, v in trainer.configs.items():
                mlflow.log_params(v)
                print('logged ', k)
            mlflow_log_history(history)

            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
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
            
            
            
            
            
            
            
            