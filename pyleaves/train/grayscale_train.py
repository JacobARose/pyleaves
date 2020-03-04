"""
Created on Mon Mar 3 11:25:32 2019

script: pyleaves/pyleaves/train/grayscale_train.py

@author: JacobARose
"""


def main(experiment_config, experiment_dir):


    trainer = BaseTrainer(experiment_config=experiment_config)
    
#     for subset, paths in trainer.tfrecord_files.items():
#         if experiment_config.verbose: print(subset)
#         for path in paths:
#             if experiment_config.verbose: print('\t',path)
#             mlflow.log_artifact(path,f'artifacts/{subset}')    

    train_data = trainer.get_data_loader(subset='train')
    val_data = trainer.get_data_loader(subset= 'val')
    test_data = trainer.get_data_loader(subset='test')

    debug=False
    if debug:
        if tf.executing_eagerly():
            batch_imgs, batch_labels = next(iter(val_data))
        else:
            validation_iterator = val_data.make_one_shot_iterator()
            val_data_next = validation_iterator.get_next()
            sess = tf.compat.v1.Session()
            batch_imgs, batch_labels = sess.run(val_data_next)

        from pyleaves.analysis.img_utils import plot_image_grid

        plot_image_grid(batch_imgs, [np.argmax(l) for l in batch_labels], 8, 8)
        for i in range(64):
            img = batch_imgs[i,...]
            print(i, f'min = {np.min(img):.2f}, max = {np.max(img):.2f}, mean = {np.mean(img):.2f}, std = {np.std(img):.2f}')
            
        #From [-1.0,1.0] to [0,255]
        uint_imgs = np.array(batch_imgs)
        uint_imgs += 1
        uint_imgs /= 2
        uint_imgs *= 255
        uint_imgs = uint_imgs.astype(np.uint8)

        print(f'min = {np.min(batch_imgs):.2f}, max = {np.max(batch_imgs):.2f}, mean = {np.mean(batch_imgs):.2f}, std = {np.std(batch_imgs):.2f}')
        print(f'min = {np.min(uint_imgs)}, max = {np.max(uint_imgs)}, mean = {np.mean(uint_imgs):.2f}, std = {np.std(uint_imgs):.2f}')

        plot_image_grid(uint_imgs, [np.argmax(l) for l in batch_labels], 8, 8)
    
    
    
    model_params = trainer.get_model_params('train')
    fit_params = trainer.get_fit_params()
    callbacks = get_callbacks(weights_best=os.path.join(experiment_dir,'weights_best.h5'), 
                              logs_dir=os.path.join(experiment_dir,'tensorboard_logs'), 
                              restore_best_weights=False,
                              val_data=None) #val_data,
#                               freq=10)

    print('model_params',model_params)
    
    model_builder = VGG16Grayscale(model_params)
    model = model_builder.build_model()
    
    
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
    parser.add_argument('-ch', '--num_channels', default=1, type=int, help='Number of input channels, either 1 for grayscale, or 3 for rgb')    
    parser.add_argument('-bsz', '--batch_size', default=64, type=int, help='Batch size. What else do you need to know?')
    parser.add_argument('-lr', '--base_learning_rate', default='1e-4', type=str, help="Starting learning rate, <float> for a single value or 'all' to loop through a hardcoded range of values")
    parser.add_argument('-thresh', '--low_class_count_thresh', default=10, type=int) #3
    parser.add_argument('-epochs', '--num_epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('-f',default='')
    args = parser.parse_args()

    import datetime
    import numpy as np
    import os
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()

    from pyleaves.utils import set_visible_gpus, ensure_dir_exists
    args.gpu_id=2
    set_visible_gpus([args.gpu_id])
    ####
    from pyleaves.leavesdb.tf_utils.tf_utils import reset_eager_session
    
    from pyleaves.models.vgg16 import VGG16Grayscale
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.config import DatasetConfig, TrainConfig, ExperimentConfig
    from pyleaves.train.base_train import BaseTrainer, BaseTrainer_v1
    from pyleaves.analysis.mlflow_utils import mlflow_log_history, mlflow_log_best_history

    import mlflow
    import mlflow.tensorflow
    
    tracking_dir = r'/media/data/jacob/Fossil_Project/experiments/mlflow'
    ensure_dir_exists(tracking_dir)
    mlflow.set_tracking_uri(tracking_dir)
    print(mlflow.tracking.get_tracking_uri())
    
    mlflow.set_experiment('baselines')
    print(mlflow.get_artifact_uri())
    
    if args.num_channels==3:
        color_type = 'rgb'
    else:
        color_type = 'grayscale'
    
    ############################
    
    if args.model_name == 'all':
        model_names = ['vgg16', 'xception', 'resnet_50_v2','resnet_101_v2', 'shallow']
    else:
        model_names=[args.model_name]
    
    if args.dataset_name == 'all':
        dataset_names = ['PNAS', 'Fossil', 'Leaves']
    else:
        dataset_names = [args.dataset_name]
        
    if args.base_learning_rate == 'all':
        learning_rates = [1e-3, 1e-4,1e-5][1:]
    else:
        learning_rates = [float(args.base_learning_rate)]
    
#     model_names=['resnet_50_v2']
#     dataset_names=['Leaves']
    
    for model_name in model_names:
        for dataset_name in dataset_names:
            for lr in learning_rates:
                
                args.base_learning_rate = lr
                args.dataset_name = dataset_name
                args.model_name = model_name
                print('model_name=',args.model_name)

                if args.model_name in ['vgg16', 'resnet_50_v2','resnet_101_v2']:
                    target_size=(224,224)
                elif args.model_name=='xception':
                    target_size=(299,299)
                else:
                    target_size=(224,224)

                histories = []
                
                with mlflow.start_run(run_name=f'{args.model_name}-{args.dataset_name}-{color_type}-lr_{args.base_learning_rate}_baseline', nested=True):
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    experiment_dir = os.path.join(r'/media/data/jacob/Fossil_Project','experiments',args.model_name,args.dataset_name,color_type,f'lr-{args.base_learning_rate}',current_time)

                    reset_eager_session()

                    dataset_config = DatasetConfig(dataset_name=args.dataset_name,
                                                   label_col='family',
                                                   target_size=target_size,
                                                   num_channels=1,
                                                   grayscale=True,
                                                   low_class_count_thresh=args.low_class_count_thresh,
                                                   data_splits={'val_size':0.2,'test_size':0.2},
                                                   tfrecord_root_dir=args.tfrecord_dir,
                                                   num_shards=10)

                    train_config = TrainConfig(model_name=args.model_name,
                                               batch_size=args.batch_size,
                                               frozen_layers=(0,-4),
                                               base_learning_rate=args.base_learning_rate,
                                               buffer_size=500,
                                               num_epochs=args.num_epochs,
                                               preprocessing=True, #'imagenet',
                                               augment_images=True,
                                               augmentations=['rotate','flip'],
                                               regularization={'l2':0.001},
                                               seed=5,
                                               verbose=True)

                    experiment_config = ExperimentConfig(dataset_config=dataset_config,
                                                         train_config=train_config)            

                    mlflow.tensorflow.autolog()
                    
                    mlflow.log_params(experiment_config)

                    print(f'BEGINNING: DATASET:{args.dataset_name}|MODEL:{args.model_name}|lr:{args.base_learning_rate}|num_channels:{args.num_channels}|Grayscale={experiment_config.grayscale}')
                    print('-'*30)

                    history = main(experiment_config, experiment_dir)

                    histories.append((dataset_name, model_name, history))
                    mlflow.log_params(args.__dict__)
                    mlflow_log_history(history)

            
            
            
            
            
            
            
            
            