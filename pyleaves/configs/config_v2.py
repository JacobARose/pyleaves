# @Author: Jacob A Rose
# @Date:   Tue, April 14th 2020, 2:51 am
# @Email:  jacobrose@brown.edu
# @Filename: config_v2.py

import argparse
import datetime
import json
import numpy as np
import funcy
import os
from pyleaves.leavesdb import experiments_db
from pyleaves.leavesdb.experiments_db import get_db_table, select_by_col, select_by_multicol
from pyleaves import EXPERIMENTS_DB
from pyleaves.utils import ensure_dir_exists
# from pyleaves.utils.csv_utils import gather_experiment_data
from toolz import diff
from stuf import stuf

EXPERIMENT_TYPES = ['A_train_val_test',
                    'A+B_train_val_test',
                    'A_train_val-B_train_val_test',
                    'A+B_leave_one_out']

# TODO PRIORITY Add simple test dataset from tf.datasets

DATASET_SOURCES = ['PNAS',
                   'Leaves',
                   'Fossil']

MODEL_CATALOG = ['vgg16', 'resnet_50_v2']

class BaseConfig:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--experiment_type', type=str, default='A_train_val_test',choices=EXPERIMENT_TYPES,
                                 help='String describing the experimental procedure in shorthand. Refer to pyleaves.configs.README for details.')
        self.parser.add_argument('--dataset_A', type=str, default='PNAS',choices=DATASET_SOURCES,
                                 help='Name of the source dataset to be used as dataset A')
        self.parser.add_argument('--dataset_B', type=str, default=None,choices=DATASET_SOURCES,
                                 help='Name of the source dataset to be used as dataset B. Only used if experiment_type references both A and B.')
        self.parser.add_argument('-m','--model', type=str, default='vgg16',choices=MODEL_CATALOG, help='Model name')
        self.parser.add_argument('-id','--run_id', dest='run_id', type=str, default=None,
                                 help='Optional unique run_id. If provided, overrides any other cmd line arg.')
        self.parser.add_argument('-bsz','--batch_size', type=int, default=32)
        self.parser.add_argument('-lr','--learning_rate', type=float, default=3e-4)
        self.parser.add_argument('-epochs','--num_epochs', dest='num_epochs', type=int, default=100)
        self.parser.add_argument('--gpu', default='0', type=str)
        self.parser.add_argument('--seed', type=int, default=10293)
        self.parser.add_argument('--rgb', dest='grayscale', default=True, action='store_false')
        self.parser.add_argument('--no-augment', dest='augment_images', default=True, action='store_false')
        self.parser.add_argument('-f', default=None)

    def parse(self, args=None, namespace=None):
        # import pdb;pdb.set_trace()
        cfg = self.parser.parse_args(args, namespace)
        if len(cfg.__dict__)>0: cfg = cfg.__dict__
        cfg = stuf(dict(cfg))
        cfg['user_config'] = self._setup_user_config()
        cfg['data_config'] = self._setup_data_config(cfg)
        cfg.update(self._setup_model_config(cfg))
        cfg.update(self._setup_logger_config(cfg))
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
        # print('os.environ["CUDA_VISIBLE_DEVICES"] : ',os.environ["CUDA_VISIBLE_DEVICES"])
        return cfg

    def _setup_user_config(self):
        return stuf({
                    'experiments_db':EXPERIMENTS_DB,
                    'user_csv_dir':r'/media/data_cifs/jacob/Fossil_Project/data/csv_data',
                    'user_tfrecord_dir':r'/media/data/jacob/Fossil_Project/data/tfrecord_data',
                    'user_model_dir':r'/media/data_cifs/jacob/Fossil_Project/models',
                    'experiment_start_time':datetime.datetime.now().strftime('%a-%m-%d-%Y_%H-%M-%S')
                    })
                    # 'user_csv_dir':r'/media/data/jacob/Fossil_Project/data/csv_data',

    def _setup_data_config(self, cfg):
        csv_root_dir = cfg.user_config.user_csv_dir
        tfrecord_root_dir = cfg.user_config.user_tfrecord_dir
        runs_table = experiments_db.get_db_table('runs',cfg.user_config.experiments_db)

        if cfg.run_id:
            #run_id overrides other run params
            run = select_by_col(runs_table,'run_id',cfg.run_id)
            cfg.experiment_type = run.experiment_type.iloc[0]
            cfg.dataset_A = run.dataset_A.iloc[0]
            cfg.dataset_B = run.dataset_B.iloc[0]
            # cfg.datasets = [d for d in [cfg.dataset_A, cfg.dataset_B] if d]
        elif cfg.experiment_type == 'A+B_train_val_test':

            run_id = select_by_multicol(runs_table, kwargs={'experiment_type':cfg.experiment_type,
                                                                'dataset_A':cfg.dataset_A,
                                                                'dataset_B':cfg.dataset_B})['run_id'].tolist()
            if len(run_id)==0:
                run_id = select_by_multicol(runs_table, kwargs={'experiment_type':cfg.experiment_type,
                                                                    'dataset_A':cfg.dataset_B,
                                                                    'dataset_B':cfg.dataset_A})['run_id'].tolist()
                if len(run_id)==0:
                    print('invalid dataset_A or dataset_B, experiment config info not found')
                    raise
                else:
                    print('Reversing order of user provided datasets from A,B to B,A. This should make no difference based on experiment type.')
                    cfg.dataset_A, cfg.dataset_B = cfg.dataset_B, cfg.dataset_A

            cfg.run_id = run_id[0]


        cfg.datasets = [d for d in [cfg.dataset_A, cfg.dataset_B] if d]
        experiment = cfg.experiment_type
        dataset_A = cfg.dataset_A
        dataset_B = cfg.dataset_B
        records_table = experiments_db.get_db_table('tfrecords',cfg.user_config.experiments_db)

        data_cfg = stuf({'batch_size':cfg.batch_size})
        if experiment == 'A_train_val_test':
            cfg.stages = ['dataset_A']
            csv_dir = os.path.join(csv_root_dir,experiment,dataset_A)
            data_cfg['name'] = dataset_A
            data_cfg['tfrecord_dir'] = os.path.join(tfrecord_root_dir,experiment,dataset_A)
            data_cfg['csv_dir'] = csv_dir
            data_cfg['label_dir'] = csv_dir
            data_cfg['threshold'] = 10
            # data_cfg['num_classes'] = int(num_classes)
            data_cfg['num_shards'] = 10
            data_cfg['resolution'] = 256
            data_cfg['file_groups'] = ['train','val','test']
            data_cfg['data_splits_meta'] = {
                                'train':0.4,
                                'val':0.1,
                                'test':0.5
                               }
            data_cfg['csv_data'] = {
                            'train':os.path.join(csv_dir,'train_data.csv'),
                            'val':os.path.join(csv_dir,'val_data.csv'),
                            'test':os.path.join(csv_dir,'test_data.csv')
                            }
            data_config = stuf({})
            data_config.dataset_A = data_cfg
            data_config.class_weights_filepath = os.path.join(csv_dir,'train_class_weights.csv')
            return data_config

        elif experiment == 'A+B_train_val_test':
            cfg.stages = ['dataset_A+dataset_B']
            data_cfg['name'] = '+'.join([dataset_A,dataset_B])
            csv_dir = os.path.join(csv_root_dir,experiment,data_cfg['name'])
            data_cfg['tfrecord_dir'] = os.path.join(tfrecord_root_dir,experiment,data_cfg['name'])
            data_cfg['csv_dir'] = csv_dir
            data_cfg['label_dir'] = csv_dir
            data_cfg['threshold'] = 10
            # data_cfg['num_classes'] = int(num_classes)
            data_cfg['num_shards'] = 10
            data_cfg['resolution'] = 256
            data_cfg['file_groups'] = ['train','val','test']
            data_cfg['data_splits_meta'] = {
                                'train':0.4,
                                'val':0.1,
                                'test':0.5
                               }
            data_cfg['csv_data'] = {
                            'train':os.path.join(csv_dir,'train_data.csv'),
                            'val':os.path.join(csv_dir,'val_data.csv'),
                            'test':os.path.join(csv_dir,'test_data.csv')
                            }
            data_config = stuf({})
            # data_config.dataset_A = data_cfg
            # data_config.dataset_B = data_cfg
            data_config.dataset_AB = data_cfg #TODO DEPRECATE THIS {AB} SYNTAX IN FAVOR OF {A+B}
            data_config['dataset_A+dataset_B'] = data_cfg
            data_config.class_weights_filepath = os.path.join(csv_dir,'train_class_weights.csv')
            return data_config




        # if dataset_name == 'flowers':
        #     num_classes = 102
        #     db_path = datasets_dir + 'flower102'
        #     db_tuple_loader = 'data_sampling.flower_tuple_loader.FLower102TupleLower'
        #     train_csv_file = '/lists/train_all_sub_list.csv'
        #     val_csv_file = '/lists/val_all_sub_list.csv'
        #     test_csv_file = '/lists/test_all_sub_list.csv'
        # elif dataset_name == 'cars':
        #     num_classes = 196
        #     db_path = datasets_dir + 'stanford_cars'
        #     db_tuple_loader = 'data_sampling.cars_tuple_loader.CarsTupleLoader'
        #     train_csv_file = '/lists/train_all_sub_list.csv'
        #     val_csv_file = '/lists/val_all_sub_list.csv'
        #     test_csv_file = '/lists/test_all_sub_list.csv'
        # elif dataset_name == 'aircrafts':
        #     num_classes = 100
        #     db_path = datasets_dir + 'aircrafts'
        #     db_tuple_loader = 'data_sampling.aircrafts_tuple_loader.AircraftsTupleLoader'
        #     train_csv_file = '/lists/train_all_sub_list.csv'
        #     val_csv_file = '/lists/val_all_sub_list.csv'
        #     test_csv_file = '/lists/test_all_sub_list.csv'
        # elif dataset_name == 'dogs':
        #     num_classes = 120
        #     db_path = datasets_dir + 'Stanford_dogs'
        #     db_tuple_loader = 'data_sampling.dogs_tuple_loader.DogsTupleLoader'
        #     train_csv_file = '/lists/train_all_sub_list.csv'
        #     val_csv_file = '/lists/val_sub_list.csv'
        #     test_csv_file = '/lists/test_all_sub_list.csv'
        # elif dataset_name == 'birds':
        #     num_classes = 555
        #     db_path = datasets_dir + 'nabirds'
        #     db_tuple_loader = 'data_sampling.birds_tuple_loader.BirdsTupleLoader'
        #     train_csv_file = '/lists/train_all_sub_list.csv'
        #     val_csv_file = '/lists/val_sub_list.csv'
        #     test_csv_file = '/lists/test_all_sub_list.csv'
        else:
            raise NotImplementedError('dataset_name not found')

        # return num_classes,db_path,db_tuple_loader,train_csv_file,val_csv_file,test_csv_file


    def _setup_model_config(self,cfg):
        config = stuf({'name':cfg.model})
        config.num_epochs = cfg.num_epochs
        config.batch_size = cfg.batch_size
        # config.num_classes = cfg.data_config.dataset_A.num_classes
        config.frozen_layers=None
        config.base_learning_rate = cfg.learning_rate
        config.regularization = {'l2':0.01}
        config.model_dir = os.path.join(cfg.user_config.user_model_dir,'_'.join(cfg.datasets))
        config.log_dir = os.path.join(config.model_dir,'logs', cfg.user_config.experiment_start_time)

        if cfg.model == 'resnet_50_v2':
            # from tensorflow.keras.applications.resnet_v2 import preprocess_input
            config.preprocessing_module = 'tensorflow.keras.applications.resnet_v2'
            config.input_shape = (224,224,1)

        elif cfg.model == 'vgg16':
            # from tensorflow.keras.applications.vgg16 import preprocess_input
            config.preprocessing_module = 'tensorflow.keras.applications.vgg16'
            config.input_shape = (224,224,1)

        # config.preprocess_func = lambda x,y: (preprocess_input(x), y)
        return stuf({'model_config':config})

    def _setup_logger_config(self,cfg):
        config = stuf({'mlflow_tracking_dir':r'/media/data/jacob/Fossil_Project/experiments/mlflow',
                        'program_log_file':os.path.join(cfg.model_config.log_dir,'mlflow_log.txt')})
        return {'logger':config}

        # elif model == 'resnet50_v1':
        #     network_name = 'nets.resnet_v1.ResNet50'
        #     imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v1_50/resnet_v1_50.ckpt'
        #     preprocess_func = 'vgg'
        #     preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        # elif model == 'densenet161':
        #     network_name = 'nets.densenet161.DenseNet161'
        #     imagenet__weights_filepath = pretrained_weights_dir + 'tf-densenet161/tf-densenet161.ckpt'
        #     preprocess_func = 'densenet'
        #     preprocessing_module = 'data_sampling.augmentation.densenet_preprocessing'
        # elif model == 'inc4':
        #     network_name = 'nets.inception_v4.InceptionV4'
        #     imagenet__weights_filepath = pretrained_weights_dir + 'inception_v4/inception_v4.ckpt'
        #     preprocess_func = 'inception_v1'
        #     preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        # elif model == 'inc3':
        #     network_name = 'nets.inception_v3.InceptionV3'
        #     imagenet__weights_filepath = pretrained_weights_dir + 'inception_v3.ckpt'
        #     preprocess_func = 'inception_v1'
        #     preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        # elif model == 'mobile':
        #     network_name = 'nets.mobilenet_v1.MobileV1'
        #     imagenet__weights_filepath = pretrained_weights_dir + 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
        #     preprocess_func = 'inception_v1'
        #     preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        # else:
        #     raise NotImplementedError('network name not found')
        #
        # return network_name,imagenet__weights_filepath,preprocess_func,preprocessing_module
