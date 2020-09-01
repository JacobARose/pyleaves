# @Author: Jacob A Rose
# @Date:   Mon, April 13th 2020, 10:52 pm
# @Email:  jacobrose@brown.edu
# @Filename: main_single_dataset_experiment.py

'''
DEPRECATED (9/1/2020) Jacob Rose

#CUDA_VISIBLE_DEVICES=4 | python /home/jacob/projects/pyleaves/pyleaves/mains/main_single_dataset_experiment.py --gpu 4 -m vgg16 --batch_size 64 --run_id 1100 --grayscale

'''






def main():
    from pprint import pprint
    import sys
    import os
    import numpy as np
    import random

    gpu = '0'
    if '--gpu' in sys.argv:
        gpu = str(sys.argv[sys.argv.index('--gpu')+1])
        print('--gpu ',gpu)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # print('os.environ["CUDA_VISIBLE_DEVICES"] : ',os.environ["CUDA_VISIBLE_DEVICES"])
    import tensorflow as tf
    from pyleaves.utils import create_session

    sess = create_session(gpus='0', timeout_sec=10)
    # tf.compat.v1.enable_eager_execution()
    print('tf.executing_eagerly()', tf.executing_eagerly())

    from stuf import stuf
    from pyleaves.configs.config_v2 import BaseConfig
    import json

    try:
        config = BaseConfig().parse(args=sys.argv[1:])
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        print('sys.argv = ', sys.argv[1:])
        exit(0)

    from pyleaves.leavesdb.tf_utils.tf_utils import set_random_seed
    set_random_seed(config.seed)

    from pyleaves.base.base_data_manager import DataManager
    from pyleaves.base.base_trainer import ModelBuilder, BaseTrainer
    from pyleaves.train.callbacks import get_callbacks
    from pyleaves.loggers.mlflow_logger import MLFlowLogger as Logger

    # import pdb; pdb.set_trace()

    data_manager = DataManager(config=config)
    # train_data = data_manager.get_data_loader(file_group='train', dataset_stage='dataset_A+dataset_B')
    val_data = data_manager.get_data_loader(file_group='val', dataset_stage='dataset_A+dataset_B')
    # test_data = data_manager.get_data_loader(file_group='test', dataset_stage='dataset_A+dataset_B')
    # import pdb; pdb.set_trace()

    model_builder = ModelBuilder(config)
    callbacks = get_callbacks(weights_best=os.path.join(config.model_config.model_dir,'weights_best.h5'),
                                  logs_dir=os.path.join(config.model_config.log_dir,'tensorboard_logs'),
                                  val_data=None,#val_data,
                                  batches_per_epoch=0,
                                  freq=0,
                                  histogram_freq=0,
                                  restore_best_weights=True,
                                  seed=config.seed,
                                  sess=sess)


    logger = Logger(config)

    trainer = BaseTrainer(config, model_builder, data_manager, logger, callbacks, sess)

    pprint(config)
    print('INITIATING TRAINING')


    trainer.train(dataset_stage='dataset_A+dataset_B')

    trainer.save_model(config.run_id+'_model')


    trainer.test(dataset_stage='dataset_A+dataset_B')

    # # create the experiments dirs
    # # create_dirs([config.summary_dir, config.checkpoint_dir])
    # # create tensorflow session
    # # sess = tf.Session()
    # # create your data generator
    # data_manager = DataManager(config=config)
    # train_data = data_manager.get_data_loader(file_group='train')
    # val_data = data_manager.get_data_loader(file_group='val')
    # test_data = data_manager.get_data_loader(file_group='test')
    #
    # # create an instance of the model you want
    # model_builder = ModelBuilder(config)
    # model = model_builder.build_model()
    # #model_factory is dynamically defined based on the type of model used, brings with it useful model save/load methods.
    # model_factory = model_builder.model_factory
    #
    # callbacks = get_callbacks(weights_best=os.path.join(config.model_config.model_dir,'target_domain_weights_best.h5'),
    #                               logs_dir=os.path.join(config.model_config.log_dir,'tensorboard_logs'),
    #                               restore_best_weights=True)
    # # create tensorboard logger
    # # logger = Logger(sess, config)
    # # create trainer and pass all the previous components to it
    # trainer = BaseTrainer(config, model, data_manager, callbacks, model_builder)
    # #load model if exists
    # # here you train your model
    # trainer.train()


if __name__ == '__main__':
    main()
