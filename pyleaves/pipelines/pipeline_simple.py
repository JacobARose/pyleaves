'''

python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py dataset@dataset=Leaves_family_100 restore_last=False dataset.val_split=0.1 run_description="''" model.regularization.l2=4e-5 model.lr=2e-4 model.head_layers=[512,256] buffer_size=512 +tags=['reference','Leaves_family_50'] use_tfrecords=False


python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py dataset@dataset=PNAS_family_100 restore_last=False dataset.val_split=0.1 run_description="'First attempt at PNAS_50 after successfully improved baseline performance on PNAS_100 by increasing l2 reg from 4e-10 -> 4e-6 to reduce overfitting'" model.regularization.l2=4e-6 model.lr=1e-4 model.head_layers=[256,128] buffer_size=512 +tags=['reference','PNAS_50'] use_tfrecords=False





python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py dataset@dataset=PNAS_family_100 model@model=resnet_50_v2 restore_last=False dataset.params.extract.val_split=0.1 misc.run_description="" model.params.regularization.l2=4e-3 model.params.lr=1e-4 model.params.head_layers=[512,256] dataset.params.training.buffer_size=128 tags=["reference","PNAS_family_100"] dataset.params.training.batch_size=16 dataset.params.training.num_epochs=30, 'model.params.frozen_layers=null,(0,-4)' -m


python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py misc.debug=True








# TODO

1. PNAS train/val/test -> evaluate on Fossil
2. Leaves train/val -> evaluate on Fossil
3. Leaves_in_PNAS train/val/test vs. PNAS_in_Leaves train/val/test
        -For the above, should I shuffle catalog_numbers and assign the same pairs to train, val, or test, respectively? e.g. catalog_number=Wolfe_2837, then if the image w/ that catalog_number in Leaves gets assigned to the Leaves validation set, its match in PNAS will be in the corresponding PNAS validation set
        -one-to-one sample-level correspondence
    vs
        -shuffling them separately, so many samples will exist in more than one subset, spanning different datasets

python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py dataset@dataset=Fossil_family_100 






freeze resnet50_v2 sweep [0,-x]

-x layer[-x].name

-4 conv5_block3_3_conv
-7 conv5_block3_2_conv
-11 conv5_block3_1_conv
-15 conv5_block2_3_conv
-18 conv5_block2_2_conv
-22 conv5_block2_1_conv
-26 conv5_block1_3_conv
-27 conv5_block1_0_conv
-30 conv5_block1_2_conv
-34 conv5_block1_1_conv
-38 conv4_block6_3_conv
-42 conv4_block6_2_conv
-46 conv4_block6_1_conv
-50 conv4_block5_3_conv
-53 conv4_block5_2_conv
-57 conv4_block5_1_conv




'''


# from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, DictConfig, ListConfig

# with initialize(config_path="configs"):
#     config = compose(config_name="config", overrides=['dataset@dataset=PNAS','use_tfrecords=False'])
#     print(config.pretty())
import hydra
# from pyleaves.pipelines.pipeline_1 import *

from pyleaves.datasets import base_dataset

from pyleaves.utils.experiment_utils import resolve_config_interpolations
from paleoai_data.utils.kfold_cross_validation import DataFold
from typing import List, Union
import random
import numpy as np
from more_itertools import unzip
from pprint import pprint
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")




import neptune
from pathlib import Path
import yaml

def log_hydra_config(backup_dir: str=None, config: DictConfig=None, experiment=None):
    experiment = experiment or neptune
    if config is not None:
        for k,v in config.dataset.params.extract.items():
            experiment.set_property('dataextract_'+k,v)
        for k,v in config.dataset.params.training.items():
            experiment.set_property('datatrain_'+k,v)
        for k,v in config.model.params.items():
            experiment.set_property('model_'+k,v)
        for k,v in config.run_dirs.items():
            experiment.set_property('rundirs_'+k,v)
        # config_output_path = os.path.join(config.run_dirs.log_dir,'config.yaml')
        # with open(config_output_path, 'w') as f:
        #     yaml.dump(resolve_config_interpolations(config=config, log_nodes=False), f)
        # experiment.log_artifact(config_output_path)
        # print(f'Logged resolved config to {config_output_path}')

    if type(config.tags)==list:
        for tag in config.tags:
            experiment.append_tag(tag)

    override_dir = os.path.join(os.getcwd(),'.hydra')
    config_files = ['config.yaml', 'overrides.yaml', 'hydra.yaml']
    for f in config_files:
        filepath = os.path.join(override_dir, f)
        if os.path.exists(filepath):
            experiment.log_artifact(filepath)

            if isinstance(backup_dir, str):
                shutil.copy(filepath, backup_dir)




def validate_model_config(config):
    """
     TODO  Add logging

    Args:
        config ([type]): [description]

    Returns:
        [type]: [description]
    """    
    assert 'params' in config.model
    model_config = config.model.params

    model_config.model_name = str(model_config.model_name)
    assert model_config.weights in ['imagenet', None] 
    assert len(model_config.input_shape) == 3 and \
            model_config.input_shape[0]==model_config.input_shape[1] and \
            model_config.input_shape[2] in [1,3]
    assert isinstance(model_config.num_classes, (int, type(None)))
    assert model_config.num_channels==model_config.input_shape[2]
    assert model_config.optimizer in ['Adam', 'SGD', 'RMSprop']
    assert model_config.loss in ['categorical_crossentropy']
    model_config.lr = float(model_config.lr)
    model_config.lr_decay = float(model_config.lr_decay)
    model_config.lr_decay_epochs = int(model_config.lr_decay_epochs)
    model_config.lr_momentum = float(model_config.lr_momentum)
    
    for regularizer_L in model_config.regularization.keys():
        if model_config.regularization[regularizer_L] is not None:
            model_config.regularization[regularizer_L] = float(model_config.regularization[regularizer_L])

    if 'l1' in model_config.regularization:
        if model_config.regularization.l1 is not None:
            model_config.regularization.l2 = None

    print('model_config.frozen_layers: ', model_config.frozen_layers)
    print('type(model_config.frozen_layers): ', type(model_config.frozen_layers))
    if model_config.frozen_layers is not None:
        if type(model_config.frozen_layers) not in [list,tuple,ListConfig]:
            model_config.frozen_layers = None
        elif len(model_config.frozen_layers)==0:
            model_config.frozen_layers = None
        else:
            for i, l in enumerate(model_config.frozen_layers):
                model_config.frozen_layers[i] = int(l)
            model_config.frozen_layers = list(model_config.frozen_layers)
    print('model_config.frozen_layers: ', model_config.frozen_layers)
    print('type(model_config.frozen_layers): ', type(model_config.frozen_layers))

    config.model.params = model_config

    return config














@hydra.main(config_path='configs', config_name='simplified_config')
def main(config : DictConfig):


    OmegaConf.set_struct(config, False)
    from hydra.core.hydra_config import HydraConfig
    # from pyleaves.train.paleoai_train import build_model
    from pyleaves.utils import set_tf_config
    from pyleaves.utils.experiment_utils import initialize_experiment, print_config
    from pyleaves.utils.pipeline_utils import create_dataset, get_callbacks, build_model
    from paleoai_data.utils.kfold_cross_validation import DataFold, StructuredDataKFold
    # config.orchestration.gpu_num = 
    # print('BEFORE PDB')
    # import pdb
    # pdb.set_trace()
    # print('AFTER PDB')
    print('hydra.job.num=task=',config.task)

    # TODO spawn 8 lock files for the GPUs
    # job_num = int(HydraConfig.get().job.num)
    job_num = config.task #os.getpid()%8
    # print(f'job_num = int(HydraConfig.get().job.num) = {job_num}')
    try:
        print(f'Waiting job_num*config.orchestration.wait = {job_num*config.orchestration.wait}')
        gpu = set_tf_config(gpu_num=config.orchestration.gpu_num, num_gpus=config.orchestration.num_gpus, wait=job_num*config.orchestration.wait)

        print(f'Job number {job_num} assigned to GPU {gpu}', dir(gpu[0]))
    except:
        print('Failed to set tf_gpu config with hydra.job.id. Continuing anyway.')
    import tensorflow as tf
    from tensorflow.keras import backend as K
    K.clear_session()

    config = initialize_experiment(config, restore_last=config.misc.restore_last, restore_tfrecords=True)
    if config.dataset.params.extract.fold_id is None:
        config.dataset.params.extract.fold_id = 0

    # config.dataset.params.extract.fold_dir = f"/media/data/jacob/Fossil_Project/data/csv_data/paleoai_data_disk_cache_dir/staged_data/{config.dataset.params.extract.dataset_name}/ksplit_2/" 


    config = validate_model_config(config) #ensure learning rate is passed as float, as well as some more checks

    data_config = config.dataset.params
    extract_config = config.dataset.params.extract
    training_config = config.dataset.params.training
    model_config = config.model.params
    preprocess_config = config.model.params.preprocess_input

    fold_path = DataFold.query_fold_dir(extract_config.fold_dir, extract_config.fold_id)
    fold = DataFold.from_artifact_path(fold_path)
    data, extracted_data, split_datasets, encoder = create_dataset(data_fold=fold,
                                                                   data_config=data_config,
                                                                   preprocess_config=preprocess_config,
                                                                   cache=True,
                                                                   cache_image_dir=config.run_dirs.cache_dir,
                                                                   seed=config.misc.seed)
    # TODO hash and log extracted_data
    if data_config.training.steps_per_epoch is None:
        data_config.training.steps_per_epoch = split_datasets['train'].num_samples//data_config.training.batch_size

    if (data_config.training.validation_steps is None) and ('val' in split_datasets):
        data_config.training.validation_steps = split_datasets['val'].num_samples//data_config.training.batch_size

    train_data, val_data, test_data = data['train'], data['val'], data['test']
    data_config.extract.num_classes=encoder.num_classes
    model_config.input_shape = (*training_config.target_size, extract_config.num_channels)
    model_config.num_classes = encoder.num_classes
    model = build_model(model_config)

    config.dataset.params = data_config
    config.model.params = model_config


    neptune.init(project_qualified_name=config.misc.neptune_project_name)
    params=resolve_config_interpolations(config=config, log_nodes=False)

    # neptune_experiment_name = config.misc.experiment_name
    with neptune.create_experiment(name=config.misc.experiment_name, params=params, upload_source_files=[os.path.join(hydra.utils.get_original_cwd(),'*.py')]) as experiment:
        model.summary(print_fn=lambda x: experiment.log_text('model_summary', x))
        log_hydra_config(backup_dir=config.run_dirs.log_dir, config=config, experiment=experiment)

        csv_path = str(Path(config.run_dirs.results_dir,f'results-fold_{extract_config.fold_id}.csv'))
        callbacks = get_callbacks(config, model_config, model, csv_path, train_data=train_data, val_data=val_data, encoder=encoder, experiment=experiment)

        print('[BEGINNING TRAINING]')
        if config.orchestration.debug:
            import pdb;pdb.set_trace()
            print_config(config)
        try:
            history = model.fit(train_data,
                                epochs=data_config.training.num_epochs,
                                callbacks=callbacks,
                                validation_data=val_data,
                                validation_freq=1,
                                shuffle=True,
                                steps_per_epoch=data_config.training.steps_per_epoch,
                                validation_steps=data_config.training.validation_steps,
                                verbose=1)

            if config.orchestration.debug:
                import pdb;pdb.set_trace()
                print_config(config)
        except Exception as e:
            model.save(config.run_dirs.saved_model_path)
            print('[Caught Exception, saving model first.\nSaved trained model located at:', config.run_dirs.saved_model_path)
            if config.orchestration.debug:
                import pdb;pdb.set_trace()
                print_config(config)
            raise e



        if os.path.exists(csv_path):
            experiment.log_artifact(csv_path)

        model.save(config.run_dirs.saved_model_path)
        print('[STAGE COMPLETED]')
        print(f'Saved trained model to {config.run_dirs.saved_model_path}')

        print('history.history.keys() =',history.history.keys())

        steps = split_datasets['test'].num_samples//data_config.training.batch_size

        test_results = evaluate(model, encoder, model_config, data_config, test_data=test_data, steps=steps, num_classes=encoder.num_classes, confusion_matrix=True, experiment=experiment)

        print('TEST RESULTS:')
        pprint(test_results)

        print(['[FINISHED TRAINING AND TESTING]'])
        # for k,v in test_results.items():
        #     neptune.log_metric(k, v)
        # predictions = model.predict(test_data, steps=split_datasets['test'].num_samples)

        # TODO walk throug below section and test
        print(f'INITIATING ZERO-SHOT TEST ON Fossil_family_100')
        test_dataset_config = OmegaConf.load('configs/dataset/Fossil_family_100_test.yaml')
        test_dataset_config = OmegaConf.merge(data_config, test_dataset_config.params)
        test_dataset_config.extract.num_classes = encoder.num_classes

        # fold_dir = f"/media/data/jacob/Fossil_Project/data/csv_data/paleoai_data_disk_cache_dir/staged_data/{test_dataset_config.dataset_name}/ksplit_2/" 
        fold_dir = test_dataset_config.extract.fold_dir
        fold_id = test_dataset_config.extract.fold_id

        fold_path = DataFold.query_fold_dir(fold_dir, fold_id)
        fold = DataFold.from_artifact_path(fold_path)
        data, extracted_data, split_datasets, encoder = create_dataset(data_fold=fold,
                                                                    data_config=test_dataset_config,
                                                                    preprocess_config=preprocess_config,
                                                                    class_names=encoder.classes,
                                                                    cache=True,
                                                                    cache_image_dir=config.run_dirs.cache_dir,
                                                                    seed=config.misc.seed)

        steps = split_datasets['test'].num_samples//data_config.training.batch_size

        test_results = evaluate(model, encoder, model_config, test_dataset_config, test_data=data['test'], steps=steps, num_classes=encoder.num_classes, confusion_matrix=True, experiment=experiment, subset_prefix='Fossil_family_100_test')







    # print(['[FINISHED TRAINING AND TESTING]'])

    return test_results

def evaluate(model, encoder, model_config, data_config, test_data=None, steps: int=None, num_classes: int=None, confusion_matrix=True, experiment=None, subset_prefix='test'):
    from pyleaves.utils.pipeline_utils import evaluate_performance
    experiment = experiment or neptune
    print('Preparing for model evaluation')
    # test_data = test_data
    # num_classes = num_classes

    text_labels = encoder.classes
    steps = steps

    callbacks=[]
    if confusion_matrix:
        from pyleaves.utils.callback_utils import NeptuneVisualizationCallback
        callbacks.append(NeptuneVisualizationCallback(test_data, num_classes=num_classes, text_labels=text_labels, steps=steps, subset_prefix=subset_prefix, experiment=experiment))

    if data_config.testing.eval_performance_w_sklearn:
        report = evaluate_performance(model, x=test_data, text_labels=text_labels)
        experiment.log_text(f'{subset_prefix}_classification_report', report)





    test_results = model.evaluate(test_data, callbacks=callbacks, steps=steps, verbose=1)

    print('Model evaluation complete.')
    print('Results:')
    for m, result in zip(model.metrics_names, test_results):
        print(f'{m}: {result}')
        experiment.log_metric(f'{subset_prefix}_{m}', result)

    return {m:result for m,result in zip(model.metrics_names, test_results)}












if __name__=='__main__':
    main()