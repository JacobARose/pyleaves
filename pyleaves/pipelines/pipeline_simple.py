'''

python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py dataset@dataset=Leaves_family_100 restore_last=False dataset.val_split=0.1 run_description="''" model.regularization.l2=4e-5 model.lr=2e-4 model.head_layers=[512,256] buffer_size=512 +tags=['reference','Leaves_family_50'] use_tfrecords=False


python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py dataset@dataset=PNAS_50 restore_last=False dataset.val_split=0.1 run_description="'First attempt at PNAS_50 after successfully improved baseline performance on PNAS_100 by increasing l2 reg from 4e-10 -> 4e-6 to reduce overfitting'" model.regularization.l2=4e-6 model.lr=1e-4 model.head_layers=[256,128] buffer_size=512 +tags=['reference','PNAS_50'] use_tfrecords=False



python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_simple.py misc.debug=True


'''


# from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, DictConfig

# with initialize(config_path="configs"):
#     config = compose(config_name="config", overrides=['dataset@dataset=PNAS','use_tfrecords=False'])
#     print(config.pretty())
import hydra
# from pyleaves.pipelines.pipeline_1 import *

from pyleaves.datasets import base_dataset
from paleoai_data.utils.kfold_cross_validation import DataFold
from typing import List, Union
import random
import numpy as np
from more_itertools import unzip
import shutil
import os
import neptune


def log_hydra_config(backup_dir: str=None):

    override_dir = os.path.join(os.getcwd(),'.hydra')
    
    config_files = ['config.yaml', 'overrides.yaml', 'hydra.yaml']

    for f in config_files:
        filepath = os.path.join(override_dir, f)
        if os.path.exists(filepath):
            neptune.log_artifact(filepath)

            if isinstance(backup_dir, str):
                shutil.copy(filepath, backup_dir)




@hydra.main(config_path='configs', config_name='simplified_config')
def main(config : DictConfig):

    import os
    from pyleaves.train.paleoai_train import build_model
    from pyleaves.utils import set_tf_config
    from pyleaves.utils.experiment_utils import initialize_experiment, print_config

    from pyleaves.utils.pipeline_utils import create_dataset, get_callbacks, build_model
    from paleoai_data.utils.kfold_cross_validation import DataFold, StructuredDataKFold
    from pprint import pprint


    gpu_num = set_tf_config(gpu_num=config.orchestration.gpu_num, num_gpus=config.orchestration.num_gpus)
    import tensorflow as tf
    from tensorflow.keras import backend as K
    K.clear_session()
    # preprocess_input(tf.zeros([4, 224, 224, 3]));


    config = initialize_experiment(config, restore_last=config.misc.restore_last, restore_tfrecords=True)

    if config.dataset.params.extract.fold_id is None:
        config.dataset.params.extract.fold_id = 0


    data_config = config.dataset.params
    extract_config = config.dataset.params.extract
    training_config = config.dataset.params.training
    model_config = config.model.params
    preprocess_config = config.model.params.preprocess_input



    fold_path = DataFold.query_fold_dir(extract_config.fold_dir, extract_config.fold_id)
    fold = DataFold.from_artifact_path(fold_path)


    # data, split_datasets, encoder = create_dataset(data_fold=fold,
    #                                                cfg=data_config)

    data, extracted_data, split_datasets, encoder = create_dataset(data_fold=fold,
                                                                   data_config=data_config,
                                                                   preprocess_config=preprocess_config,
                                                                   cache=True,
                                                                   seed=config.misc.seed)
 

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

    callbacks = get_callbacks(config, model_config, model, fold.fold_id, train_data=train_data, val_data=val_data, encoder=encoder)

    print('[BEGINNING TRAINING]')
    params={}#**OmegaConf.to_container(data_config),
    #         **OmegaConf.to_container(model_config),
    #         **{k:v for k,v in OmegaConf.to_container(config).items() if ('_dir' in k) and (type(v) != dict)}}

    # pprint(params)

    neptune_experiment_name = config.misc.experiment_name
    with neptune.create_experiment(name=neptune_experiment_name, params=params, upload_source_files=['*.py']):

        model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
        log_hydra_config(backup_dir=config.run_dirs.log_dir)

        if config.orchestration.debug:
            import pdb;pdb.set_trace()
            print_config(config)        

        try:
            history = model.fit(train_data,
                                epochs=model_config.num_epochs,
                                callbacks=callbacks,
                                validation_data=val_data,
                                validation_freq=1,
                                shuffle=True,
                                steps_per_epoch=model_config.steps_per_epoch,
                                validation_steps=model_config.validation_steps,
                                verbose=1)
        except Exception as e:
            model.save(model_config['saved_model_path'])
            print('[Caught Exception, saving model first.\nSaved trained model located at:', model_config['saved_model_path'])
            raise e

        model.save(model_config['saved_model_path'])



        print('history.history.keys() =',history.history.keys())

        steps = split_datasets['test'].num_samples//data_config['batch_size']

        test_results = evaluate(model, encoder, model_config, data_config, test_data=test_data, steps=steps, num_classes=encoder.num_classes, confusion_matrix=True)

        print('TEST RESULTS:')
        pprint(test_results)

        for k,v in test_results.items():
            neptune.log_metric(k, v)
        # predictions = model.predict(test_data, steps=split_datasets['test'].num_samples)
        
    print(['[FINISHED TRAINING AND TESTING]'])

    return test_results

def evaluate(model, encoder, model_config, data_config, test_data=None, steps: int=None, num_classes: int=None, confusion_matrix=True):

    print('Preparing for model evaluation')

    test_data = test_data
    num_classes = num_classes

    text_labels = encoder.classes
    steps = steps

    callbacks=[]
    if confusion_matrix:
        from pyleaves.utils.callback_utils import NeptuneVisualizationCallback
        callbacks.append(NeptuneVisualizationCallback(test_data, num_classes=num_classes, text_labels=text_labels, steps=steps))

    test_results = model.evaluate(test_data, callbacks=callbacks, steps=steps, verbose=1)

    print('Model evaluation complete.')
    print('Results:')
    for m, result in zip(model.metrics_names, test_results):
        print(f'{m}: {result}')

        neptune.log_metric(f'test_{m}', result)

    return {m:result for m,result in zip(model.metrics_names, test_results)}










if __name__=='__main__':
    main()