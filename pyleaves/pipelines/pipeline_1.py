# @Author: Jacob A Rose
# @Date:   Sun September 6th 2020, 10:42 pm
# @Email:  jacobrose@brown.edu
# @Filename: pipeline_1.py

"""
This is meant to be a modularization of common model training scripts, allowing for further customization of the order and content of each piepline node/step/stage.



python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_1.py dataset@dataset='PNAS'

python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_1.py dataset@dataset=PNAS restore_last=False dataset.val_split=0.1 run_description="''" model.regularization.l2=4e-10 model.lr=1e-4 model.head_layers=[256,128] buffer_size=512


python /home/jacob/projects/pyleaves/pyleaves/pipelines/pipeline_1.py dataset@dataset=PNAS restore_last=False dataset.val_split=0.1 run_description="'increased model head size, shuffle buffer siz: 1024->512'" model.regularization.l2=4e-10 model.lr=1e-4 model.head_layers=[256,128] buffer_size=512
"""

from functools import partial
import hydra
import json
# from omegaconf import OmegaConf
import numpy as np
import os
from pathlib import Path
from prefect import Flow, task
from paleoai_data.utils.kfold_cross_validation import DataFold, KFoldLoader
from paleoai_data.dataset_drivers.base_dataset import BaseDataset
from pyleaves.models import resnet, vgg16
from pyleaves.base import base_model
from pyleaves.utils.neptune_utils import neptune
from pyleaves.train.paleoai_train import preprocess_input


from datetime import datetime
from pyleaves.utils import ensure_dir_exists
import os
import shutil

# @task
# def extract():
#     return [1, 2, 3]


# @task
# def transform(x):
#     return [i * 10 for i in x]


# @task
# def load(y):
#     print("Received y: {}".format(y))


# with Flow("ETL") as flow:
#     e = extract()
#     t = transform(e)
#     l = load(t)

# flow.run()



# 1. Create dataset

# 2a. Initialize model

# 2b. Load model

# 3. model.fit

# 4. model.predict

# 5. model.evaluate

from typing import List, Tuple, Dict
from omegaconf import DictConfig, OmegaConf
from pyleaves.utils import ensure_dir_exists
from pyleaves.mains.paleoai_main import to_serializable

def create_dataset_config(dataset_name: str='Leaves-PNAS',
                          exclude_classes: List[str]=['notcataloged','notcatalogued', 'II. IDs, families uncertain', 'Unidentified'],
                          include_classes: List[str]=[],
                          target_size: Tuple[int,int]=(512,512),
                          num_channels: int=3,
                          color_mode: str='grayscale',
                          val_split: float=0.0,
                          fold_dir= '/home/jacob/projects/paleoai_data/paleoai_data/v0_2/data/staged_data/Leaves-PNAS/ksplit_10',
                          batch_size: int=1,
                          buffer_size: int=200,
                          augmentations: Dict[str,float]={'flip':0.0},
                          seed: int=None,
                          use_tfrecords: bool=True,
                          tfrecord_dir: str=None,
                          samples_per_shard: int=300,
                          debug=False,
                          debugging: Dict[str,bool]=None,
                          **kwargs):
    
    if tfrecord_dir:
        ensure_dir_exists(tfrecord_dir)

    if debugging is None:
        debugging = {'overfit_one_batch':False}

    if type(debugging)==DictConfig:
        debugging = OmegaConf.to_container(debugging)


    params = OmegaConf.create(
                dict(dataset_name=dataset_name,
                     exclude_classes=exclude_classes,
                     include_classes=include_classes,
                     target_size=target_size,
                     num_channels=num_channels,
                     color_mode=color_mode,
                     val_split=val_split,
                     fold_dir=fold_dir,
                     batch_size=batch_size,
                     buffer_size=buffer_size,
                     augmentations=augmentations,
                     seed=seed,
                     use_tfrecords=use_tfrecords,
                     tfrecord_dir=tfrecord_dir,
                     samples_per_shard=samples_per_shard,
                     debug=debug,
                     debugging=debugging)
            )
    return params




    # params = locals()
    # params.pop('kwargs') #ignore any extra parameters in kwargs
    # return OmegaConf.create(params)


def create_model_config(model_name: str='resnet_50_v2',
                        optimizer: str='Adam',
                        loss: str='categorical_crossentropy',
                        weights: str='imagenet',
                        lr: float=4.0e-5,
                        lr_decay: float=None,
                        lr_momentum: float=None,
                        lr_decay_epochs: int=10,
                        regularization: Dict[str,float]={'l1': None},
                        METRICS: List[str]=['f1','accuracy'],
                        head_layers: List[int]=[512,256],
                        batch_size: int=16,
                        buffer_size: int=200,
                        num_epochs: int=150,
                        frozen_layers: List[int]=None,
                        augmentations: List[Dict[str,float]]=[{'flip': 1.0}],
                        num_classes: int=None,
                        target_size: Tuple[int,int]=(512,512),
                        num_channels: int=3,
                        steps_per_epoch: int=None,
                        validation_steps: int=None,
                        model_dir: str=None,
                        saved_model_path: str=None,
                        results_dir: str=None,
                        **kwargs):
    
    input_shape = (*target_size, num_channels)
    params = locals()
    params.pop('kwargs')
    return OmegaConf.create(params)


"""
Path configs to coordinate:

tfrecord_dir


results_dir

model_dir,
    saved_model_path


"""



def initialize_experiment(config, restore_last=True, restore_tfrecords=True):
    date_format = '%Y-%m-%d_%H-%M-%S'

    config.experiment_start_time = datetime.now().strftime(date_format)
    config.experiment_name = '_'.join([config.dataset.dataset_name, config.model.model_name,str(config.dataset.target_size)])
    config.experiment_dir = os.path.join(config.neptune_experiment_dir,config.experiment_name, config.experiment_start_time)
    config.model_dir = os.path.join(config.experiment_dir,'model')
    config.saved_model_path = os.path.join(config.model_dir,'saved_model')
    config.checkpoints_path = os.path.join(config.model_dir,'checkpoints')

    config.results_dir = os.path.join(config.experiment_dir,'results') #, config.experiment_start_time)
    config.tfrecord_dir = f'/media/data/jacob/experimental_data/tfrecords/{config.dataset.dataset_name}'

    if not restore_last:
        clean_experiment_tree(config)
        
    if not restore_tfrecords:
        cleanup_tfrecords(config)

    for k,v in config.items():
        if '_dir' in k:
            ensure_dir_exists(v)

    print('='*40)
    print('Initializing experiment with the following configuration:')
    for k,v in config.items():
        if '_dir' in k or '_path' in k:
            print(f'{k}: {v}')
            print('-'*10)
    print('='*40)
    
    return config
    
    
def clean_experiment_tree(config):
    if not os.path.isdir(config.experiment_dir):
        print(f'Attempted to clean nonexistent experiment directory at {config.experiment_dir}. Continuing without action.')
        return
    print('Cleaning experiment file tree from root:\n',config.experiment_dir)
    shutil.rmtree(config.experiment_dir)
    assert not os.path.isdir(config.experiment_dir)
    
def cleanup_tfrecords(config):
    if not os.path.isdir(config.tfrecord_dir):
        print(f'Attempted to clean nonexistent tfrecord directory at {config.tfrecord_dir}. Continuing without action.')
        return
    print('Cleaning up tfrecord files located at:\n',config.tfrecord_dir)
    shutil.rmtree(config.tfrecord_dir)
    assert not os.path.isdir(config.tfrecord_dir)
            

def flatten_dict(x: dict, exceptions: List[str]=None):
    exceptions = exceptions or []
    out = {}
    for k,v in x.items():
        if isinstance(v,(DictConfig,dict)) and (k not in exceptions):
            out.update(**flatten_dict(v))
        else:
            out[k] = v

    return DictConfig(out)




# @task
def create_dataset(data_fold: DataFold,
                   cfg: DictConfig):
    from pyleaves.train.paleoai_train import load_data, prep_dataset

    split_data, split_datasets, encoder = load_data(data_fold=data_fold,
                                                    exclude_classes=cfg.exclude_classes,
                                                    include_classes=cfg.include_classes,
                                                    use_tfrecords=cfg.use_tfrecords,
                                                    tfrecord_dir=cfg.tfrecord_dir,
                                                    val_split=cfg.val_split,
                                                    samples_per_shard=cfg.samples_per_shard,
                                                    seed=cfg.seed)
    cfg.num_classes = split_datasets['train'].num_classes

    train_data = prep_dataset(split_data['train'],
                              batch_size=cfg.batch_size,
                              buffer_size=cfg.buffer_size,
                              shuffle=True,
                              target_size=cfg.target_size,
                              num_channels=cfg.num_channels,
                              color_mode=cfg.color_mode,
                              num_classes=cfg.num_classes,
                              augmentations=cfg.augmentations,
                              training=True,
                              seed=cfg.seed)
    val_data=None
    if 'val' in split_data:
        if split_data['val'] is not None:
            val_data = prep_dataset(split_data['val'],
                                    batch_size=cfg.batch_size,
                                    target_size=cfg.target_size,
                                    num_channels=cfg.num_channels,
                                    color_mode=cfg.color_mode,
                                    num_classes=cfg.num_classes,
                                    training=False,
                                    seed=cfg.seed)

    test_data = prep_dataset(split_data['test'],
                            batch_size=cfg.batch_size,
                            target_size=cfg.target_size,
                            num_channels=cfg.num_channels,
                            color_mode=cfg.color_mode,
                            num_classes=cfg.num_classes,
                            training=False,
                            seed=cfg.seed)

    split_data = {'train':train_data,'val':val_data,'test':test_data}
    # split_data = {k:v for k,v in split_data.items() if v is not None}

    if cfg.debug:
        
        print('debug==True, plotting 1 batch from each data subset to neptune.')
        for k,v in split_data.items():
            print(f'Uploading first batch from {k}, {cfg.dataset_name}, {cfg.fold_dir}')
            batch = next(iter(v))
            x, y = batch[0].numpy(), batch[1].numpy()
            for i in range(y.shape[0]):
                neptune.log_image(log_name=f'debug-{k}-images', x=i, y=x[i,...])
        import pdb;pdb.set_trace()
        if cfg.debugging.overfit_one_batch:
            print('[DEBUGGING] Attempting to overfit to 1 batch. All Data subsets are limited to first batch.')
            split_data = get_one_batch_dataloaders(split_data)

        if cfg.debugging.log_model_input_stats:
            log_model_input_stats(split_data=split_data)


    return split_data, split_datasets, encoder

def get_one_batch_dataloaders(split_data: dict):
    """Testing utility function for converting all data subset Dataset loaders into Datasets containing 1 batch repeated forever.

    Use case: Overfit to one batch

    split_data: dict mapping subset name:tf.data.Dataset from prep_dataset

    Returns:
        Dict[str,tf.data.Dataset]: Identical dict to input, but with tf.data.Datasets truncated to 1 batch
    """        
    output = {}
    for k,v in split_data.items():
        output[k] = v.take(1).repeat()
    return output

def log_model_input_stats(split_data: dict):
    from neptunecontrib.api import log_table
    import pandas as pd

    stats = []; idx=[]
    for k,v in split_data.items():
        stat_record={}
        batch = next(iter(v))
        x, y = batch[0].numpy(), batch[1].numpy()
        stat_record[f'x_mean'] = np.mean(x)
        # for dim in range(x.ndim):
        #     stat_record[f'x_mean_across_dim-{dim}'] = np.mean(x,axis=dim)
        stats.append(stat_record)
        idx.append(k)
    stats = pd.DataFrame.from_records(stats, index=idx)
    log_table('model_input_stats', stats)
    




def get_callbacks(cfg, model_config, model, fold_id: int=-1, train_data=None, val_data=None, encoder=None):
    from neptunecontrib.monitoring.keras import NeptuneMonitor
    from pyleaves.train.paleoai_train import EarlyStopping, CSVLogger, tf_data2np
    from pyleaves.utils.callback_utils import BackupAndRestore, NeptuneVisualizationCallback,ReduceLROnPlateau
    from pyleaves.utils.neptune_utils import ImageLoggerCallback

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=10, min_lr=model_config.lr*0.1)
    backup_callback = BackupAndRestore(cfg['checkpoints_path'])
    backup_callback.set_model(model)

    print('building callbacks')
    callbacks = [backup_callback,
                 reduce_lr,
                 NeptuneMonitor(),
                 CSVLogger(str(Path(cfg.results_dir,f'results-fold_{fold_id}.csv')), separator=',', append=True),
                 EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)]

    if cfg.log_images and (train_data is not None):
        callbacks.append(ImageLoggerCallback(data=train_data, freq=10, max_images=-1, name='train', encoder=encoder, neptune_logger=neptune,include_predictions=True, log_epochs=cfg.log_epochs))

    if cfg.train_confusion_matrix and (val_data is not None):
        validation_data_np = tf_data2np(data=val_data, num_batches=6)
        neptune_visualization_callback = NeptuneVisualizationCallback(validation_data_np, num_classes=model_config.num_classes)
        callbacks.append(neptune_visualization_callback)

    if cfg.log_images and (val_data is not None):
        callbacks.append(ImageLoggerCallback(data=val_data, freq=10, max_images=-1, name='val', encoder=encoder, neptune_logger=neptune, include_predictions=True, log_epochs=cfg.log_epochs))
        
    return callbacks



# @task
# def fit_model(model, callbacks, train_data, val_data, cfg):

#     history = model.fit(train_data,
#                         epochs=cfg.training['num_epochs'],
#                         callbacks=callbacks,
#                         validation_data=val_data,
#                         validation_freq=1,
#                         shuffle=True,
#                         steps_per_epoch=cfg['steps_per_epoch'],
#                         validation_steps=cfg['validation_steps'],
#                         verbose=1)

#     return history

class Trainer:

    def __init__(self,
                 fold: DataFold, 
                 cfg : DictConfig,
                 worker_id=None,
                 gpu_num: int=None,
                 neptune=None,
                 verbose: bool=True):

        self.fold = fold
        self.config = flatten_dict(cfg, exceptions=['debugging'])

        self.worker_id = worker_id

        # from pyleaves.utils import set_tf_config
        # gpu_num = set_tf_config(gpu_num=gpu_num, num_gpus=1, seed=cfg.seed, wait=fold.fold_id)
        self.gpu_num = gpu_num

        if neptune is None:
            import neptune
        self.neptune = neptune
        self.verbose = verbose

        self.data_config = create_dataset_config(**self.config)
        self.initialize_dataset(self.data_config)

        self.model_config = create_model_config(**OmegaConf.merge(self.config,self.data_config))
        self.initialize_model(self.model_config)
        
    def initialize_dataset(self, data_config: DictConfig):
        # self.data_config = create_dataset_config(**self.config)
        self.data, self.split_datasets, self.encoder = create_dataset(data_fold=self.fold,
                                                                      cfg=data_config)

        if self.config['steps_per_epoch'] is None:
            self.config['steps_per_epoch'] = self.split_datasets['train'].num_samples//data_config['batch_size']

        if (self.config['validation_steps'] is None) and (self.split_datasets['val'] is not None):
            self.config['validation_steps'] = self.split_datasets['val'].num_samples//data_config['batch_size']

        self.train_data, self.val_data, self.test_data = self.data['train'], self.data['val'], self.data['test']

    def initialize_model(self, model_config: DictConfig):
        from pyleaves.train.paleoai_train import build_model
        # self.model_config = create_model_config(**OmegaConf.merge(self.config,self.data_config))#**self.config,**self.data_config)
        self.model = build_model(model_config)
        self.callbacks = get_callbacks(self.config, model_config, self.model, self.fold.fold_id, train_data=self.train_data, val_data=self.val_data, encoder=self.encoder)

        self.model.save(model_config['saved_model_path'])

    def train(self):
        from tensorflow.keras import backend as K

        try:
            history = self.model.fit(self.train_data,
                                epochs=self.model_config.num_epochs,
                                callbacks=self.callbacks,
                                validation_data=self.val_data,
                                validation_freq=1,
                                shuffle=True,
                                steps_per_epoch=self.model_config.steps_per_epoch,
                                validation_steps=self.model_config.validation_steps,
                                verbose=1)
        except Exception as e:
            self.model.save(self.model_config['saved_model_path'])
            print('[Caught Exception, saving model first.\nSaved trained model located at:', self.model_config['saved_model_path'])
            raise e

        self.model.save(self.model_config['saved_model_path'])
        try:
            history_path = str(Path(self.model_config.results_dir,f'training-history_fold-{self.fold.fold_id}.json'))
            with open(history_path,'w') as f:
                json.dump(history.history, f, default=to_serializable)
            if os.path.isfile(history_path):
                print(f'Saved history results for fold-{self.fold.fold_id} to {history_path}')
                self.model_config['history_results_path'] = history_path
            else:
                raise Exception('File save failed')
        except Exception as e:
            print(e)
            print(f'WARNING:  Failed saving training history for fold {self.fold.fold_id} into json file.\n Continuing anyway.')
            
        K.clear_session()
        
        return history

    def predict(self, test_data=None, steps=None):
        '''Currently runs in an infinite loop. do not use'''
        test_data = test_data or self.test_data

        test_data = test_data.map(lambda x,y: (self.model.predict(x), y))

        all_y_true, all_y_pred = [],[]
        i=0
        for y_pred, y_true in test_data:
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            print('Finished batch',i)
            i+=1
            if i >= steps:
                break

        return np.stack(all_y_true), np.stack(all_y_pred)


    def evaluate(self, test_data=None, steps: int=None, num_classes: int=None, confusion_matrix=True):

        print('Preparing for model evaluation')

        test_data = test_data or self.test_data
        num_classes = num_classes or self.model_config.num_classes

        text_labels = self.encoder.classes
        steps = steps or self.split_datasets['test'].num_samples//self.data_config['batch_size']

        callbacks=[]
        if confusion_matrix:
            from pyleaves.utils.callback_utils import NeptuneVisualizationCallback
            callbacks.append(NeptuneVisualizationCallback(test_data, num_classes=num_classes, text_labels=text_labels, steps=steps))

        test_results = self.model.evaluate(test_data, callbacks=callbacks, steps=steps, verbose=1)

        print('Model evaluation complete.')
        print('Results:')
        for m, result in zip(self.model.metrics_names, test_results):
            print(f'{m}: {result}')

            neptune.log_metric(f'test_{m}', result)

        return {m:result for m,result in zip(self.model.metrics_names, test_results)}




@hydra.main(config_path='configs', config_name='config')
def main(cfg : DictConfig):


    from pyleaves.utils import set_tf_config
    gpu_num = set_tf_config(gpu_num=cfg.gpu_num, num_gpus=1)

    import tensorflow as tf
    from tensorflow.keras import backend as K
    K.clear_session()
    preprocess_input(tf.zeros([4, 224, 224, 3]))


    cfg = initialize_experiment(cfg, restore_last=cfg.restore_last, restore_tfrecords=True)

    print('Dataset config: \n',cfg.dataset.pretty())
    kfold_loader = KFoldLoader(root_dir=cfg.dataset.fold_dir)
    if cfg.fold_id is None:
        cfg.fold_id = 0
    fold = kfold_loader.folds[cfg.fold_id]


    neptune.init(project_qualified_name=cfg.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    with neptune.create_experiment(name=cfg.experiment_name+'-'+str(cfg.dataset.dataset_name), params=params):

        trainer = Trainer(fold,
                          cfg,
                          neptune=neptune,
                          verbose=True)

        date_format = '%Y-%m-%d_%H-%M-%S'
        train_start_time = datetime.now().strftime(date_format)

        print(f'Trainer constructed. Beginning training at {train_start_time}')

        history = trainer.train()

        test_results = trainer.evaluate(steps=20, confusion_matrix=cfg.eval_confusion_matrix)




if __name__=='__main__':



    main()






# y_true, y_pred = predict_single_fold(model=model,
#                                             fold=fold,
#                                             cfg=cfg,
#                                             predict_on_full_dataset=False,
#                                             worker_id=worker_id,
#                                             save=True,
#                                             verbose=verbose)




# def train_pipeline(fold: DataFold, 
#                    cfg : DictConfig,
#                    worker_id=None,
#                    gpu_num: int=None,
#                    neptune=None,
#                    verbose: bool=True) -> None:
#     from pyleaves.utils import set_tf_config
#     set_tf_config(gpu_num=gpu_num, num_gpus=1, seed=cfg.misc.seed, wait=fold.fold_id)
    
#     import tensorflow as tf
#     from tensorflow.keras import backend as K
#     from pyleaves.train.paleoai_train import preprocess_input, create_dataset, build_model
#     if neptune is None:
#         import neptune
#     K.clear_session()
#     preprocess_input(tf.zeros([4, 224, 224, 3]))

#     # if verbose:
#     #     print('='*20)
#     #     print(f'RUNNING: fold {fold.fold_id} in process {worker_id or "None"}')
#     #     print(fold)
#     #     print(cfg.tfrecord_dir)
#     #     print('='*20)
        
#     data, split_datasets, encoder = create_dataset(data_fold=fold,
#                                                     cfg=cfg)

#     train_data, val_data, test_data = data['train'], data['val'], data['test']

#     # if val_data is None:
#     #     log_dataset(cfg=cfg, train_dataset=train_dataset) #, neptune=neptune)
#     # else:
#     #     log_dataset(cfg=cfg, train_dataset=train_dataset, val_split=cfg.dataset.val_split) #, neptune=neptune)

#     # cfg['model']['num_classes'] = cfg['dataset']['num_classes']
#     # model_config = cfg.model

#     model = build_model(model_config)
#     callbacks = get_callbacks(cfg, model_config, model, fold, test_data)

#     # log_config(cfg=cfg, neptune=neptune)
#     # model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
#     # for k,v in model_config.items():
#     #     neptune.set_property(k, v)
#     # fold_model_path = str(Path(cfg['saved_model_path'],f'fold-{fold.fold_id}'))

#     model.save(fold_model_path)
#     # print('Saved model located at:', fold_model_path)
#     # neptune.set_property('model_path',fold_model_path)
#     # print(f'Initiating model.fit for fold-{fold.fold_id}')

#     try:
#         history = model.fit(train_data,
#                             epochs=cfg.training['num_epochs'],
#                             callbacks=callbacks,
#                             validation_data=val_data,
#                             validation_freq=1,
#                             shuffle=True,
#                             steps_per_epoch=cfg['steps_per_epoch'],
#                             validation_steps=cfg['validation_steps'],
#                             verbose=1)
#     except Exception as e:
#         # model.save(str(Path(cfg['saved_model_path'],f'fold-{fold.fold_id}')))
#         # print('[Caught Exception, saving model first.\nSaved trained model located at:', fold_model_path)
#         raise e    
#     # print('Saved trained model located at:', fold_model_path)
#     model.save(str(Path(cfg['saved_model_path'],f'fold-{fold.fold_id}')))
#     y_true, y_pred = predict_single_fold(model=model,
#                                          fold=fold,
#                                          cfg=cfg,
#                                          predict_on_full_dataset=False,
#                                          worker_id=worker_id,
#                                          save=True,
#                                          verbose=verbose)

#     try:
#         history_path = str(Path(cfg.results_dir,f'training-history_fold-{fold.fold_id}.json'))
#         with open(history_path,'w') as f:
#             json.dump(history.history, f, default=to_serializable)
#         if os.path.isfile(history_path):
#             print(f'Saved history results for fold-{fold.fold_id} to {history_path}')
#             cfg.training['history_results_path'] = history_path
#         else:
#             raise Exception('File save failed')
#     except Exception as e:
#         print(e)
#         print(f'WARNING:  Failed saving training history for fold {fold.fold_id} into json file.\n Continuing anyway.')
        
#     K.clear_session()
    
   
#     return history


# @hydra.main(config_path=Path(CONFIG_DIR,'config.yaml'))
# def main():
    

# def build_base_vgg16_RGB(cfg):

#     base = tf.keras.applications.vgg16.VGG16(weights=cfg['weights'],
#                                              include_top=False,
#                                              input_tensor=Input(shape=(*cfg['target_size'],3)))

#     return base


# def build_classifier_head(base, num_classes=10, cfg: DictConfig=None):
#     if cfg is None:
#         global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#         dense1 = tf.keras.layers.Dense(2048,activation='relu',name='dense1')
#         dense2 = tf.keras.layers.Dense(512,activation='relu',name='dense2')
#         prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
#         model = tf.keras.Sequential([
#             base,
#             global_average_layer,dense1,dense2,
#             prediction_layer
#             ])

#     else:
#         layers = [base]
#         layers.append(tf.keras.layers.GlobalAveragePooling2D())
#         for layer_num, layer_units in enumerate(cfg.head_layers):
#             layers.append(tf.keras.layers.Dense(layer_units,activation='relu',name=f'dense{layer_num}'))
        
#         layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
#         model = tf.keras.Sequential(layers)

#     return model


# def build_model(cfg: DictConfig):
#     '''
#         model_config(model_name: str='resnet_50_v2',
#                      optimizer: str='Adam',
#                      loss: str='categorical_crossentropy',
#                      weights: str='imagenet',
#                      lr: float=4.0e-5,
#                      lr_decay: float=None,
#                      lr_momentum: float=None,
#                      lr_decay_epochs: int=10,
#                      regularization: Dict[str,float]={'l1': None},
#                      METRICS: List[str]=['f1','accuracy'],
#                      head_layers: List[int]=[1024,256],
#                      batch_size: int=16,
#                      buffer_size: int=200,
#                      num_epochs: int=150,
#                      frozen_layers: List[int]=None,
#                      augmentations: List[Dict[str,float]]=[{'flip': 1.0}],
#                      num_classes: int=None,
#                      target_size: Tuple[int,int]=(512,512),
#                      num_channels: int=3)
#     '''

#     if cfg['model_name']=='vgg16':
#         if cfg['num_channels']==1:
#             model_builder = vgg16.VGG16GrayScale(cfg)
#             build_base = model_builder.build_base
#         else:
#             build_base = partial(build_base_vgg16_RGB, cfg=cfg)

#     elif cfg['model_name'].startswith('resnet'):
#         model_builder = resnet.ResNet(cfg)
#         build_base = model_builder.build_base

#     base = build_base()
#     model = build_classifier_head(base, num_classes=cfg['num_classes'], cfg=cfg)
    
#     model = base_model.Model.add_regularization(model, **cfg.regularization)

#     # initial_learning_rate = cfg['lr']
#     # lr_schedule = cfg['lr'] #tf.keras.optimizers.schedules.ExponentialDecay(
#                             # initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True

#     if cfg['optimizer'] == "RMSprop":
#         optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['lr'], momentum=cfg['momentum'], decay=cfg['decay'])
#     elif cfg['optimizer'] == "SGD":
#         optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['lr'], momentum=cfg['momentum'])
#     elif cfg['optimizer'] == "Adam":
#         optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['lr'])

#     if cfg['loss']=='categorical_crossentropy':
#         loss = 'categorical_crossentropy'

#     METRICS = []
#     if 'f1' in cfg['METRICS']:
#         METRICS.append(tfa.metrics.F1Score(num_classes=cfg['num_classes'],
#                                            average='weighted',
#                                            name='weighted_f1'))
#     if 'accuracy' in cfg['METRICS']:
#         METRICS.append('accuracy')
#     if 'precision' in cfg['METRICS']:
#         METRICS.append(tf.keras.metrics.Precision())
#     if 'recall' in cfg['METRICS']:
#         METRICS.append(tf.keras.metrics.Recall())

#     model.compile(optimizer=optimizer,
#                   loss=loss,
#                   metrics=METRICS)

#     model.save(cfg['saved_model_path'])
#     return model