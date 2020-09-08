# @Author: Jacob A Rose
# @Date:   Sun September 6th 2020, 10:42 pm
# @Email:  jacobrose@brown.edu
# @Filename: pipeline_1.py

"""
This is meant to be a modularization of common model training scripts, allowing for further customization of the order and content of each piepline node/step/stage.


"""



from prefect import Flow, task
from paleoai_data.utils.kfold_cross_validation import DataFold
from paleoai_data.dataset_drivers.base_dataset import BaseDataset
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


def create_dataset_config(dataset_name: str='Leaves-PNAS',
                          batch_size: int=1,
                          buffer_size: int=200,
                          exclude_classes: List[str]=['notcataloged','notcatalogued', 'II. IDs, families uncertain', 'Unidentified'],
                          include_classes: List[str]=[],
                          target_size: Tuple[int,int]=(512,512),
                          num_channels: int=3,
                          color_mode: str='grayscale',
                          val_split: float=0.0,
                          augmentations: Dict[str,float]={'flip':0.0},
                          seed: int=None,
                          fold_dir= '/home/jacob/projects/paleoai_data/paleoai_data/v0_2/data/staged_data/Leaves-PNAS/ksplit_10',
                          use_tfrecords: bool=True,
                          tfrecord_dir: str=None,
                          samples_per_shard: int=300):
    if tfrecord_dir:
        ensure_dir_exists(tfrecord_dir)
    return OmegaConf.create(locals())


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
                        head_layers: List[int]=[1024,256],
                        batch_size: int=16,
                        buffer_size: int=200,
                        num_epochs: int=150,
                        frozen_layers: List[int]=None,
                        augmentations: List[Dict[str,float]]=[{'flip': 1.0}],
                        num_classes: int=None,
                        target_size: Tuple[int,int]=(512,512),
                        num_channels: int=3):
    
    input_shape = (*target_size, num_channels)
    return OmegaConf.create(locals())





@task
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
    if split_data['val'] is not None:
        val_data = prep_dataset(split_data['val'],
                                batch_size=cfg.batch_size,
                                target_size=cfg.target_size,
                                num_channels=cfg.num_channels,
                                color_mode=cfg.color_mode,
                                num_classes=cfg.num_classes,
                                training=False,
                                seed=cfg.seed)
    else:
        val_data=None

    test_data = prep_dataset(split_data['test'],
                            batch_size=cfg.batch_size,
                            target_size=cfg.target_size,
                            num_channels=cfg.num_channels,
                            color_mode=cfg.color_mode,
                            num_classes=cfg.num_classes,
                            training=False,
                            seed=cfg.seed)

    split_data = {'train':train_data,'val':val_data,'test':test_data}

    return split_data, split_datasets, encoder


def get_callbacks(cfg, model_config, model, fold, test_data):
    from neptunecontrib.monitoring.keras import NeptuneMonitor
    from pyleaves.train.paleoai_train import EarlyStopping, CSVLogger, tf_data2np
    from pyleaves.utils.callback_utils import BackupAndRestore, NeptuneVisualizationCallback, ReduceLROnPlateau

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=10, min_lr=model_config.lr*0.1)
    backup_callback = BackupAndRestore(cfg['checkpoints_path'])
    backup_callback.set_model(model)

    validation_data_np = tf_data2np(data=test_data, num_batches=2)
    neptune_visualization_callback = NeptuneVisualizationCallback(validation_data_np, num_classes=cfg.dataset.num_classes)

    print('building callbacks')
    callbacks = [backup_callback,
                 reduce_lr,
                 NeptuneMonitor(),
                 neptune_visualization_callback,
                 CSVLogger(str(Path(cfg.results_dir,f'results-fold_{fold.fold_id}.csv')), separator=',', append=True),
                 EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)]
    return callbacks


def build_base_vgg16_RGB(cfg):

    base = tf.keras.applications.vgg16.VGG16(weights=cfg['weights'],
                                             include_top=False,
                                             input_tensor=Input(shape=(*cfg['target_size'],3)))

    return base


def build_classifier_head(base, num_classes=10, cfg: DictConfig=None):
    if cfg is None:
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        dense1 = tf.keras.layers.Dense(2048,activation='relu',name='dense1')
        dense2 = tf.keras.layers.Dense(512,activation='relu',name='dense2')
        prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
        model = tf.keras.Sequential([
            base,
            global_average_layer,dense1,dense2,
            prediction_layer
            ])

    else:
        layers = [base]
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        for layer_num, layer_units in enumerate(cfg.head_layers):
            layers.append(tf.keras.layers.Dense(layer_units,activation='relu',name=f'dense{layer_num}'))
        
        layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
        model = tf.keras.Sequential(layers)

    return model


def build_model(cfg: DictConfig):
    '''
        model_config(model_name: str='resnet_50_v2',
                     optimizer: str='Adam',
                     loss: str='categorical_crossentropy',
                     weights: str='imagenet',
                     lr: float=4.0e-5,
                     lr_decay: float=None,
                     lr_momentum: float=None,
                     lr_decay_epochs: int=10,
                     regularization: Dict[str,float]={'l1': None},
                     METRICS: List[str]=['f1','accuracy'],
                     head_layers: List[int]=[1024,256],
                     batch_size: int=16,
                     buffer_size: int=200,
                     num_epochs: int=150,
                     frozen_layers: List[int]=None,
                     augmentations: List[Dict[str,float]]=[{'flip': 1.0}],
                     num_classes: int=None,
                     target_size: Tuple[int,int]=(512,512),
                     num_channels: int=3)
    '''

    if cfg['model_name']=='vgg16':
        if cfg['num_channels']==1:
            model_builder = vgg16.VGG16GrayScale(cfg)
            build_base = model_builder.build_base
        else:
            build_base = partial(build_base_vgg16_RGB, cfg=cfg)

    elif cfg['model_name'].startswith('resnet'):
        model_builder = resnet.ResNet(cfg)
        build_base = model_builder.build_base

    base = build_base()
    model = build_classifier_head(base, num_classes=cfg['num_classes'], cfg=cfg)
    
    model = base_model.Model.add_regularization(model, **cfg.regularization)

    # initial_learning_rate = cfg['lr']
    # lr_schedule = cfg['lr'] #tf.keras.optimizers.schedules.ExponentialDecay(
                            # initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True

    if cfg['optimizer'] == "RMSprop":
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['lr'], momentum=cfg['momentum'], decay=cfg['decay'])
    elif cfg['optimizer'] == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['lr'], momentum=cfg['momentum'])
    elif cfg['optimizer'] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['lr'])

    if cfg['loss']=='categorical_crossentropy':
        loss = 'categorical_crossentropy'

    METRICS = []
    if 'f1' in cfg['METRICS']:
        METRICS.append(tfa.metrics.F1Score(num_classes=cfg['num_classes'],
                                           average='weighted',
                                           name='weighted_f1'))
    if 'accuracy' in cfg['METRICS']:
        METRICS.append('accuracy')
    if 'precision' in cfg['METRICS']:
        METRICS.append(tf.keras.metrics.Precision())
    if 'recall' in cfg['METRICS']:
        METRICS.append(tf.keras.metrics.Recall())

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=METRICS)

    model.save(cfg['saved_model_path'])
    return model


@task
def fit_model(model, callbacks, train_data, val_data, cfg):

    history = model.fit(train_data,
                        epochs=cfg.training['num_epochs'],
                        callbacks=callbacks,
                        validation_data=val_data,
                        validation_freq=1,
                        shuffle=True,
                        steps_per_epoch=cfg['steps_per_epoch'],
                        validation_steps=cfg['validation_steps'],
                        verbose=1)

    return history









def train_pipeline(fold: DataFold, 
                   cfg : DictConfig,
                   worker_id=None,
                   gpu_num: int=None,
                   neptune=None,
                   verbose: bool=True) -> None:
    from pyleaves.utils import set_tf_config
    set_tf_config(gpu_num=gpu_num, num_gpus=1, seed=cfg.misc.seed, wait=fold.fold_id)
    
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from pyleaves.train.paleoai_train import preprocess_input, create_dataset, build_model
    if neptune is None:
        import neptune
    K.clear_session()
    preprocess_input(tf.zeros([4, 224, 224, 3]))

    # if verbose:
    #     print('='*20)
    #     print(f'RUNNING: fold {fold.fold_id} in process {worker_id or "None"}')
    #     print(fold)
    #     print(cfg.tfrecord_dir)
    #     print('='*20)
        
    data, split_datasets, encoder = create_dataset(data_fold=fold,
                                                    cfg=cfg)

    train_data, val_data, test_data = data['train'], data['val'], data['test']

    # if val_data is None:
    #     log_dataset(cfg=cfg, train_dataset=train_dataset) #, neptune=neptune)
    # else:
    #     log_dataset(cfg=cfg, train_dataset=train_dataset, val_split=cfg.dataset.val_split) #, neptune=neptune)

    # cfg['model']['num_classes'] = cfg['dataset']['num_classes']
    # model_config = cfg.model

    model = build_model(model_config)
    callbacks = get_callbacks(cfg, model_config, model, fold, test_data)

    # log_config(cfg=cfg, neptune=neptune)
    # model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
    # for k,v in model_config.items():
    #     neptune.set_property(k, v)
    # fold_model_path = str(Path(cfg['saved_model_path'],f'fold-{fold.fold_id}'))

    model.save(fold_model_path)
    # print('Saved model located at:', fold_model_path)
    # neptune.set_property('model_path',fold_model_path)
    # print(f'Initiating model.fit for fold-{fold.fold_id}')

    try:
        history = model.fit(train_data,
                            epochs=cfg.training['num_epochs'],
                            callbacks=callbacks,
                            validation_data=val_data,
                            validation_freq=1,
                            shuffle=True,
                            steps_per_epoch=cfg['steps_per_epoch'],
                            validation_steps=cfg['validation_steps'],
                            verbose=1)
    except Exception as e:
        # model.save(str(Path(cfg['saved_model_path'],f'fold-{fold.fold_id}')))
        # print('[Caught Exception, saving model first.\nSaved trained model located at:', fold_model_path)
        raise e    
    # print('Saved trained model located at:', fold_model_path)
    model.save(str(Path(cfg['saved_model_path'],f'fold-{fold.fold_id}')))
    y_true, y_pred = predict_single_fold(model=model,
                                         fold=fold,
                                         cfg=cfg,
                                         predict_on_full_dataset=False,
                                         worker_id=worker_id,
                                         save=True,
                                         verbose=verbose)

    try:
        history_path = str(Path(cfg.results_dir,f'training-history_fold-{fold.fold_id}.json'))
        with open(history_path,'w') as f:
            json.dump(history.history, f, default=to_serializable)
        if os.path.isfile(history_path):
            print(f'Saved history results for fold-{fold.fold_id} to {history_path}')
            cfg.training['history_results_path'] = history_path
        else:
            raise Exception('File save failed')
    except Exception as e:
        print(e)
        print(f'WARNING:  Failed saving training history for fold {fold.fold_id} into json file.\n Continuing anyway.')
        
    K.clear_session()
    
   
    return history


# @hydra.main(config_path=Path(CONFIG_DIR,'config.yaml'))
# def main():
    
