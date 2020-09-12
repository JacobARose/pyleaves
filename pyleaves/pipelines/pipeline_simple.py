# from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, DictConfig

# with initialize(config_path="configs"):
#     config = compose(config_name="config", overrides=['dataset@dataset=PNAS','use_tfrecords=False'])
#     print(config.pretty())
import hydra

@hydra.main(config_path='configs', config_name='config')
def main(config : DictConfig):

    from pyleaves.train.paleoai_train import build_model
    from pyleaves.pipelines.pipeline_1 import *
    from pyleaves.utils import set_tf_config
    gpu_num = set_tf_config(gpu_num=config.gpu_num, num_gpus=1)
    import tensorflow as tf
    from tensorflow.keras import backend as K
    K.clear_session()
    preprocess_input(tf.zeros([4, 224, 224, 3]));

    config = initialize_experiment(config, restore_last=config.restore_last, restore_tfrecords=True)
    # config.dataset.fold_dir = '/home/jacob/projects/paleoai_data/paleoai_data/v0_2/data/staged_data/PNAS_family_100/ksplit_2'
    kfold_loader = KFoldLoader(root_dir=config.dataset.fold_dir)
    if config.fold_id is None:
        config.fold_id = 0
    fold = kfold_loader.folds[config.fold_id]
    # fold
    fold.train_data

    config = flatten_dict(config, exceptions=['debugging'])
    data_config = create_dataset_config(**config)

    # config=cfg
    data, split_datasets, encoder = create_dataset(data_fold=fold,
                                                cfg=data_config)

    if config['steps_per_epoch'] is None:
        config['steps_per_epoch'] = split_datasets['train'].num_samples//data_config['batch_size']

    if (config['validation_steps'] is None) and (split_datasets['val'] is not None):
        config['validation_steps'] = split_datasets['val'].num_samples//data_config['batch_size']

    train_data, val_data, test_data = data['train'], data['val'], data['test']
    data_config.num_classes=split_datasets['train'].num_classes


    print('Dataset config: \n',data_config.pretty())

    model_config = create_model_config(**OmegaConf.merge(config,data_config))
    model = build_model(model_config)
    print('Model config: \n',model_config.pretty())
    model.summary()
    callbacks = get_callbacks(config, model_config, model, fold.fold_id, train_data=train_data, val_data=val_data, encoder=encoder)

    neptune.init(project_qualified_name=config.neptune_project_name)
    config.dataset = data_config
    config.model = model_config
    params=OmegaConf.to_container(config)
    print(config.pretty())
    with neptune.create_experiment(name='-'.join([config.experiment_name, str(config.dataset_name),str(config.input_shape)]), params=params):

        history = model.fit(train_data,
                            epochs=model_config.num_epochs,
                            callbacks=callbacks,
                            validation_data=val_data,
                            validation_freq=1,
                            shuffle=True,
                            steps_per_epoch=model_config.steps_per_epoch,
                            validation_steps=model_config.validation_steps,
                            verbose=1)