# @Author: Jacob A Rose
# @Date:   Wed September 2nd 2020, 4:00 pm
# @Email:  jacobrose@brown.edu
# @Filename: model_interpretability.py


'''
This script is meant to store definitions of various utilities for interpreting the reasoning of black box models


Current interpretability tools include:

    - CAM (Class Activation Map)


python '/home/jacob/projects/pyleaves/pyleaves/analysis/model_interpretability.py' num_gpus=1 dataset.dataset_name='Leaves-PNAS' experiment.experiment_name='hparam_studies_Leaves-PNAS_resnet_50_v2_res512' dataset.target_size=[768,768] stage_0.model.head_layers=[512,256] saved_model_path="/media/data/jacob/sandbox_logs/PNAS_resnet_50_v2/log_dir__2020-09-03_00-47-32/model_dir/saved_model/fold-6"


'''

import copy
from datetime import datetime
import hydra
import numpy as np
import scipy
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
from typing import Union, List, Tuple
from tqdm import trange
from paleoai_data.utils.kfold_cross_validation import DataFold
from pyleaves.train.paleoai_train import create_prediction_dataset
from pyleaves.utils.neptune_utils import neptune
from pyleaves.utils import ensure_dir_exists
import pyleaves
CONFIG_DIR = str(Path(pyleaves.RESOURCES_DIR,'..','..','configs','hydra'))
date_format = '%Y-%m-%d_%H-%M-%S'

def generateCAM(model, fold: DataFold, cfg: DictConfig, use_max_samples: Union[int,str]='all', neptune=None):
    import tensorflow as tf

    target_size=cfg.dataset.target_size
    num_channels = cfg.dataset.num_channels

    gap_weights = model.layers[-1].get_weights()[0]
    for i, l in enumerate(model.layers[::-1]):
        if 'global_average_pooling' in l.name:
            CAM_output_layer = model.layers[-i]
    model_output_layer = model.layers[-1]
    cam_model = tf.keras.models.Model(inputs=model.input,
                      outputs=(CAM_output_layer.output, model_output_layer.output)) 

    inputs = tf.keras.Input(shape=(target_size,num_channels))
    cam_model(inputs)
    cam_model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

    pred_data, pred_dataset, encoder = create_prediction_dataset(data_fold = fold,
                                                                 predict_on_full_dataset=False,
                                                                 batch_size=1,
                                                                 exclude_classes=cfg.dataset.exclude_classes,
                                                                 include_classes=cfg.dataset.include_classes,
                                                                 target_size=target_size,
                                                                 num_channels=cfg.dataset.num_channels,
                                                                 color_mode=cfg.dataset.color_mode,
                                                                 seed=cfg.misc.seed)

    x_true, y_true = [], []
    print(pred_dataset.num_samples)
    data_iter = iter(pred_data)
    for i in trange(pred_dataset.num_samples):
        x, y = next(data_iter)
        x_true.append(x.numpy())
        y_true.append(y.numpy())

    x_true = np.vstack(x_true)
    y_true = np.vstack(y_true)

    class_names = encoder.classes

    features, results = cam_model.predict(x_true)

    if use_max_samples=='all':
        max_samples = features.shape[0]
    else:
        max_samples = use_max_samples

    for idx in range(max_samples):
        # get the feature map of the test image
        img_features = features[idx, :, :, :]

        # map the feature map to the original size
        height_roomout = target_size[0] / img_features.shape[0]
        width_roomout = target_size[0] / img_features.shape[1]
        cam_features = scipy.ndimage.zoom(img_features, (height_roomout, width_roomout, 1), order=2)
            
        # get the predicted label with the maximum probability
        pred = np.argmax(results[idx])
        
        cam_weights = gap_weights[:, pred]
        cam_output = np.dot(cam_features, cam_weights)

        fig, ax = plt.subplots(1,1)        
        # draw the class activation map
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        true_class = class_names[np.argmax(y_true[idx,:])]
        sample_name = f'{idx}_{true_class}'
        
        buf = f'True Class = {true_class}, Predicted Class = {class_names[pred]}, Probability = {str(results[idx][pred])}'
        plt.xlabel(buf)
        plt.imshow(x_true[idx,...], alpha=0.5)
        plt.imshow(cam_output, cmap='jet', alpha=0.5)

        if neptune is None:
            fig.savefig(str(Path(cfg.results_dir, sample_name+'.png')))
        else:
            neptune.log_image(sample_name, fig)
        del fig
        


def initialize_dirs(cfg: DictConfig, experiment_start_time=None):
    
    cfg.experiment.experiment_name = '_'.join([cfg.dataset.dataset_name, cfg.model.model_name])
    cfg.experiment.experiment_dir = str(Path(cfg.experiment.neptune_experiment_dir, cfg.experiment.experiment_name))
    if 'db' in cfg:
        ensure_dir_exists(Path("/" + cfg.db.storage.strip('sqlite:/')).parent)
    cfg.experiment.experiment_start_time = experiment_start_time or datetime.now().strftime(date_format)
    cfg.update(log_dir = str(Path(cfg.experiment.experiment_dir, 'log_dir__'+cfg.experiment.experiment_start_time)))
    cfg.update(results_dir = str(Path(cfg.log_dir,'results')))
    cfg.update(tfrecord_dir = str(Path(cfg.log_dir,'tfrecord_dir')))
    if cfg.saved_model_path is None:
        cfg.saved_model_path = str(Path(cfg.model_dir) / Path('saved_model'))
    else:
        cfg.model_dir = str(Path(cfg.saved_model_path).parent)
    cfg.checkpoints_path = str(Path(cfg.model_dir) / Path('checkpoints'))

def restore_experiment_dirs(cfg, prefix='log_dir__', verbose=0):
#     date_format = '%Y-%m-%d_%H-%M-%S'
    cfg = copy.deepcopy(cfg)
    cfg.experiment.experiment_name = '_'.join([cfg.dataset.dataset_name, cfg.model.model_name])
    cfg.experiment.experiment_dir = os.path.join(cfg.experiment.neptune_experiment_dir, cfg.experiment.experiment_name)
    ensure_dir_exists(cfg.experiment.experiment_dir)

    experiment_files = [(exp_name.split(prefix)[-1], exp_name) for exp_name in os.listdir(cfg.experiment.experiment_dir)]
    keep_files = []
    for i in range(len(experiment_files)):
        exp = experiment_files[i]
        try:
            keep_files.append((datetime.strptime(exp[0], date_format), exp[1]))
            if verbose >= 1: print(f'Found previous experiment {exp[1]}')
        except ValueError:
            if verbose >=2: print(f'skipping invalid file {exp[1]}')
            pass

    experiment_files = sorted(keep_files, key= lambda exp: exp[0])
    if type(experiment_files)==list and len(experiment_files)>0:
        experiment_file = experiment_files[-1]
        cfg.experiment.experiment_start_time = experiment_file[0].strftime(date_format)
        initialize_dirs(cfg, experiment_start_time=cfg.experiment.experiment_start_time)
        if verbose >= 1: print(f'Continuing experiment with start time =', cfg.experiment.experiment_start_time)
        return cfg

    print('No previous experiment in',cfg.experiment.experiment_dir, 'with prefix',prefix)
    cfg.experiment.experiment_start_time = datetime.now().strftime(date_format)
    initialize_dirs(cfg, experiment_start_time=cfg.experiment.experiment_start_time)
    if verbose >= 1: print('Initializing new experiment at time:', cfg.experiment.experiment_start_time )
    return cfg



@hydra.main(config_path=Path(CONFIG_DIR,'interpret_model_config.yaml'))
def main(cfg : DictConfig) -> None:

    from pyleaves.utils import set_tf_config
    set_tf_config(num_gpus=cfg.num_gpus, seed=cfg.misc.seed, wait=0)
    import tensorflow as tf
    from paleoai_data.utils.kfold_cross_validation import KFoldLoader

    OmegaConf.set_struct(cfg, False)
    restore_experiment_dirs(cfg, verbose=cfg.verbose)

    kfold_loader = KFoldLoader(root_dir=cfg.dataset.fold_dir)
    fold_id = cfg.fold_id or 1
    fold = kfold_loader.folds[fold_id]

    model = tf.keras.models.load_model(str(Path(cfg['saved_model_path'],f'fold-{fold.fold_id}')))

    neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.dataset.dataset_name), params=params):
        generateCAM(model=model, fold=fold, cfg=cfg, use_max_samples=cfg.misc.use_max_samples, neptune=neptune)

if __name__=="__main__":

    main()

