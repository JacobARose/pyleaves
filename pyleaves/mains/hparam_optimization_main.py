'''
hparam_optimization_main.py


This is meant to be a basic script for initiating or resuming an Optuna hyperparameter-optimization study.


python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py' stage_0.misc.use_tfrecords=False num_gpus=4 n_jobs=4


python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py' stage_0.dataset.dataset_name="Leaves-PNAS"
'''

from box import Box
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import os
import itertools
from typing import List
import copy
import psutil
import time
from paleoai_data.utils.kfold_cross_validation import KFoldLoader
from pyleaves import RESOURCES_DIR
from pyleaves.mains.paleoai_main import restore_or_initialize_experiment, train_single_fold, log_config#, train_paleoai_dataset
from pyleaves.mains import paleoai_main
CONFIG_DIR = str(Path(RESOURCES_DIR,'..','..','configs','hydra'))
from pyleaves.utils.neptune_utils import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
from optuna.samplers import TPESampler


def get_optimizer_config(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = Box()
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["lr"] = trial.suggest_float("rmsprop_learning_rate", 1e-5, 1e-1, log=True)
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["lr"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-2, log=True)
    elif optimizer_selected == "SGD":
        kwargs["lr"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-2, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    kwargs['optimizer'] = optimizer_selected
    
    return kwargs


def get_model_config(trial, cfg: DictConfig):
    cfg['model']['lr'] = cfg['lr']
    cfg['model']['input_shape'] = (*cfg.dataset['target_size'],cfg.dataset['num_channels'])
    cfg['model']['model_dir'] = cfg['model_dir']
    cfg['model']['num_classes'] = cfg['dataset']['num_classes']

    optimizer_params = get_optimizer_config(trial)
    model_config = OmegaConf.merge(cfg.model, cfg.training, optimizer_params)
    # model_config['n_layers'] = trial.suggest_int("n_layers", 1, 3)
    model_config['regularization'] = {'l2':trial.suggest_float("l2", 1e-10, 1e-3, log=True)}

    return model_config

class Objective:

    def __init__(self, cfg: DictConfig):
        self.config = cfg

        cfg_0 = cfg.stage_0
        self.kfold_loader = KFoldLoader(root_dir=cfg_0.dataset.fold_dir)
        self.num_splits = self.kfold_loader.num_splits

    def __call__(self, trial):
        config = copy.deepcopy(self.config)
        config.model = get_model_config(trial, config)

        fold_id = trial.number % self.num_splits
        fold = self.kfold_loader.folds[fold_id]
        worker_id = psutil.Process(os.getpid())
        history = paleoai_main.optuna_train_single_fold(fold, config, worker_id)

        best_accuracy = np.max(history.history['val_accuracy'])

        return 1.0 - best_accuracy


def optimize_hyperparameters(cfg : DictConfig, fold_ids: List[int]=[0], n_trials: int=5, n_jobs: int=1, gc_after_trial=True, verbose: bool=False) -> None:

    print(f'Beginning training of models with fold_ids: {fold_ids}')
        
    neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

    sampler = TPESampler(seed=cfg.stage_0.misc.seed)
    study = optuna.create_study(study_name=cfg.study_name, sampler=sampler, direction="maximize", storage=cfg.db.storage)

    objective = Objective(cfg)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=gc_after_trial, callbacks=[neptune_callback])

    print("Number of finished trials: ", len(study.trials))
    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    log_config(cfg=cfg, verbose=verbose, neptune=neptune)

    return trial







@hydra.main(config_path=Path(CONFIG_DIR,'config.yaml'))
def main(cfg : DictConfig) -> None:

    OmegaConf.set_struct(cfg, False)
    cfg = restore_or_initialize_experiment(cfg, restore_last=cfg.experiment.restore_last, prefix='log_dir__', verbose=0)


    neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    # with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name)+'-'+str(cfg.fold_id), params=params):
        # train_pyleaves_dataset(cfg)
    with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name)+'-'+str(fold.fold_id), params=params):
        optimize_hyperparameters(cfg=cfg, fold_ids=list(range(10)), n_trials=cfg.n_trials, n_jobs=cfg.n_jobs, gc_after_trial=cfg.gc_after_trial, verbose=True)

if __name__=="__main__":

    main()