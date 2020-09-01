'''
This is meant to be a basic script for launching 


python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py' stage_0.misc.use_tfrecords=False num_gpus=4 n_jobs=4


python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py' stage_0.dataset.dataset_name="Leaves-PNAS"
'''

import hydra
# from multiprocessing import Pool
# import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
# import os
import itertools
from typing import List
import copy
import time
from paleoai_data.utils.kfold_cross_validation import KFoldLoader
from pyleaves.mains.paleoai_main import restore_or_initialize_experiment, train_single_fold, log_config#, train_paleoai_dataset
from pyleaves import RESOURCES_DIR
from pyleaves.utils.multiprocessing_utils import perform_concurrent_tasks, RunAsCUDASubprocess
from pyleaves.utils import multiprocessing_utils
from pyleaves.mains import paleoai_main
CONFIG_DIR = str(Path(RESOURCES_DIR,'..','..','configs','hydra'))
from pyleaves.utils.neptune_utils import neptune






def train_paleoai_dataset(cfg : DictConfig, fold_ids: List[int]=[0], n_jobs: int=1, verbose: bool=False) -> None:

    cfg_0 = cfg.stage_0
    # cfg_1 = cfg.stage_1
    kfold_loader = KFoldLoader(root_dir=cfg_0.dataset.fold_dir)
    kfold_iter = kfold_loader.iter_folds(repeats=1)
    
    print(f'Beginning training of models with fold_ids: {fold_ids}')
    
    neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    
    results = []
    for worker_id, fold in enumerate(kfold_iter):
        args = (fold, copy.deepcopy(cfg), worker_id)
        
        with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name)+'-'+str(fold.fold_id), params=params):
            cfg = paleoai_main.neptune_train_single_fold(*args)
            
            log_config(cfg=cfg, verbose=verbose, neptune=neptune)

    print(paleoai_main.evaluate_predictions(results_dir=cfg.results_dir))
    return results



@hydra.main(config_path=Path(CONFIG_DIR,'config.yaml'))
def train(cfg : DictConfig) -> None:

    OmegaConf.set_struct(cfg, False)
    cfg = restore_or_initialize_experiment(cfg, restore_last=cfg.experiment.restore_last, prefix='log_dir__', verbose=0)


    # neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    # params=OmegaConf.to_container(cfg)
    # with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name)+'-'+str(cfg.fold_id), params=params):
        # train_pyleaves_dataset(cfg)
    train_paleoai_dataset(cfg=cfg, fold_ids=list(range(10)), n_jobs=cfg.n_jobs, verbose=True)

if __name__=="__main__":

    train()