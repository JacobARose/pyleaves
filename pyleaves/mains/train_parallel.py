'''
This is meant to be a basic script for launching 


python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py' stage_0.misc.use_tfrecords=False

'''

import hydra
from multiprocessing import Pool
import numpy as np
import neptune
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import os
from typing import List
import copy
from paleoai_data.utils.kfold_cross_validation import KFoldLoader
from pyleaves.mains.paleoai_main import restore_or_initialize_experiment, train_single_fold, log_config#, train_paleoai_dataset
from pyleaves import RESOURCES_DIR
CONFIG_DIR = str(Path(RESOURCES_DIR,'..','..','configs','hydra'))



def train_paleoai_dataset(cfg : DictConfig, fold_ids: List[int]=[0], n_jobs: int=1, verbose: bool=False) -> None:

    histories = []
    def log_history(history: dict):
        try:
            histories.append(history)
        except:
            histories.append(None)


    cfg_0 = cfg.stage_0
    # cfg_1 = cfg.stage_1
    log_config(cfg=cfg, verbose=verbose)
    kfold_loader = KFoldLoader(root_dir=cfg_0.dataset.fold_dir)
    kfold_iter = kfold_loader.iter_folds(repeats=1)
    # histories = Parallel(n_jobs=n_jobs)(delayed(train_single_fold)(fold=fold, cfg=copy.deepcopy(cfg_0), gpu_device=gpus[i]) for i, fold in enumerate(kfold_iter) if i < n_jobs)

    print(f'Beginning training of models with fold_ids: {fold_ids}')
    with Pool(processes=n_jobs) as pool:
        for i, fold in enumerate(kfold_iter):
            # if i in fold_ids:
            history = pool.apply_async(train_single_fold, 
                                        args=(fold, copy.deepcopy(cfg_0)),
                                        callback = log_history)#, gpu_device=gpus[0])
            print(i)
                # fold_ids.pop(np.where(fold_ids==i))
            # if len(fold_ids)==0:
                # break
    pool.join()

    return history





@hydra.main(config_path=Path(CONFIG_DIR,'Leaves-PNAS.yaml'))
def train(cfg : DictConfig) -> None:

    OmegaConf.set_struct(cfg, False)
    cfg = restore_or_initialize_experiment(cfg, restore_last=True, prefix='log_dir__', verbose=0)

    neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name)+'-'+str(cfg.fold_id), params=params):
        # train_pyleaves_dataset(cfg)
        # [cfg.fold_id]
        train_paleoai_dataset(cfg=cfg, fold_ids=list(range(10)), n_jobs=1, verbose=True)

if __name__=="__main__":

    train()