'''
This is meant to be a basic script for launching 


python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py' stage_0.misc.use_tfrecords=False

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
CONFIG_DIR = str(Path(RESOURCES_DIR,'..','..','configs','hydra'))
import neptune


def train_paleoai_dataset(cfg : DictConfig, fold_ids: List[int]=[0], n_jobs: int=1, verbose: bool=False) -> None:

    # histories = []
    # def log_history(history: dict):
    #     try:
    #         histories.append(history)
    #     except:
    #         histories.append(None)


    cfg_0 = cfg.stage_0
    # cfg_1 = cfg.stage_1
    log_config(cfg=cfg, verbose=verbose, neptune=neptune)
    kfold_loader = KFoldLoader(root_dir=cfg_0.dataset.fold_dir)
    kfold_iter = kfold_loader.iter_folds(repeats=1)
    # histories = CUDA_ERROR_NOT_INITIALIZED: initialization error Parallel(n_jobs=n_jobs)(delayed(train_single_fold)(fold=fold, cfg=copy.deepcopy(cfg_0), gpu_device=gpus[i]) for i, fold in enumerate(kfold_iter) if i < n_jobs)
    histories = []
    print(f'Beginning training of models with fold_ids: {fold_ids}')
    # histories = perform_concurrent_tasks(train_single_fold,
    #                         tasks_to_do=((fold, copy.deepcopy(cfg_0), neptune, worker_id) for worker_id, fold in enumerate(kfold_iter)),
    #                         max_processes=n_jobs)
    # return histories
    pool = RunAsCUDASubprocess(num_gpus=cfg.num_gpus, memory_fraction=0.9)

    histories = []
    for worker_id, fold in enumerate(itertools.islice(kfold_iter, n_jobs)):
    # for task in itertools.islice(tasks_to_do, max_processes)
        print(worker_id)
        print(fold)
        histories.append(pool(train_single_fold)(fold, copy.deepcopy(cfg_0), neptune, worker_id))
        time.wait(10)
    return histories




    # pool = Pool(processes=n_jobs)
    # for i, fold in enumerate(kfold_iter):
    #     history = pool.apply_async(train_single_fold, 
    #                                 args=(fold, copy.deepcopy(cfg_0), neptune),
    #                                 callback = log_history)#, gpu_device=gpus[0])
    #     print(i)
    #     histories.append(history)
    # histories = [h.wait() for h in histories]
    # with Pool(processes=n_jobs) as pool:
    #     for i, fold in enumerate(kfold_iter):
    #         # if i in fold_ids:
    #         history = pool.apply_async(train_single_fold, 
    #                                     args=(fold, copy.deepcopy(cfg_0), neptune),
    #                                     callback = log_history)#, gpu_device=gpus[0])
    #         print(i)
    #         histories.append(history)
    #     histories = [h.wait() for h in histories]
    # import pdb;pdb.set_trace()
    # pool.join()

    # return histories





@hydra.main(config_path=Path(CONFIG_DIR,'Leaves-PNAS.yaml'))
def train(cfg : DictConfig) -> None:

    OmegaConf.set_struct(cfg, False)
    cfg = restore_or_initialize_experiment(cfg, restore_last=True, prefix='log_dir__', verbose=0)


    neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name)+'-'+str(cfg.fold_id), params=params):
        # train_pyleaves_dataset(cfg)
        train_paleoai_dataset(cfg=cfg, fold_ids=list(range(10)), n_jobs=cfg.n_jobs, verbose=True)

if __name__=="__main__":

    train()