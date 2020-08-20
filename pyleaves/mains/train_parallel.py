'''
This is meant to be a basic script for launching 


python '/home/jacob/projects/pyleaves/pyleaves/mains/train_parallel.py' stage_0.misc.use_tfrecords=False

'''

import hydra
import neptune
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import os

from pyleaves.mains.paleoai_main import restore_or_initialize_experiment, train_paleoai_dataset
from pyleaves import RESOURCES_DIR
CONFIG_DIR = str(Path(RESOURCES_DIR,'..','..','configs','hydra'))

@hydra.main(config_path=Path(CONFIG_DIR,'Leaves-PNAS.yaml'))
def train(cfg : DictConfig) -> None:

    OmegaConf.set_struct(cfg, False)
    cfg = restore_or_initialize_experiment(cfg, restore_last=True, prefix='log_dir__', verbose=2)

    neptune.init(project_qualified_name=cfg.experiment.neptune_project_name)
    params=OmegaConf.to_container(cfg)
    with neptune.create_experiment(name=cfg.experiment.experiment_name+'-'+str(cfg.stage_0.dataset.dataset_name), params=params):
        # train_pyleaves_dataset(cfg)

        train_paleoai_dataset(cfg=cfg, n_jobs=1, verbose=True)

