

import hydra
from omegaconf import DictConfig


@hydra.main(config_path='configs', config_name='config')
def main(config : DictConfig):

    print(config.pretty())