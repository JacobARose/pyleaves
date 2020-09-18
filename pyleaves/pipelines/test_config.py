

import hydra
from omegaconf import DictConfig


@hydra.main(config_path='configs', config_name='config')
def main(config : DictConfig):

    print(config.pretty())

    if config.debug:
        import pdb
        pdb.set_trace()


if __name__=="__main__":
    main()