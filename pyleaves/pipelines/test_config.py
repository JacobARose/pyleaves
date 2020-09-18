

import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pprint import pprint

@hydra.main(config_path='configs', config_name='simplified_config')
def main(config : DictConfig):

    date_format = '%Y-%m-%d_%H-%M-%S'
    config.misc.experiment_start_time = datetime.now().strftime(date_format)

    

    pprint(OmegaConf.to_container(config))
    # print(config.pretty())

    if config.debug:
        import pdb
        pdb.set_trace()

    #   print(HydraConfig.get().overrides.task)

if __name__=="__main__":
    main()