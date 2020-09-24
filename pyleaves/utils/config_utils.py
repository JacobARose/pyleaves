# @Author: Jacob A Rose
# @Date:   Wed, September 23rd, 2020, 11:00 pm
# @Email:  jacobrose@brown.edu
# @Filename: config_utils.py

import hashlib
import json

from hydra.experimental import compose, initialize_config_dir#initialize
import os
from omegaconf import OmegaConf, DictConfig

DEFAULT_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','pipelines','configs'))


def init_hydra_config(config_name: str="simplified_config", overrides: list=None, config_dir: str=None) -> DictConfig:
    """Helper function for initializing Hydra configs separatelly from the main function

    Example uses:
        Loading a separate config for a test dataset after training a model

    Args:
        config_name (str, optional): [description]. Defaults to "simplified_config".
        overrides (list, optional): [description]. Defaults to None.
        config_dir (str, optional): [description]. Defaults to None.

    Returns:
        DictConfig: [description]
    """    
    overrides = overrides or []
    config_dir = config_dir or DEFAULT_CONFIG_DIR

    with initialize_config_dir(config_dir=config_dir):
        config = compose(config_name=config_name, overrides=overrides)

    return config

def resolve_test_config(main_config, test_config) -> DictConfig:
    """
    Creates a new DictConfig containing all values from test_config's 'dataset' sub-config, placed in a global config built from values otherwise taken from main_config. The important thing to note is that any OmegaConf shell interpolations are resolved prior to merging, so if the main config has run_dirs named after its main dataset, these aren't altered when creating the new config with a test dataset.

    Args:
        main_config ([type]): [description]
        test_config ([type]): [description]

    Returns:
        DictConfig: [description]
    """    
    copied_test_cfg = OmegaConf.masked_copy(test_config, ["dataset"])
    resolved_main_cfg = OmegaConf.to_container(main_config, resolve=True)
    resolved_main_cfg = OmegaConf.create(resolved_main_cfg)
    merged_config = OmegaConf.merge(resolved_main_cfg, copied_test_cfg)
    
    return merged_config


def init_test_config(main_config, config_name: str="simplified_config", overrides: list=None, config_dir: str=None):
    """
    Helper function for initializing a Hydra config object for preparing a test dataset as a stage of another model training pipeline.

    Args:
        main_config ([type]): [description]
        config_name (str, optional): [description]. Defaults to "simplified_config".
        overrides (list, optional): [description]. Defaults to None.
        config_dir (str, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """    
    overrides = overrides or []

    test_cfg = init_hydra_config(config_name=config_name, overrides=overrides, config_dir=config_dir)
    resolved_test_cfg = resolve_test_config(main_config, test_cfg)
    return resolved_test_cfg

def init_Fossil_family_100_test_config(main_config):
    """
    Run this function to initialize all requisite values for a data config to load Fossil_family_100 for testing, while automatically passing key parameters to make sure it is prepared for the model specified in main_config. e.g. target_size comes from main_config, while fold_dir comes from the test_config.
    Args:
        main_config ([type]): [description]

    Returns:
        [type]: [description]
    """
    overrides=['dataset@dataset=Fossil_family_100_test',
                '~dataset.params.training.target_size',
                'dataset.params.training.augmentations=[]']

    return init_test_config(main_config, config_name="simplified_config", overrides=overrides, config_dir=DEFAULT_CONFIG_DIR)



    # for p in overrides:
    #     if not p.startswith('dataset@dataset'):
    #         overrides += ['dataset@dataset=Fossil_family_100_test']
    #     if 'dataset.params.training.target_size' not in p:
    #         overrides += ['~dataset.params.training.target_size']
    #     if 'dataset.params.training.augmentations' not in p:
    #         overrides += ['dataset.params.training.augmentations=[]']



def get_config_uuid(config: dict):
    """
    Helper function for generating a unique, repeatable id from a dictionary of config parameters.
    Useful for identifying whether an experiment has been run before or not.

    Parameters
    ----------
    config : dict
        Dictionary containing all relevant parameters to retrieve this experiment's data/results in the future

    Returns
    -------
    str
        Description of returned object.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()




# def load_cfg(yaml_filepath):
#     """
#     Load a YAML configuration file.
#     Parameters
#     ----------
#     yaml_filepath : str
#     Returns
#     -------
#     cfg : dict
#     """
#     # Read YAML experiment definition file
#     with open(yaml_filepath, 'r') as stream:
#         cfg = yaml.load(stream)
#     cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
#     return cfg
