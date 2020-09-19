# @Author: Jacob A Rose
# @Date:   fri, September 18th 2020, 3:32 pm
# @Email:  jacobrose@brown.edu
# @Filename: experiment_utils.py

"""
This script is meant to define helper functions for initializiing a new experiment from a Hydra configuration yaml file, which includes creating or cleaning up directories

"""

from datetime import datetime
from pprint import pprint
from pyleaves.utils import ensure_dir_exists
import os
import shutil
from omegaconf import DictConfig
    
def clean_experiment_tree(config: DictConfig):
    if not os.path.isdir(config.misc.experiment_dir):
        print(f'Attempted to clean nonexistent experiment directory at {config.misc.experiment_dir}. Continuing without action.')
        return
    print('Cleaning experiment file tree from root:\n',config.misc.experiment_dir)
    shutil.rmtree(config.misc.experiment_dir)
    assert not os.path.isdir(config.misc.experiment_dir)
    
def cleanup_tfrecords(config: DictConfig):
    if not os.path.isdir(config.run_dirs.tfrecord_dir):
        print(f'Attempted to clean nonexistent tfrecord directory at {config.run_dirs.tfrecord_dir}. Continuing without action.')
        return
    print('Cleaning up tfrecord files located at:\n',config.run_dirs.tfrecord_dir)
    shutil.rmtree(config.run_dirs.tfrecord_dir)
    assert not os.path.isdir(config.run_dirs.tfrecord_dir)


def resolve_config_interpolations(config: DictConfig) -> dict:
    """Recursively walk through a DictConfig object an convert any values that are also of type DictConfig to a regular dict.

    This is useful for displaying the true contents of a DictConfig that contains value interpolations, since by default they display the raw interpolation instead of its resolved value.

    Args:
        config (DictConfig): [description]

    Returns:
        dict: [description]
    """    
    
    pretty_config = {}
    for k,v in config.items():
        if isinstance(v,DictConfig):
            pretty_config[k] = resolve_config_interpolations(v)
        else:
            pretty_config[k] = v
    return pretty_config

def print_config(config):
    pprint(resolve_config_interpolations(config))


def recursively_instantiate_dirs(config: DictConfig, dir_suffix: str='_dir', verbose=False):
    """Walk through a DictConfig, searching for any keys that end in dir_suffix and creating the directory if it doesn't already exist. Will recursively search for keys if any value is itself dict-like.

    Args:
        config (DictConfig): [description]
        dir_suffix (str, optional): [description]. Defaults to '_dir'.
        verbose (bool, optional): [description]. Defaults to False.
    """    
    for k,v in config.items():
        if isinstance(v,(DictConfig,dict)):
            recursively_instantiate_dirs(config=v, dir_suffix=dir_suffix, verbose=verbose)
        if k.endswith(dir_suffix):
            if verbose:
                if os.path.exists(v):
                    print(f'Using existing directory: {v}')
                else:
                    print(f'Creating new directory: {v}')
            ensure_dir_exists(v)


def initialize_experiment(config: DictConfig, restore_last: bool=True, restore_tfrecords: bool=True, verbose=True):
    date_format = '%Y-%m-%d_%H-%M-%S'

    config.misc.experiment_start_time = datetime.now().strftime(date_format)

    if not restore_last:
        clean_experiment_tree(config)
        
    if not restore_tfrecords:
        cleanup_tfrecords(config)

    recursively_instantiate_dirs(config, dir_suffix='_dir', verbose=verbose)

    if verbose:
        print('='*40)
        print('Initializing experiment with the following configuration:')
        pprint(resolve_config_interpolations(config))
        print('='*40)
    
    return config