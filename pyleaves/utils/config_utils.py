

import hashlib
import json


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




def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.
    Parameters
    ----------
    yaml_filepath : str
    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg
