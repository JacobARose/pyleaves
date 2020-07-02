# @Author: Jacob A Rose
# @Date:   Mon, April 20th 2020, 3:55 am
# @Email:  jacobrose@brown.edu
# @Filename: mlflow_logger.py

import json
import mlflow
import mlflow.tensorflow

import loguru
import os

from pyleaves.utils import ensure_dir_exists
from pyleaves.utils.mlflow_utils import mlflow_log_params_dict, mlflow_log_history, mlflow_log_best_history




class MLFlowLogger:

    def __init__(self, config):

        self.config = config
        self.logger = loguru.logger
        # self.logger.add()

        self.tracking_dir = config.logger.mlflow_tracking_dir
        self.log_file = config.logger.program_log_file
        ensure_dir_exists(os.path.dirname(self.log_file))
        self.handlers = [self.logger.add(open(self.log_file,'w'), level="DEBUG")]

        ensure_dir_exists(self.tracking_dir)
        mlflow.set_tracking_uri(self.tracking_dir)
        print(mlflow.tracking.get_tracking_uri())
        mlflow.set_experiment(config.experiment_type)

    def autolog(self):
        mlflow.tensorflow.autolog()

    def log_metrics(history : dict, log_name=''):
        for k,v in history.items():
            history[k] = normalize_list_floats(log_item=v)

        for k, v in history.items():
            mlflow.log_metric(log_name+k, v)
        self.logger.info(json.dumps(history, indent='  '))

    def log_params_dict(self, params_dict: dict, prefix='', sep=',', max_recursion=3):
        mlflow_log_params_dict(params_dict=params_dict, prefix=prefix, sep=sep, max_recursion=max_recursion)
        self.logger.info(json.dumps(params_dict, indent='  '))

    def log_history(self, history, history_name=''):

        for k,v in history.history.items():
            history.history[k] = normalize_list_floats(log_item=v)

        mlflow_log_history(history, history_name=history_name)
        self.logger.info(json.dumps(history.history, indent='  '))



def normalize_list_floats(log_item : list):
    '''
    Converts all of the contents of a list to np.float64 to ensure they are JSON serializable.
    See: https://ellisvalentiner.com/post/serializing-numpyfloat32-json/
    '''
    for i in range(len(log_item)):
        log_item[i] *= 1.0
    return log_item
