# @Author: Jacob A Rose
# @Date:   Tue, April 28th 2020, 4:17 pm
# @Email:  jacobrose@brown.edu
# @Filename: csv_logger.py

'''
Logging utilities for managing all kinds of csv logs
'''


import json
# import mlflow
# import mlflow.tensorflow

import loguru
import os

from pyleaves.utils.csv_utils import save_csv_data, load_csv_data
from pyleaves.utils import ensure_dir_exists

# from pyleaves.utils.mlflow_utils import mlflow_log_params_dict, mlflow_log_history, mlflow_log_best_history




class CSVLogger:

    def __init__(self, config, csv_dir):

        self.config = config
        self.logger = loguru.logger
        # self.logger.add()

        self.csv_dir = csv_dir
        ensure_dir_exists(self.csv_dir)
        self.log_file = os.path.join(csv_dir,'log_file.csv')
        self.handlers = [self.logger.add(open(self.log_file,'a'), level="INFO")]

    def log_dict(self, data_dict, filepath=None):
        if not filepath:
            filepath = self.log_file
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.csv_dir, filepath)

        keys, values = [], []
        for k,v in data_dict.items():
            keys.append(k)
            values.append(v)
        assert len(keys) == len(values)
        assert len(keys)>0
        save_csv_data(x=keys, y=values, filepath=filepath)
        self.logger.info(json.dumps(data_dict, indent='  '))

        return filepath

    # def autolog(self):
    #     mlflow.tensorflow.autolog()
    #
    # def log_metrics(history : dict, log_name=''):
    #     for k, v in history.items():
    #         mlflow.log_metric(log_name+k, v)
    #     self.logger.info(json.dumps(history, indent='  '))
    #
    # def log_params_dict(self, params_dict: dict, prefix='', sep=',', max_recursion=3):
    #     mlflow_log_params_dict(params_dict=params_dict, prefix=prefix, sep=sep, max_recursion=max_recursion)
    #     self.logger.info(json.dumps(params_dict, indent='  '))
    #
    # def log_history(self, history, history_name=''):
    #     mlflow_log_history(history, history_name=history_name)
    #     self.logger.info(json.dumps(history.history, indent='  '))



        # , mlflow_log_best_history
