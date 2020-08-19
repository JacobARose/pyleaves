# @Author: Jacob A Rose
# @Date:   Thu, July 30th 2020, 1:03 am
# @Email:  jacobrose@brown.edu
# @Filename: cross_validation_utils.py



"""

Based on repo wtfml:
__url__: https://github.com/abhishekkrthakur/wtfml/blob/master/wtfml/cross_validation/fold_generator.py
__author__: Abhishek Thakur
"""

import pandas as pd
import scipy as sp
from sklearn import model_selection


class FoldGenerator:
    def __init__(
        self,
        targets,
        num_splits=5,
        shuffle=True,
        task="binary_classification",
        random_state=None,
    ):
        self.task = task
        self.targets = targets
        if isinstance(self.targets, pd.DataFrame) or isinstance(
            self.targets, pd.Series
        ):
            self.targets = self.targets.values

        if len(self.targets.shape) == 1:
            self.targets = self.targets.reshape((-1, 1))
        self.num_splits = num_splits

        if self.task == "binary_classification":
            self.folds = model_selection.StratifiedKFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "multiclass_classification":
            self.folds = model_selection.StratifiedKFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "multilabel_classification":
            self.folds = model_selection.KFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "single_col_regression":
            self.folds = model_selection.KFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        elif self.task == "multi_col_regression":
            self.folds = model_selection.KFold(
                n_splits=self.num_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            raise Exception("Task not understood")

        self.splits = dict()
        for fold_, (trn, val) in enumerate(
            self.folds.split(self.targets, self.targets)
        ):
            self.splits[fold_] = dict()
            self.splits[fold_]["train_idx"] = trn
            self.splits[fold_]["valid_idx"] = val

    def get_fold(self, data, fold):
        if fold >= self.num_splits or fold < 0:
            raise Exception("Invalid fold number specified")
        if isinstance(data, pd.DataFrame):
            if self.targets.shape[1] == 1:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data.loc[self.splits[fold]["train_idx"], :],
                    data.loc[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"]].ravel(),
                    self.targets[self.splits[fold]["valid_idx"]].ravel(),
                )
            else:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data.loc[self.splits[fold]["train_idx"], :],
                    data.loc[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"], :],
                    self.targets[self.splits[fold]["valid_idx"], :],
                )

        elif isinstance(data, sp.sparse.coo.coo_matrix) or \
             isinstance(data, sp.sparse.csc.csc_matrix):
            if self.targets.shape[1] == 1:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"]],
                    data[self.splits[fold]["valid_idx"]],
                    self.targets[self.splits[fold]["train_idx"]].ravel(),
                    self.targets[self.splits[fold]["valid_idx"]].ravel(),
                )
            else:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"]],
                    data[self.splits[fold]["valid_idx"]],
                    self.targets[self.splits[fold]["train_idx"], :],
                    self.targets[self.splits[fold]["valid_idx"], :],
                )
        else:
            if self.targets.shape[1] == 1:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"], :],
                    data[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"]].ravel(),
                    self.targets[self.splits[fold]["valid_idx"]].ravel(),
                )
            else:
                return (
                    self.splits[fold]["train_idx"],
                    self.splits[fold]["valid_idx"],
                    data[self.splits[fold]["train_idx"], :],
                    data[self.splits[fold]["valid_idx"], :],
                    self.targets[self.splits[fold]["train_idx"], :],
                    self.targets[self.splits[fold]["valid_idx"], :],
                )
