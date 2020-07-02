
# configs

### This module in the pyleaves library is used to setup the configuration for all experiments.

- config_v2.py is the most up-to-date config file (as of 4/14/20), but for backwards compatibility we currently use __init__.py to define config = config_v1 automatically. Future updates will deprecate this functionality and use config_v2 as config, but for now when creating new scripts, user should explicitly import config_v2.

- EXPERIMENT_TYPES in config_v2 uses a shorthand for describing dataset splits and order of training/eval.

## Experiment key:

    -> A,B refer to dataset 1 and 2, respectively (in order of intended use)
    -> Experiments can have 1 or more datasets, with the first always being referenced by A
    -> A+B means A and B were combined before doing any splitting or label encoding
    -> A or B followed by a sequence of underscore-separated (' _ ') words in the set {train, val, test} indicates that dataset's train, val, and/or test set, respectively
    -> A dash ('-') indicates the end of one stage and transition to another
        e.g. 'A_train_val-B_train_val_test'
