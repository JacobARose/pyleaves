# @Author: Jacob A Rose
# @Date:   Wed, April 15th 2020, 10:39 pm
# @Email:  jacobrose@brown.edu
# @Filename: create_experiments.py

'''

This is intended to be primarily a command line script for coordinating the creation of all requisite files for each experiment definition.

'''
from more_itertools import unzip
from functools import partial
import pandas as pd
import pyleaves
# from pyleaves import leavesdb
from pyleaves.configs.config_v2 import BaseConfig
from pyleaves.leavesdb import experiments_db
from pyleaves.datasets import leaves_dataset, fossil_dataset, pnas_dataset, base_dataset
from pyleaves.leavesdb.experiments_db import DataBase, Table, TFRecordsTable, EXPERIMENTS_SCHEMA, TFRecordItem, select_by_col
from pyleaves.utils.csv_utils import save_csv_data, load_csv_data
from pyleaves.utils.utils import ensure_dir_exists
from pyleaves.leavesdb.tf_utils.create_tfrecords import save_tfrecords
from pyleaves.loggers.csv_logger import CSVLogger
import random
from stuf import stuf
import time


EXPERIMENT_TYPES = ['A_train_val_test',
                    'A+B_train_val_test',
                    'A_train_val-B_train_val_test',
                    'A+B_leave_one_out']


# experiments_db.create_db()
tables = experiments_db.get_db_contents()


def create_experiment(experiment_name='A_train_val_test', src_db=pyleaves.DATABASE_PATH, include_runs=['all']):
    """
    Main function for calling the correct experiment creation function based on user provided experiment_name

    Parameters
    ----------
    experiment_name : str
        Str name of experiment type to create. Should correspond to one of the pre-defined EXPERIMENT_TYPES defined in create_experiments
    src_db : str
        Path to the data source database, defaults to the leavesdb default db.
    include_runs : list
        Defaults to 'all' to include all runs defined in experiment_db corresponding to given experiment type. Otherwise, user should
        provide a list of strings corresponding to a subset of these run_ids.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> create_experiment(experiment_name='A_train_val_test', src_db=pyleaves.DATABASE_PATH, include_runs=['1100','1200'])

    """

    if experiment_name == 'A_train_val_test':
        print('CREATING EXPERIMENT: <A_train_val_test>')
        create_experiment__A_train_val_test(src_db=src_db, incl=include_runs)

    elif experiment_name == 'A+B_train_val_test':
        print('CREATING EXPERIMENT: <A+B_train_val_test>')
        create_experiment__AB_train_val_test(src_db=src_db, incl=include_runs)

    elif experiment_name == 'A_train_val-B_train_val_test':
        raise NotImplementedError
        # print('CREATING EXPERIMENT: <A_train_val-B_train_val_test>')
        # create_experiment__A_train_val__B_train_val_test(src_db=src_db, incl=include_runs)

    elif experiment_name == 'A+B_leave_one_out':
        raise NotImplementedError
        # print('CREATING EXPERIMENT: <A+B_leave_one_out>')
        # create_experiment__AB_leave_one_out(src_db=src_db, incl=include_runs)












def create_experiment__A_train_val_test(src_db=pyleaves.DATABASE_PATH, include_runs=['all']):
    start = time.time()
    experiment_view = select_by_col(table=tables['runs'],column='experiment_type',value='A_train_val_test')
    datasets = {
            'PNAS': pnas_dataset.PNASDataset(src_db=src_db),
            'Leaves': leaves_dataset.LeavesDataset(src_db=src_db),
            'Fossil': fossil_dataset.FossilDataset(src_db=src_db)
            }

    logger = stuf({})
    for i, run in experiment_view.iterrows():
        run_config = stuf(run.to_dict())

        if include_runs != ['all'] and (run_config['run_id'] not in include_runs):
            print(f"SKIPPING run with run_id={run_config['run_id']}")
            continue

        print('BEGINNING ', run)
        config = BaseConfig().parse(namespace=run_config)
        name = config.dataset_A
        logger[name] = stuf({})

        tfrecords_table = TFRecordsTable(db_path=config.user_config.experiments_db)

        data_config_A = config.data_config.dataset_A
        num_shards = data_config_A.num_shards
        resolution = data_config_A.resolution

        csv_logger = CSVLogger(config, csv_dir=data_config_A.csv_dir)

        data = datasets[name]
        encoder = base_dataset.LabelEncoder(data.data.family)
        ensure_dir_exists(data_config_A.csv_dir)
        num_classes = data_config_A.num_classes = encoder.num_classes

        processed = base_dataset.preprocess_data(data, encoder, data_config_A)
        for subset in processed.keys():
            Item = partial(TFRecordItem,**{
                                'run_id':run_config['run_id'],
                                'experiment_type':run_config['experiment_type'],
                                'file_group':subset,
                                'dataset_stage':'dataset_A',
                                'dataset_name':name,
                                'resolution':resolution,
                                'num_channels':3,
                                'num_classes':num_classes,
                                'num_shards':num_shards
                                })

            random.shuffle(processed[subset])

            x, y = [list(i) for i in unzip(processed[subset])]
            save_csv_data(x, y, filepath = data_config_A.csv_data[subset])
            encoder.save_config(data_config_A.label_dir)

            class_counts = base_dataset.calculate_class_counts(y_data=y)
            class_weights = base_dataset.calculate_class_weights(y_data=y)

            class_counts_fpath = csv_logger.log_dict(class_counts, filepath=f'{subset}_class_counts.csv')
            class_weights_fpath = csv_logger.log_dict(class_weights, filepath=f'{subset}_class_weights.csv')

            print(f'logged class counts and weights to {class_counts_fpath} and {class_weights_fpath}, respectively')

            logger[name][subset] = stuf({'start':time.time()})
            file_log = save_tfrecords(data=processed[subset],
                           output_dir=data_config_A.tfrecord_dir,
                           file_prefix=subset,
                           target_size=(resolution, resolution),
                           num_channels=3,
                           num_classes=encoder.num_classes,
                           num_shards=num_shards,
                           TFRecordItem_factory=Item,
                           tfrecords_table=tfrecords_table,
                           verbose=True)

            logger[name][subset].end = time.time()
            logger[name][subset].total = logger[name][subset].end - logger[name][subset].start
            print(name, subset, f'took {logger[name][subset].total:.2f} sec to collect/convert to TFRecords')
        print(f'Full experiment took a total of {time.time()-start}')

##############################################################################################################################################

def create_experiment__AB_train_val_test(src_db=pyleaves.DATABASE_PATH, include_runs=['all']):
    start = time.time()
    experiment_view = select_by_col(table=tables['runs'],column='experiment_type',value='A+B_train_val_test')
    datasets = {
            'PNAS': pnas_dataset.PNASDataset(src_db=src_db),
            'Leaves': leaves_dataset.LeavesDataset(src_db=src_db),
            'Fossil': fossil_dataset.FossilDataset(src_db=src_db)
            }

    logger = stuf({})
    for i, run in experiment_view.iterrows():
        run_config = stuf(run.to_dict())

        if include_runs != ['all'] and (run_config['run_id'] not in include_runs):
            print(f"SKIPPING run with run_id={run_config['run_id']}")
            continue

        print('BEGINNING ', run)
        config = BaseConfig().parse(namespace=run_config)
        data_config_AB = config.data_config.dataset_AB
        log_name = data_config_AB.name
        logger[log_name] = stuf({})

        tfrecords_table = TFRecordsTable(db_path=config.user_config.experiments_db)
        num_shards = data_config_AB.num_shards
        resolution = data_config_AB.resolution

        names = config.datasets

        data_A = datasets[names[0]]
        data_B = datasets[names[1]]
        data_AB = data_A + data_B

        print('A:',data_A)
        print('B:',data_B)
        print('A+B:',data_AB)

        encoder = base_dataset.LabelEncoder(data_AB.data.family)
        ensure_dir_exists(data_config_AB.csv_dir)
        num_classes = data_config_AB.num_classes = encoder.num_classes

        processed = base_dataset.preprocess_data(data_AB, encoder, data_config_AB)

        print('A+B:',data_AB)


        for subset in processed.keys():
            Item = partial(TFRecordItem,**{
                                'run_id':run_config['run_id'],
                                'experiment_type':run_config['experiment_type'],
                                'file_group':subset,
                                'dataset_stage':'dataset_A+dataset_B',
                                'dataset_name':data_config_AB.name,
                                'resolution':resolution,
                                'num_channels':3,
                                'num_classes':num_classes,
                                'num_shards':num_shards
                                })

            random.shuffle(processed[subset])

            x, y = [list(i) for i in unzip(processed[subset])]
            save_csv_data(x, y, filepath = data_config_AB.csv_data[subset])
            encoder.save_config(data_config_AB.label_dir)

            logger[log_name][subset] = stuf({'start':time.time()})
            file_log = save_tfrecords(data=processed[subset],
                           output_dir=data_config_AB.tfrecord_dir,
                           file_prefix=subset,
                           target_size=(resolution, resolution),
                           num_channels=3,
                           num_classes=encoder.num_classes,
                           num_shards=num_shards,
                           TFRecordItem_factory=Item,
                           tfrecords_table=tfrecords_table,
                           verbose=True)

            logger[log_name][subset].end = time.time()
            logger[log_name][subset].total = logger[log_name][subset].end - logger[log_name][subset].start
            print(log_name, subset, f'took {logger[log_name][subset].total:.2f} sec to collect/convert to TFRecords')
        print(f'Full experiment took a total of {time.time()-start:.2f}')


##############################################################################################
##############################################################################################
































    logger = stuf({})
    for i, run in experiment_view.iterrows():
        run_config = stuf(run.to_dict())

        if include_runs != ['all'] and (run_config['run_id'] not in include_runs):
            print(f"SKIPPING run with run_id={run_config['run_id']}")
            continue

        print('BEGINNING ', run)
        config = BaseConfig().parse(namespace=run_config)
        data_config_A = config.data_config.dataset_A

        tfrecords_table = TFRecordsTable(db_path=config.user_config.experiments_db)
        num_shards = data_config_A.num_shards
        resolution = data_config_A.resolution

        name = config.dataset_A
        logger[name] = stuf({})
        data = datasets[name]
        encoder = base_dataset.LabelEncoder(data.data.family)
        ensure_dir_exists(data_config_A.csv_dir)
        num_classes = data_config_A.num_classes = encoder.num_classes

        processed = base_dataset.preprocess_data(data, encoder, data_config_A)
        for subset in processed.keys():
            Item = partial(TFRecordItem,**{
                                'run_id':run_config['run_id'],
                                'experiment_type':run_config['experiment_type'],
                                'file_group':subset,
                                'dataset_stage':'dataset_A',
                                'dataset_name':name,
                                'resolution':resolution,
                                'num_channels':3,
                                'num_classes':num_classes,
                                'num_shards':num_shards
                                })

            random.shuffle(processed[subset])

            x, y = [list(i) for i in unzip(processed[subset])]
            save_csv_data(x, y, filepath = data_config_A.csv_data[subset])
            encoder.save_config(data_config_A.label_dir)

            logger[name][subset] = stuf({'start':time.time()})
            file_log = save_tfrecords(data=processed[subset],
                           output_dir=data_config_A.tfrecord_dir,
                           file_prefix=subset,
                           target_size=(resolution, resolution),
                           num_channels=3,
                           num_classes=encoder.num_classes,
                           num_shards=num_shards,
                           TFRecordItem_factory=Item,
                           tfrecords_table=tfrecords_table,
                           verbose=True)

            logger[name][subset].end = time.time()
            logger[name][subset].total = logger[name][subset].end - logger[name][subset].start
            print(name, subset, f'took {logger[name][subset].total:.2f} sec to collect/convert to TFRecords')
        print(f'Full experiment took a total of {time.time()-start}')
