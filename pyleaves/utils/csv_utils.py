# @Author: Jacob A Rose
# @Date:   Tue, March 31st 2020, 12:36 am
# @Email:  jacobrose@brown.edu
# @Filename: csv_utils.py


'''

Reference (3/13/20): Experiment directory layout for CSV experiments

directory name(integer indicating tree depth): how it relates to input parameters

root dir (0): experiment_root_dir
    e.g. r'/media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments'
       or
         r'/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments'

run_dir (1): experiment_root_dir + '/' + run_name
    e.g. r'Leaves'
       or
         r'PNAS_Leaves'

dataset/domain dir (2):
    For single-domain experiments, just run_dir + '/' + dataset_name
        e.g. 'PNAS'

    For 2-domain experiments, run_dir + '/' + domain + '_' + dataset_name
        e.g. 'source_PNAS', 'target_Leaves'

subset (3):
    Level (2) contains up to 5 CSV files (considered level (3)), up to 3 of which contain data necessary to load images for train, val and/or test subsets.
        e.g. train_data.csv, val_data.csv, test_data.csv, meta.csv, label_mappings.csv





'''


import dataset
import json
import numpy as np
import os
import pandas as pd
from stuf import stuf

from pyleaves.leavesdb import db_query
from pyleaves.utils import ensure_dir_exists
from pyleaves.data_pipeline.preprocessing import filter_low_count_labels, LabelEncoder #, get_class_counts
from pyleaves.leavesdb.tf_utils.tf_utils import train_val_test_split, get_data_splits_metadata



#TODO Write test for save and load csv

def save_csv_data(x, y, filepath, other_data={}):
    data = pd.DataFrame({'x':x,'y':y, **other_data},index=list(range(len(y))))
    data.to_csv(filepath)
    print(f'saved {len(y)} samples to {filepath}')

def load_csv_data(filepath):
    data = pd.read_csv(filepath, usecols=['x','y']) #drop_index=True)
    return data




#TODO Remove most of the functions below this line


















def load_dataset(local_db, dataset_name, x_col='path', y_col='family'):
    db = dataset.connect(f"sqlite:///{local_db}", row_type=stuf)
    data = pd.DataFrame(db_query.load_data(db=db, x_col=x_col, y_col=y_col, dataset=dataset_name))
    return data


def preprocess_data(data_df, encoder, validation_splits, output_dir, threshold=10, merge_new_labels=True, other_data_keys=[]):
    data_df = filter_low_count_labels(data_df, threshold=threshold, verbose=False)

    if merge_new_labels:
        encoder.merge_labels(labels=list(data_df['family']))
    encoder.save_labels(os.path.join(output_dir, 'label_mappings.csv'))

    data_df = encoder.filter(data_df, text_label_col='family')

    x = data_df['path'].values.reshape((-1,1))
    y = np.array(encoder.transform(data_df['family']))

    data_splits = train_val_test_split(x, y, val_size=validation_splits['val_size'], test_size=validation_splits['test_size'])
    metadata_splits = get_data_splits_metadata(data_splits, data_df, encoder=encoder, verbose=False)

    return data_splits, metadata_splits


def save_paths_w_labels(x, y, encoder, data_dir, subset, other_data={}):
    data = pd.DataFrame({'x':x,'y':y, **other_data},index=list(range(len(y))))
    filepath = os.path.join(data_dir, subset + '_data.csv')
    data.to_csv(filepath)

    print(f'saved {os.path.basename(data_dir)} dataset {subset} subset to {filepath}')


def save_label_maps(encoder, data_dir):
    out_path = os.path.join(data_dir, 'label_mappings.csv')
    encoder.save_labels(out_path)
    print(f'saved label maps to {out_path}')


def save_metadata(metadata, data_dir):
    metadata = pd.DataFrame(list(metadata.values()),index=list(metadata.keys()))
    metadata.to_csv(os.path.join(data_dir,'meta.csv'))
########################################################################
########################################################################
########################################################################
def process_and_save_dataset(data_df, name, encoder, validation_splits, experiment_dir, merge_new_labels=True, other_data_keys=[]):
    '''
    Utility function for processing and saving data provided as a dataframe

    other_data_keys: list
        list of str indicating keys of additional columns to save alongside x, y (e.g. 'dataset')
    '''
    data_dir = os.path.join(experiment_dir,name)
    ensure_dir_exists(data_dir)

    data_splits, metadata_splits = preprocess_data(data_df, encoder, validation_splits=validation_splits, output_dir=data_dir, threshold=10, merge_new_labels=merge_new_labels, other_data_keys=other_data_keys)
    metadata_splits.pop('label_map')
    for subset, d in data_splits.items():
        if len(d['path'])==0:
            continue
        x, y = list(d['path'].flatten()), list(d['label'])
        other_data = {k:d[k] for k in other_data_keys}
        save_paths_w_labels(x, y, encoder, data_dir, subset, other_data=other_data)

    save_metadata(metadata_splits, data_dir)
    save_label_maps(encoder, data_dir)


def process_and_save_singledomain_datasets(data_dict: dict, dataset_names: list, validation_splits: dict, output_root_dir: str, merge_new_labels: bool = True):
    '''
    Generate CSV datasets for single domain experiment, one for each individual dataset
    '''
    for dataset_name in dataset_names:
        encoder = LabelEncoder()
        experiment_dir = os.path.join(output_root_dir, dataset_name)
        process_and_save_dataset(data_dict[dataset_name],
                                     name=dataset_name,
                                     encoder=encoder,
                                     validation_splits=validation_splits,
                                     experiment_dir=experiment_dir,
                                     merge_new_labels=merge_new_labels)


def process_and_save_multidomain_datasets(data_dict, dataset_name_pairs, validation_splits, output_root_dir):

    encoder = LabelEncoder()
    domain_names = list(validation_splits.keys())

    for dataset_name_pair in dataset_name_pairs:
        encoder = LabelEncoder()
        experiment = '_'.join(dataset_name_pair)
        experiment_dir = os.path.join(output_root_dir, experiment)

        for i, dataset_name in enumerate(dataset_name_pair):
            domain = domain_names[i]
            if domain=='source':
                merge_new_labels=True
            else:
                merge_new_labels=False

            process_and_save_dataset(data_dict[dataset_name],
                                     '_'.join([domain,dataset_name]),
                                     encoder=encoder,
                                     validation_splits=validation_splits[domain],
                                     experiment_dir=experiment_dir,
                                     merge_new_labels=merge_new_labels)


def process_and_save_multidataset_singledomain_datasets(data_dict: dict, dataset_names: list, validation_splits: dict, output_root_dir: str, merge_new_labels: bool = True):
    '''
    Generate CSV datasets for single domain experiment, but for each pair of datasets to be merged
    Arguments:
        data_dict: dict
        dataset_names: list
            e.g. ['PNAS','Fossil','Leaves']. Pairs will be created within function.
        validation_splits: dict
        output_root_dir: str
        merge_new_labels: bool = True

    '''
    for i, dataset_1 in enumerate(dataset_names):
        for j, dataset_2 in enumerate(dataset_names):
            if j==i:
                continue
            dataset_name = '+'.join([dataset_1,dataset_2])

            encoder = LabelEncoder()
            experiment_dir = os.path.join(output_root_dir, dataset_name)

            input_data = pd.concat([data_dict[dataset_1], data_dict[dataset_2]])
            process_and_save_dataset(input_data,
                                     name=dataset_name,
                                     encoder=encoder,
                                     validation_splits=validation_splits,
                                     experiment_dir=experiment_dir,
                                     merge_new_labels=merge_new_labels,
                                     other_data_keys=['dataset'])















# def load_csv_data(filepath):
#     data = pd.read_csv(filepath, drop_index=True)
#     return data


def get_dataset_filepath(experiment_root_dir, run_name, dataset_name: str=None, domain: str=None, subset=None):

    filetree = os.path.join(experiment_root_dir, run_name)
    run_contents = os.listdir(filetree)

    for item in run_contents:
        if (str(dataset_name) in item) or (str(domain) in item):
            filetree = os.path.join(filetree,item)
            break

    return {
               'label_mappings':os.path.join(filetree,'label_mappings.csv'),
               'meta':os.path.join(filetree,'meta.csv'),
                subset+'_data':os.path.join(filetree,subset+'_data.csv')
              }




def gather_subset_data(experiment_root_dir, run, domain, subset, return_type='records'):
    data = get_dataset_filepath(experiment_root_dir,
                                     run_name=run,
                                     domain=domain,
                                     subset=subset)
    if return_type=='nested_dict':
        return {subset:data}
    elif return_type=='records':
        return {'subset':subset,
               **data}


def gather_domain_data(experiment_root_dir,
                       run,
                       domain,
                       return_type='records',
                       collect_data_subsets=True):
    '''
    collect_data_subsets, bool: defualt=True
        if True, return domain data in the format
        [
          {
            "domain": "PNAS",
            "dataset_name": "PNAS",
            "data": {
              "test": ".../single-domain_experiments/PNAS/PNAS/test_data.csv",
              "val": ".../single-domain_experiments/PNAS/PNAS/val_data.csv",
              "train": ".../single-domain_experiments/PNAS/PNAS/train_data.csv"
            },
            "meta": ".../single-domain_experiments/PNAS/PNAS/meta.csv",
            "label_mappings": ".../single-domain_experiments/PNAS/PNAS/label_mappings.csv"
          }
        ]
        if False, return the above list of 1 dict as a list of 3 dicts, where each corresponds to its respective subset.

        The key point here is:
        *set this to False if you want separate meta files and label mappings for individual subsets*
        *set it to True if you plan to have a single meta file and a single label map for all subsets in this domain.*

    '''



    if return_type=='nested_dict':
        domain_data = {domain:{}}
    if return_type=='records':
        domain_data = []
    run_dir = os.path.join(experiment_root_dir, run)
    for domain_name in os.listdir(run_dir):

        blacklist = ['.ipynb_checkpoints', 'frozen-data-config.json', 'frozen_data_config.json']
        if domain_name in blacklist:
            continue

        domain_name_parts = domain_name.split('_')
        if (domain_name_parts[0] != domain) | ('.ipynb' in domain_name_parts):
            continue
        if len(domain_name_parts)==2:
            domain, dataset_name = domain_name_parts
        elif len(domain_name_parts)==1:
            domain, dataset_name = domain_name_parts*2
        else:
            domain = domain_name
            print('domain_name = ',domain_name)
        data_dir = os.path.join(run_dir, domain_name)

#         import pdb; pdb.set_trace()
        if collect_data_subsets:
            data_output = {'data':{}, 'subsets':[]}

            for f in os.listdir(data_dir):
                if not f.endswith('_data.csv'):
                    f_key = f.replace('.csv','')
                    data_output[f_key] = os.path.join(data_dir,f)
                    continue

                subset = f.replace('_data.csv','')
#                 data_output['subsets'].update(subset)
                subset_data = gather_subset_data(experiment_root_dir, run, domain, subset, return_type=return_type)
                for k, v in subset_data.items():
                    if k.endswith('_data'):
                        k = k.replace('_data','')
                        data_output['data'].update({k:v})

            for s in ['train','val','test']: # add subset names in order if they are included in this domain
                if s in data_output['data'].keys():
                    data_output['subsets'].append(s)

            if return_type=='nested_dict':
                domain_data[domain].update(data_output) #subset_data)
            elif return_type=='records':
                domain_data.append({'domain':domain,
                                    'dataset_name':dataset_name,
                                    **data_output})


        else:
            for f in os.listdir(data_dir):
                if not f.endswith('_data.csv'):
                    continue

                subset = f.replace('_data.csv','')
                subset_data = gather_subset_data(experiment_root_dir, run, domain, subset, return_type=return_type)

                if return_type=='nested_dict':
                    domain_data[domain].update(subset_data)
                elif return_type=='records':
                    domain_data.append({'domain':domain,
                                        'dataset_name':dataset_name,
                                        **subset_data})

    return domain_data



def gather_run_data(experiment_root_dir, run, return_type='records', collect_data_subsets=True):
#     import pdb; pdb.set_trace()
    if return_type=='nested_dict':
        run_data = {run:{}}
    if return_type=='records':
        run_data = []
    run_dir = os.path.join(experiment_root_dir, run)
    domain_names = os.listdir(run_dir)
    for domain_name in domain_names:
        domain = domain_name.split('_')[0]
        data_dir = os.path.join(run_dir, domain_name)
        domain_data = gather_domain_data(experiment_root_dir,
                                         run,
                                         domain,
                                         return_type=return_type,
                                         collect_data_subsets=collect_data_subsets)
        if return_type=='nested_dict':
            run_data[run].update(domain_data)
        elif return_type=='records':
            for subset_data in domain_data:
                run_data.append({'run':run,
                                 **subset_data})

    return run_data



def gather_experiment_data(experiment_root_dir,
                           return_type='records',
                           collect_data_subsets=True):
    experiment_name = os.path.basename(experiment_root_dir)
    if return_type=='nested_dict':
        experiment_data = {experiment_name:{}}
    if return_type=='records':
        experiment_data = []

    runs = os.listdir(experiment_root_dir)
    for run in runs:
        run_dir = os.path.join(experiment_root_dir, run)
        run_data = gather_run_data(experiment_root_dir,
                                   run,
                                   return_type=return_type,
                                   collect_data_subsets=collect_data_subsets)
        if return_type=='nested_dict':
            experiment_data[experiment_name].update(run_data)
        elif return_type=='records':
            for domain_data in run_data:
                experiment_data.append({'experiment_name':experiment_name,
                                         **domain_data})

    return experiment_data









if __name__=='__main__':

    experiment_root_dir = r'/media/data_cifs/jacob/Fossil_Project/replication_data/single-domain_experiments'

    experiment_data = gather_experiment_data(experiment_root_dir, return_type='nested_dict')
#   print(json.dumps(experiment_data,indent=4))
    experiment_records = gather_experiment_data(experiment_root_dir, return_type='records')
    for r in experiment_records:
        print(json.dumps(r, indent=4))

####################################################################
####################################################################

    experiment_root_dir = r'/media/data_cifs/jacob/Fossil_Project/replication_data/2-domain_experiments'
    experiment_data = gather_experiment_data(experiment_root_dir, return_type='nested_dict')
    experiment_records = gather_experiment_data(experiment_root_dir, return_type='records')

    for r in experiment_records:
        print(json.dumps(r, indent=4))



#     experiment_records = list(pd.json_normalize(experiment_data, max_level=1).values.ravel())
#     print(json.dumps(experiment_records[0], indent=4))











#     d = gather_domain_data(experiment_root_dir, run='PNAS', domain='PNAS')

#     d = gather_domain_data(experiment_root_dir, run='PNAS', domain='PNAS', return_type='records')

#     r = gather_run_data(experiment_root_dir, run='PNAS', return_type='records')
