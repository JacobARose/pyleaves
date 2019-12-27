'''
Functions for creating, archiving, and reconstructing SQLite database as .db and .json files.

1. .db files are for backend data management, typically as part of preparing data for an experiment
2. .json files are for human data management and curation, convenient format for modifying database with new datasets or metadata


CURRENT: source json should be pyleaves/leavesdb/resources/full_dataset_frozen.json




Old format json: 
    Nested categories, single json file
    e.g. 
    { 'dataset1' : {'family1': 
                            'genus1':{ 
                                'specie1':{
                                    'paths': [path1.jpg,...],
                                             ...
                                          },
                                     ...    
                                     },
                    'family2':{{... , ...}}
    
    
    
New format json:
    Flat records, single or multiple json files
    e.g.
    {
  "count": 119084,
  "results": [
    {
      "id": 1,
      "species": "menziesii",
      "genus": "nothofagus",
      "path": "/media/data_cifs/sven2/leaves/sorted/PNAS_DataSource/PNAS_19/Split_S1/Res_768/Testing_files/Fagaceae/Fagaceae_Nothofagus_menziesii_10654 {WolfeUSGS} [1.96x].jpg",
      "family": "Fagaceae",
      "dataset": "PNAS"
    },
    {
      "id": 2,
      "species": "resinosa",
      "genus": "nothofagus",
      "path": "/media/data_cifs/sven2/leaves/sorted/PNAS_DataSource/PNAS_19/Split_S1/Res_768/Training_files/Fagaceae/Fagaceae_Nothofagus_resinosa_8535 {WolfeUSGS} [1.96x].jpg",
      "family": "Fagaceae",
      "dataset": "PNAS"
    },
    ...

KEY DETAILS:
    1. Old Usage 'specie' column renamed to 'species'. Will cause errors if attempting to add new json records to old formatted db file.

Example usage:

######################################
## one JSON -> one DB
or
## multiple JSON -> one DB

>> db = build_db_from_json(frozen_json_filepaths=[r'resources/full_dataset_frozen.json'], db_path=r'resources/leavesdb.db')
or
>> db = build_db_from_json(frozen_json_filepaths=[r'resources/Fossil_frozen.json',r'resources/Leaves_frozen.json'], db_path=r'resources/leavesdb.db')

######################################
## one DB -> one JSON

>> full_dataset_json_filepath = freeze_full_database(json_filepath='resources/full_dataset_frozen.json', db_path='resources/leavesdb.db')

######################################
## one DB -> multiple JSON
    # Creates one JSON file for each unique dataset for future use, 
    # potentially for creating new database with only a subset of all datasets

>> frozen_json_filepaths = freeze_db_by_dataset(db_path='resources/leavesdb.db', prefix='resources', freeze_key='dataset')

######################################
'''


import pandas as pd 
import dataset
import datafreeze
import json
from pyleaves.leavesdb.db_utils import load, flattenit, image_checker, TimeLogs
import cv2 
import os

PATH = r'resources/datasets.json' #/media/data_cifs/irodri15/data/processed/datasets.json'
# OUTPUT = 'sqlite:///resources/leavesdb.db'

def create_db(jsonpath=PATH, folder= 'resources'):
    '''
    #TODO DEPRECATE
    
    Function to create a db from a json file. check the structure of the Json.
    The function would look for the key 'paths' as a stop key. 
    Arguments:
        - Json file with the following structure: 
            file: 
            { 'dataset1' : {'family1': 
                                'genus1':{ 
                                    'specie1':{
                                        'paths': [path1.jpg,...],
                                                 ...
                                              },
                                     ...    
                                         },
                            'family2':{{... , ...}}
    Returns
    '''
    json_file = load(jsonpath)
    db_path = os.path.join(folder,'leavesdb.db')
    print(db_path)
    output = f'sqlite:///{db_path}'
    print(output)
    
    db = dataset.connect(output)
    db.begin()

    table = db['dataset']
    counter= 0 
    invalid_images=[]
    for data_set in json_file:
        res = {k:v for k, v in flattenit(json_file[data_set])}
        print(data_set)
        for key in res: 
            if 'paths' in key:
                names = key.split('_')[:-1]
                if len(names)==1:
                    continue 
                    print(names[0])
                    for p  in res[key]:
                        sample= dict(path=p,
                                      dataset=data_set,
                                      family=names[0],
                                      specie='nn',
                                      genus='nn'
                                      )
                        if image_checker:
                            table.insert(sample)
                            counter+=1
                        else: 
                            invalid_images.append([p,data_set,family,'nn','nn'])
                            

                else:   
                    print(names)
                    for p  in res[key]:
                        family = names[0]
                        if 'uncertain' in family:
                            family='uncertain'
                        if image_checker(p):
                            
                            sample= dict(path=p,
                                      dataset=data_set,
                                      family=names[0],
                                      specie=names[2],
                                      genus=names[1]
                                      )
                            table.insert(sample)
                            counter+=1
                        else: 
                            invalid_images.append([p,data_set,names[0],names[2],names[1]])
                        if counter%1000==0:
                            print(counter)
                            
    df_inv = pd.DataFrame(invalid_images,columns=['path','dataset','family','specie','genus'])
    output_csv= os.path.join(folder,'invalid_paths.csv')
    df_inv.to_csv(output_csv)
    db.commit()

def build_db_from_json(frozen_json_filepaths=[], db_path=r'resources/leavesdb.db', exist_ok=False):
    '''
    Reconstruct database .db file from collection of individual json records, each of which may contain one or more data samples.

    Parameters
    ----------
    frozen_json_filepaths : list(str)
        All significant data should be contained in json files, whose path location is pointed to by each str in list.
    db_path : str
        File path in which to create database.
    exist_ok : bool
        If False and the db exists, deletes db and constructs solely with data contained in provided json.
        If True and the db exists, create temporary backup json, delete db, then recreate from list of 
        archived + new JSON archives

    Returns
    -------
    db : dataset.database.Database
        Open database connection

    '''
    assert type(frozen_json_filepaths)==list
    db_exists = os.path.isfile(db_path)
    
#     if not exist_ok:
#         assert not db_exists
    
    temp_db_path=''
    archive_json=''
    
    if db_exists:
        if exist_ok:
            print(f'{db_path} already exists, creating backup in case of failed reconstruction')
    #         assert exist_ok
            db_dir = ''.join(os.path.split(db_path)[:-1])
            archive_json = freeze_full_database(db_path, prefix=db_dir, filename='temp_data_frozen_archive.json')
            archive_json = archive_json['full_dataset']
            frozen_json_filepaths.append(archive_json)
        
            temp_db_path = os.path.join(db_dir,'temp_db.db')
            os.rename(db_path,temp_db_path)
        else:
            print('Deleting previous db and replacing with contents of frozen_json_filepaths')
            os.remove(db_path)
        
    db_URI = f'sqlite:///{db_path}'
    db = dataset.connect(db_URI)
    table = db.create_table('dataset', primary_id='id')
    
    count = 0
    for filepath in frozen_json_filepaths:
        try:
            # db.begin()
            print(f'Loading {filepath}')
            json_file = load(filepath)
            table.insert_many(json_file['results'], ensure=True)
            db.commit()
            file_count = len(json_file['results'])
            count += file_count
            print(f'committed {file_count} row entries for data from {filepath}')
        except Exception as e:
            print(e)
            db.rollback()
            print('[Error] : Error encountered, rolling back changes')
            os.remove(db_path)
            os.rename(temp_db_path,db_path)
            print('restored db in its previous state')
            return {'success_count': 0}

        
    if os.path.isfile(temp_db_path):
        os.remove(temp_db_path)
    if os.path.isfile(archive_json):
        os.remove(archive_json)

        
    print(f'[SUCCESS]: Database created at {db_path} with {count} added samples')
    
    return {'success_count':count}
    
    
# def build_db_from_json(frozen_json_filepaths=[], db_path=r'resources/leavesdb.db'):
#     '''
#     Reconstruct database .db file from collection of individual json records, each of which may contain one or more data samples.

#     Parameters
#     ----------
#     frozen_json_filepaths : list(str)
#         All significant data should be contained in json files, whose path location is pointed to by each str in list.
#     db_path : str
#         File path in which to create database.

#     Returns
#     -------
#     db : dataset.database.Database
#         Open database connection

#     '''
#     assert type(frozen_json_filepaths)==list
    
#     db_URI = f'sqlite:///{db_path}'
#     db = dataset.connect(db_URI)
#     table = db.create_table('dataset', primary_id='id')
    
#     for filepath in frozen_json_filepaths:
#         json_file = load(filepath)
#         table.insert_many(json_file['results'], ensure=True)
#     return db

####

def update_db(json_filepath, db_path):
    
    TOP_LEVEL_KEYS = ['meta','count','results']
    json_file = load(json_filepath)
    ######
    if not os.path.isfile(db_path):
        print(f'[ERROR]: {db_path} does not exist. In order to use update_db user must first instantiate db using build_db_from_json()')
        raise Exception
    
    for key in json_file.keys():
        if key not in TOP_LEVEL_KEYS:
            print(f'[ERROR]: JSON format invalid, {key} is not in set of admissable top level keys. Must be one of {TOP_LEVEL_KEYS}')
            raise Exception
            
    if len(json_file['results'])==0:
        print('[ERROR]: Provided JSON file contains empty list of new entries')
        raise Exception
    ######        
    
    db_URI = f'sqlite:///{db_path}'
    db = dataset.connect(db_URI)
    table = db['dataset']
    
    table.upsert()


########


def freeze_full_database(db_path, prefix, filename='full_dataset_frozen.json'):
    '''
    Create one frozen json file for all data samples in .db file located at db_path.

    Saves json as flat list of individual records


    Parameters
    ----------
    db_path : str
        Absolute path to source db file. e.g. /media/data/pyleaves/leavesdb/resources/leavesdb.db
    prefix : str
        DESCRIPTION.

    Returns
    -------
    frozen_json_filepaths

    '''
    db_URI = f'sqlite:///{db_path}'
    db = dataset.connect(db_URI)
    table = db['dataset']

    # dataset_rows = list(table.distinct('id'))
    dataset_rows = list(table.all(order_by='id'))
    

    datafreeze.freeze(dataset_rows, 
                      mode='list',
                      format='json', 
                      filename=filename,
                      prefix=prefix)
    frozen_json_filepath = {'full_dataset':os.path.join(prefix,filename)}
    
    return frozen_json_filepath


# def freeze_full_database(json_filepath='resources/full_dataset_frozen.json', db_path='resources/leavesdb.db'):
#     '''
#     Create one frozen json file for all data samples in .db file located at db_path.

#     Saves json as flat list of individual records


#     Parameters
#     ----------
#     json_filepath : str
#         DESCRIPTION.
#     db_path : str
#         Absolute path to source db file. e.g. /media/data/pyleaves/leavesdb/resources/leavesdb.db

#     Returns
#     -------
#     frozen_json_filepaths : dict({str:str})

#     '''
#     db_URI = f'sqlite:///{db_path}'
#     db = dataset.connect(db_URI)
#     table = db['dataset']

#     dataset_rows = list(table.all(order_by='id'))
    
#     datafreeze.freeze(dataset_rows, 
#                       mode='list',
#                       format='json', 
#                       filename=filename,
#                       prefix=prefix)
#     frozen_json_filepath = {'full_dataset':os.path.join(prefix,filename)}
    
#     return frozen_json_filepath

####
    
def freeze_db_by_dataset(db_path='resources/leavesdb.db', prefix='resources', freeze_key='dataset'):
    '''
    Create frozen json files for each dataset in .db file located at db_path.

    Saves json as flat list of individual records


    Parameters
    ----------
    db_path : str
        Absolute path to source db file. e.g. /media/data/pyleaves/leavesdb/resources/leavesdb.db
    prefix : str
        DESCRIPTION.
    freeze_key : str, optional
        DESCRIPTION. The default is 'dataset'.

    Returns
    -------
    frozen_json_filepaths : dict({str:str})

    '''
    db_URI = f'sqlite:///{db_path}'
    db = dataset.connect(db_URI)
    table = db['dataset']

    dataset_names = table.distinct(freeze_key)
    
    frozen_json_filepaths = {}
    for dataset_name in dataset_names:
        dataset_name = dataset_name[freeze_key]
        dataset_rows = table.find(dataset=dataset_name)
        
        filename=f'{dataset_name}_frozen.json'
        datafreeze.freeze(dataset_rows, 
                          mode='list',
                          format='json', 
                          filename=filename,
                          prefix=prefix)
        frozen_json_filepaths.update({dataset_name:os.path.join(prefix,filename)})
    
    return frozen_json_filepaths

        
    
    
    
    
    
def main():
    join = os.path.join
    cwd = os.getcwd()

    prefix = join(cwd,'resources')
    db_path = join(cwd,r'resources/leavesdb.db')
    SOURCE_full_json = join(cwd,r'resources/full_dataset_frozen.json')
    run_logs = TimeLogs()
    
    pipeline = [dict(func=build_db_from_json,
                     name='build_db_from_full_json', 
                     frozen_json_filepaths=[SOURCE_full_json],
                     db_path=db_path)]

    for node in pipeline:
        run_logs.timeit(**node)
        
    print(run_logs)
        
#     frozen_json=[
#                  r'resources/Fossil_frozen.json',
#                  r'resources/Leaves_frozen.json',
#                  r'resources/PNAS_frozen.json',
#                  r'resources/plant_village_frozen.json',
#                  ]
        
    run_logs.timeit(func=freeze_db_by_dataset,
                    name='freeze_db_by_dataset',
                    db_path=db_path,
                    prefix=prefix, 
                    freeze_key='dataset')
    
    print(run_logs)
#     frozen_json = freeze_db_by_dataset(db_path, prefix, freeze_key='dataset')
#     print(json.dumps(frozen_json, indent='\t'))

    
    
if __name__=='__main__':
    
    main()
