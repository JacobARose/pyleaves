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

import numpy as np
import pandas as pd 
import dataset
import datafreeze
import json
import pyleaves
from pyleaves import leavesdb
from pyleaves.leavesdb.db_utils import load, flattenit, image_checker, TimeLogs
import cv2 
import os
from stuf import stuf

PATH = os.path.abspath(os.path.join(os.getcwd(),'..','leavesdb','resources'))#/datasets.json') #/media/data_cifs/irodri15/data/processed/datasets.json'
# OUTPUT = 'sqlite:///resources/leavesdb.db'


def dict2json(data, prefix, filename):
    '''
    Convert a list of dicts into proper json format for contructing sql db
    
    Arguments:
        data, list(dict):
            e.g. [{'id':1,'path':...,'family':...}, {'id':2,...}]
        prefix, str:
            Directory in which to save file
        filename, str:
            Name of file in which to save json
        
    Return:
        
    '''
    data_count = len(data)
    json_output = {"count":data_count,
                   "results":data,
                   "meta":[]}

    datafreeze.freeze(data, 
                      mode='list',
                      format='json', 
                      filename=filename,
                      prefix=prefix)
    
    return json_output


# def insert_files2table(filepath, table, db):
#     #TODO Batch insert files loaded from json fileparh, add function to create master copy as jpeg on /media/data
#     print(f'Loading {filepath}')
#     json_file = load(filepath)
#     table.insert_many(json_file['results'], ensure=True)
#     db.commit()
#     file_count = len(json_file['results'])
#     count += file_count
#     print(f'committed {file_count} row entries for data from {filepath}')

def archivedb_to_json(db_path, exist_ok):
    temp_db_path = ''
    archive_json = ''
    if exist_ok:
        print(f'{db_path} already exists, creating backup in case of failed reconstruction')
#        assert exist_ok
        db_dir = ''.join(os.path.split(db_path)[:-1])
        archive_json = freeze_full_database(db_path, prefix=db_dir, filename='temp_data_frozen_archive.json')
        archive_json = archive_json['full_dataset']
        frozen_json_filepaths.append(archive_json)
        temp_db_path = os.path.join(db_dir,'temp_db.db')
        os.rename(db_path,temp_db_path)
        
    else:
        print('Deleting previous db and replacing with contents of frozen_json_filepaths')
        os.remove(db_path)
        
    return archive_json, temp_db_path


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
        #Temporarily store current database as a json archive
        archive_json, temp_db_path = archivedb_to_json(db_path, exist_ok)
        
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
            if db_exists:
                print('restoring previous db')
                os.rename(temp_db_path,db_path)
            print('restored db in its previous state')
            return {'success_count': 0}

        
    if os.path.isfile(temp_db_path):
        os.remove(temp_db_path)
    if os.path.isfile(archive_json):
        os.remove(archive_json)        
    print(f'[SUCCESS]: Database created at {db_path} with {count} added samples')
    
    return {'success_count':count}


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

def remove_invalid_images_from_db(invalid_paths: list, path_col='path', db=None, local_db=None, prefix = None, json_filename='database_records.json'):
    '''
    Provide a list of path names to remove from the database.
    
    path_col, str:
        Either 'path' or 'source_path', depending on which contains the path name you want to filter out
    '''
    if db is None:
        if local_db is None:
            local_db = leavesdb.init_local_db()
            
        db = dataset.connect(f"sqlite:///{local_db}", row_type=stuf)
        
    if prefix is None:
        prefix = pyleaves.RESOURCES_DIR
    
    data = pd.DataFrame(db['dataset'].all())
    
    data_filter = data.loc[:,path_col].isin(invalid_paths)
    
    filtered_data = data[~data_filter]
    
    filtered_records = filtered_data.to_dict('records')
    
    dict2json(filtered_records, prefix=prefix, filename=json_filename)
    db_json_path = os.path.join(prefix,json_filename)
    db_path = os.path.join(prefix,'leavesdb.db')
    #CREATE & WRITE new SQLite .db file from newly created JSON
    build_db_from_json(frozen_json_filepaths=[db_json_path], db_path=db_path)

    print(f'FILTERED database of {data.shape[0] - filtered_data.shape[0]} duplicates.')
    print(f'Previous size = {data.shape[0]}')
    print(f'New size = {filtered_data.shape[0]}')
    
    


def clear_duplicates_from_db(db=None, local_db=None, prefix = None, json_filename='database_records.json'):
    '''
    Function checks db file for duplicate filenames in 'path' column, recreates JSON and db without duplicates.
    
    data_records = clear_duplicates_from_db(db=None, local_db='./resources/leavesdb.db', prefix = './resources', json_filename='database_records.json')
    
    db, Database,
        open connection to database
    local_db, str:
        abs path to .db file
    prefix, str:
        abs path to directory in which to save filtered json and .db files.
        
    Return:
        unique_data, pd.DataFrame:
            DataFrame containing only unique 
    '''
    if db is None:
        if local_db is None:
            local_db = leavesdb.init_local_db()
            
        db = dataset.connect(f"sqlite:///{local_db}", row_type=stuf)
        
    if prefix is None:
        prefix = PATH
        print(PATH)
    
    data = pd.DataFrame(db['dataset'].all())
    paths, indices, counts = np.unique(data['path'], return_index=True, return_counts=True)
    #SELECT only rows with unique file paths
    unique_data = data.iloc[indices,:]
    #CONVERT DataFrame to list of dict records
    data_records = unique_data.to_dict('records')
    #CREATE & WRITE JSON records file containing previous file info combined with new file paths
    dict2json(data_records, prefix=prefix, filename=json_filename)
    db_json_path = os.path.join(prefix,json_filename)
    db_path = os.path.join(prefix,'leavesdb.db')
    #CREATE & WRITE new SQLite .db file from newly created JSON
    build_db_from_json(frozen_json_filepaths=[db_json_path], db_path=db_path)

    print(f'FILTERED database of {data.shape[0] - unique_data.shape[0]} duplicates.')
    print(f'Previous size = {data.shape[0]}')
    print(f'New size = {unique_data.shape[0]}')
    return unique_data


def analyze_db_contents():
    
    local_db = leavesdb.init_local_db()
            
    db = dataset.connect(f"sqlite:///{local_db}", row_type=stuf)    
    
    data = pd.DataFrame(db['dataset'].all())
    paths, indices, counts = np.unique(data['path'], return_index=True, return_counts=True)

    count_number, duplicate_counts = np.unique(counts, return_counts=True)

    print('FOUND:')
    for i in range(len(count_number)):
        print(f'{duplicate_counts[i]} UNIQUE paths with {count_number[i]} duplicates')
    print('-'*10)
    print(f'Keeping a total of {sum(duplicate_counts)} paths and discarding {data.shape[0]-len(indices)} duplicates')
        
        
    

    
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





############################################################################

def create_db(jsonpath=PATH, folder= 'resources'):
    '''
    #DEPRECATED
    
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
    
####################################################################
    