# @Author: Jacob A Rose
# @Date:   Wed, April 15th 2020, 3:46 am
# @Email:  jacobrose@brown.edu
# @Filename: experiments_db.py

'''
Script for managing sqlite database containing experiment and run metadata

'''

from IPython.core.display import display, HTML
import copy
import json
import numpy as np
import os
import dataset
from stuf import stuf
import pandas as pd
from tabulate import tabulate

from pyleaves import EXPERIMENTS_DB, RESOURCES_DIR


EXPERIMENTS_SCHEMA = os.path.join(RESOURCES_DIR,'experiments_schema.sql')



def execute_sql_file(db, filepath):
    schema=''.join(open(filepath,'r').readlines())
    for stmt in schema.split(';'):
        db.engine.execute(stmt)

def _print_table(table_title, table_data):
    pdtabulate=lambda df:tabulate(pd.DataFrame(df),headers='keys')
    return '\n'.join([table_title.capitalize(), '='*20, pdtabulate(table_data)])

def print_tables(table_dict):
     return '\n\n'.join([_print_table(title, tbl) for title, tbl in table_dict.items()])


def select_by_col(table: pd.DataFrame, column: str=None, value=None):
    if column == None:
        return table
    if value != None:
        return table[table.loc[:,column]==value]
    else:
        return table.loc[:,column]


def select_by_multicol(table, kwargs={}):
    select = np.array([True]*table.shape[0])
    for key,value in kwargs.items():
        if value:
            select = select & (table.loc[:,key]==value)
    return table[select]




def create_db(db_path=EXPERIMENTS_DB, schema_path=EXPERIMENTS_SCHEMA):
    """
    Use this function to create the experiment management sqlite database defined by a schema sql file.

    Parameters
    ----------
    db_path : str
        Abs path at which to save created database.
    schema_path : str
        Abs path at which to find database schema defined in an sql file.

    Returns
    -------
    dict
        Optionally use this function in another python script and return a dict mapping table names to their contents as pd.DataFrames

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    db = dataset.connect('sqlite:///'+db_path+'?charset=utf8')
    execute_sql_file(db, filepath=schema_path)
    tables = {}
    for tbl in db.tables:
        tables.update({tbl:pd.DataFrame(db[tbl])})
    return tables

def get_db_contents(db_path=EXPERIMENTS_DB):
    db = dataset.connect('sqlite:///'+db_path+'?charset=utf8')
    tables = {}
    for tbl in db.tables:
        tables.update({tbl:pd.DataFrame(db[tbl])})
    return tables

def get_db_table(tablename, db_path=EXPERIMENTS_DB):
    tables = get_db_contents(db_path=db_path)
    return tables[tablename]


class DataBase:
    db_path = None
    db = None

    @classmethod
    def connect(cls, db_path=None):
        cls.db_path = db_path or cls.db_path
        cls.db = dataset.connect('sqlite:///'+db_path)



class Table(DataBase):
    __tablename__ = None
    columns = []

    def __init__(self, db_path=None, tablename=None):
        cls = type(self)
        if not cls.db:
            cls.connect(db_path=db_path)
        self.__tablename__ = tablename or cls.__tablename__
        self.__tabledata__ = cls.db[self.__tablename__]
        self.columns = set(self.__tabledata__.columns)

    @property
    def table(self):
        return self.__tabledata__

    def to_dataframe(self):
        return pd.DataFrame(self.table)

    def display_html(cls):
        '''Displays a cleanly formatted view of the table optimized for web documents, Jupyter Notebooks in particular.'''
        display(HTML(cls.to_dataframe().to_html()))

    @property
    def num_entries(self):
        return self.to_dataframe().shape[0]

    def info(self):
        print(
        '\n'.join([
                    f'Database connection = {type(self).db_path}',
                    f'Table name = {self.__tablename__},',
                    f'num_entries = {self.num_entries}',
                     'columns:\n   ' + '\n   '.join(self.columns)
                 ])
        )


class TFRecordItem:

        def __init__(self,
                     file_path : str,
                     file_group : str,
                     dataset_stage : str,
    	             run_id : str,
        	         experiment_type : str,
            	     dataset_name : str,
                     resolution : int,
                     num_channels : int,
                     num_classes : int,
                     num_shards : int,
                     num_samples : int = 0,
                     subrun_id : str = None):
            """
            Container class for rigidly defining a single TFRecord shard item, which represents
            one value for every column in the tfrecords SQLite table.


            Returns
            -------
            type
                Description of returned object.

            Examples
            -------
            Examples should be written in doctest format, and
            should illustrate how to use the function/class.
            >>>    tfrecord_db_items = []
            ...    tfrecord_db_items.append(
            ...                             TFRecordItem(**{
            ...                                        	'file_path':r'/media/data/jacob/Fossil_Project/data/tfrecord_data/A_train_val_test/Leaves/train-00000-of-00010.tfrecord',
            ...                                         'file_group':'train',
            ...                                         'dataset_stage':'dataset_A'
            ...                                        	'run_id':'1200',
            ...                                        	'experiment_type':'A_train_val_test',
            ...                                        	'dataset_name':'Leaves',
            ...                                         'resolution':224,
            ...                                         'num_channels':3,
            ...                                         'num_classes':190,
            ...                                         'num_samples':0,
            ...                                         'subrun_id':None,
            ...                                         'num_shards':10
            ...                                         })
            ...                             )

            """

            self._params = {
                             'file_path':file_path,
                             'file_group':file_group,
                             'dataset_stage':dataset_stage,
            	             'run_id':run_id,
                	         'experiment_type':experiment_type,
                    	     'dataset_name':dataset_name,
                             'resolution':resolution,
                             'num_channels':num_channels,
                             'num_classes':num_classes,
                             'num_shards':num_shards,
                             'num_samples':num_samples,
                             'subrun_id':subrun_id or run_id

            }
            self.columns = tuple(self._params.keys())

        @property
        def params(self):
            return self._params

        def __repr__(self):
            return json.dumps(self.params,indent='  ')

        def __eq__(self, other):
            return self.params == other.params

        def __hash__(self):
            return hash(tuple(self.params.items()))


columns = ['file_path',
       'file_group',
       'dataset_stage',
       'run_id',
       'experiment_type',
       'dataset_name',
       'resolution',
       'num_channels',
       'num_classes',
       'num_shards',
       'num_samples',
       'subrun_id']




class TFRecordsTable(Table):

    __tablename__ = 'tfrecords'

    def __init__(self, db_path=None):
        super().__init__(db_path=db_path, tablename=type(self).__tablename__)
        self.logs=[]
        for i, row in self.to_dataframe().iterrows():
            self.logs.append(TFRecordItem(**row.to_dict()))

        for c in self.columns:
            assert c in columns, print(c)

    def log_tfrecord(self, tfrecord_item):
        assert type(tfrecord_item) == TFRecordItem
        assert set(tfrecord_item.columns) == set(self.columns)
        if tfrecord_item in self.logs:
            print('Attempted to log an already existing tfrecord, returning False.')
            return False
        else:
            self.table.insert(tfrecord_item.params)
            print('Logged TFRecord')
            self.logs.append(tfrecord_item)
            return True

    def log_tfrecords(self, tfrecord_items : list):
        for item in tfrecord_items:
            self.log_tfrecord(tfrecord_item=item)

    def check_if_logged(self, tfrecord_items : list):
        return [(item in self.logs) for item in tfrecord_items]


#
#
#     def __init__(self,
#                  file_path,
#                  file_group,
# 	             run_id,
#     	         experiment_type,
#         	     dataset_name,
#                  resolution,
#                  num_channels,
#                  num_classes,
#                  num_shards):
#         self._metadata = {
#                          'file_path':file_path,
#                          'file_group':file_group,
#         	             'run_id':run_id,
#             	         'experiment_type':experiment_type,
#                 	     'dataset_name':dataset_name,
#                          'resolution':resolution,
#                          'num_channels':num_channels,
#                          'num_classes':num_classes,
#                          'num_shards':num_shards
#         }
#
# def log_tfrecord(run_id, log_params, db_path=EXPERIMENTS_DB):
#
#     db = dataset.connect('sqlite:///'+db_path+'?charset=utf8')
#     run_view = select_by_col(table=tables['tfrecords'],column='run_id',value=run_id)
#
#
#     log_params = {
#     	'file_path':r'/media/data/jacob/Fossil_Project/data/tfrecord_data/A_train_val_test/Leaves/train-00000-of-00010.tfrecord',
#         'file_group':'train',
#     	'run_id':'1200',
#     	'experiment_type':'A_train_val_test',
#     	'dataset_name':'Leaves',
#         'resolution':224,
#         'num_channels':3,
#         'num_classes':190,
#         'num_shards':10
#         }
#
#     db.query(
#     '''
#     INSERT INTO tfrecords (file_path,file_group,subrun_id,run_id,experiment_type,dataset_name,resolution,num_channels,num_classes,num_shards)
#     VALUES ()
#         ''')
#
#     return

def query_tfrecords():



    return

if __name__=='__main__':
    print('Creating experiment manager database')

    create_db()

    print('Loading experiment manager database')

    tables = get_db_contents()

    print('Printing experiment manager database')

    print(print_tables(tables))
