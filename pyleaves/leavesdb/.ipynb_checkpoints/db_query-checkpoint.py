'''
Convenience functions for common queries from database. Useful as templates for more complex queries.

'''

import dataset
import pandas as pd
from stuf import stuf

from pyleaves.data_pipeline.preprocessing import generate_encoding_map, encode_labels, filter_low_count_labels
from pyleaves import leavesdb

def get_label_encodings(data_df=None, dataset='Fossil', y_col='family', low_count_thresh=0):
    '''
    Arguments:
        db: dataset.database.Database, Must be an open connection to a database
        x_col: str, Inputs column. Should usually be the column containing filepaths for each sample
        y_col: str, Labels column. Can be any of {'family','genus','species'}
        dataset: str, Can be any dataset name that is contained in db

    Return:
        label_maps:  dictionary mapping integer labels to their corresponding text label
            e.g.        {0:'Annonaceae',
                    ...
                    19:'Passifloraceae'}


    '''
    if data_df is None:
        data, _ = load_from_db(dataset_name=dataset)
    else:
        data=data_df
    
    
    data_df = encode_labels(data)
    data_df = filter_low_count_labels(data_df, threshold=low_count_thresh)
    data_df = encode_labels(data_df) #Re-encode numeric labels after removing sub-threshold classes so that max(labels) == len(labels)

    label_maps = generate_encoding_map(data_df, text_label_col=y_col, int_label_col='label')

    return label_maps

def load_all_data(db, x_col='path', y_col='family'):
    '''
    Function to load x_col and y_col for each row in db from all datasets
    '''
#     paths_labels = list(db['dataset'].distinct(x_col, y_col, 'dataset'))
    data = pd.DataFrame(db['dataset'].distinct(x_col, y_col, 'dataset'))
    return data

#     data_by_dataset = data.groupby(by='dataset')
#     data_by_dataset_dict = {k:v for k,v in data_by_dataset}
    
    
    

def load_data(db, x_col='path', y_col='family', dataset='Fossil'):
	'''
	General data loader function with flexibility for all query types.

	Arguments:
		db: dataset.database.Database, Must be an open connection to a database
		x_col: str, Inputs column. Should usually be the column containing filepaths for each sample
		y_col: str, Labels column. Can be any of {'family','genus','species'}
		dataset: str, Can be any dataset name that is contained in db

	Return:
		paths_labels: dataset.util.ResultIter,

	'''
	paths_labels = db['dataset'].distinct(x_col, y_col, dataset=dataset)
	return paths_labels

def load_from_db(dataset_name='PNAS'):
    local_db = leavesdb.init_local_db()
    print(local_db)
    db = dataset.connect(f'sqlite:///{local_db}', row_type=stuf)
    data = leavesdb.db_query.load_data(db, dataset=dataset_name)
    return data, db

####################################################
'''
Functions to load one pre-determined dataset by name
'''

def load_Fossil_data(db):
    return load_data(db=db, x_col='path', y_col='family', dataset='Fossil')

def load_Leaves_data(db):
    return load_data(db=db, x_col='path', y_col='family', dataset='Leaves')

def load_plant_village_data(db):
    return load_data(db=db, x_col='path', y_col='family', dataset='plant_village')

def load_PNAS_data(db):
    return load_data(db=db, x_col='path', y_col='family', dataset='PNAS')
