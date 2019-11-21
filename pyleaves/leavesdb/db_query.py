'''
Convenience functions for common queries from database. Useful as templates for more complex queries.

'''

import dataset
import pandas as pd
from stuf import stuf


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

