'''
Functions for preprocessing/transforming data between extraction from the database and input to the model.
'''

import pandas as pd
import dataset



def encode_labels(data, y_col='family'):
	'''
	Create 'label' column in data_df that features integer values corresponding to text labels contained in y_col.
	
	Arguments:
		data: dataset.util.ResultIter, Should be the returned result from loading data from the leavesdb database (e.g. data = leavesdb.db_query.load_data(db)).
		y_col: str, name of the columns containing text labels for each sample in data.
	Returns:
		data_df: pd.DataFrame, Contains 3 columns, one for paths, one for str labels, and one for int labels.
	'''
	data = pd.DataFrame(data)
	data['label'] = pd.Categorical(data[y_col])
	data['label'] = data['label'].cat.codes
	return data

