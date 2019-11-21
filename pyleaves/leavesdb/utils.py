import os
import dataset
from stuf import stuf

'''HELPER FUNCTIONS'''

def init_local_db(local_db = os.path.expanduser(r'~/scripts/leavesdb.db'), src_db = r'/media/data_cifs/irodri15/data/db/leavesdb.db'):
	'''
	Whenever working on a new machine, run this function in order to make sure the main leavesdb.db file is stored locally to avoid CIFS permissions issues.
	
	usage: init_local_db()
	'''

	if not os.path.isfile(local_db):
		print(f'Copying sql db file from {src_db} to {local_db}')
		shutil.copyfile(src_db, local_db)
	print(f'Proceeding with sql db at location {local_db}')
	
	return local_db



def __get_family_names_per_dataset(db):
    '''
    Helper function that returns dataset_families, a list of tuples: [(,),(,),...]
    db = dataset.connect(f'sqlite:///{db_path}', row_type=stuf)
    dataset_families contains tuples of len == 2, where item 0 is a dataset name, and item 1 is a list of strings, one for each family name in the dataset.
    e.g. [('Fossil',['Adoxaceae', 'Anacardiaceae',...]),
            ('PNAS',['Apocynaceae','Betulaceae',...]),
            ...]
    '''
    dataset_families = []
    for dataset in db['dataset'].distinct('dataset'):
        dataset_name = dataset.dataset
        distinct_families = db['dataset'].distinct('family', dataset=dataset_name)
        dataset_families.append((dataset_name, [fam.family for fam in distinct_families]))
    return dataset_families

def __get_num_families_per_dataset(db):
    '''
    Helper function similar to __get_family_names_per_dataset, but instead of tuple containing (dataset_name, list(family names)), 
    returns the total number of unique families for each dataset.
    Arguments:
        db : open connection to database
        e.g. db = dataset.connect(f'sqlite:///{db_path}', row_type=stuf)
    Return:
        num_families_per_dataset : list(tuples(str,int))
        e.g. [('Fossil',27),
                ('PNAS',19),
                ...]
    '''
    num_families_per_dataset = []
    dataset_families = __get_family_names_per_dataset(db)
    for dataset in dataset_families:
        num_families_per_dataset.append((dataset[0], len(dataset[1])))
    return num_families_per_dataset


def summarize_db(db):
    '''
    Combines helper functions to summarize key info about the data in opened database, db.
    '''
    print('Database column keys:\n', db['dataset'].columns)
    print('Number of distinct families:\n', __get_num_families_per_dataset(db))
    print(f"Number of rows in db:\n {len(db['dataset'])}")