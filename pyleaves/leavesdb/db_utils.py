import numpy as np
import os
import pandas as pd
import dataset
from stuf import stuf
import json
import cv2
import shutil
'''HELPER FUNCTIONS'''
from pyleaves.utils import ensure_dir_exists
import pyleaves


# print(dir(pyleaves))

def init_local_db(local_db = os.path.expanduser(r'~/scripts/leavesdb.db'), src_db = pyleaves.DATABASE_PATH, force_update=True, verbose=True):
	'''
	Whenever working on a new machine, run this function in order to make sure the main leavesdb.db file is stored locally to avoid CIFS permissions issues.

	usage: init_local_db()

    force_update, bool:
        default True, if false, then only copy from src_db if local_db doesn't exist.
	'''
	ensure_dir_exists(os.path.dirname(local_db))

	if (not os.path.isfile(local_db)) or force_update:
		if verbose: print(f'Copying sql db file from {src_db} to {local_db}')
		shutil.copyfile(src_db, local_db)
	if verbose: print(f'Proceeding with sql db at location {local_db}')

	return local_db

###############################################################################################################
###############################################################################################################





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
def __get_datasets_per_family(db):
    '''
    Helper function similar to __get_family_names_per_dataset, but instead datasets per family
    Arguments:
        db : open connection to database
        e.g. db = dataset.connect(f'sqlite:///{db_path}', row_type=stuf)
    Return:
        datasets_per family : dict{family:set(datasets)})
        e.g. {'Adoxaceae': {'Fossil', 'Leaves'},...
    '''
    dataset_families = __get_family_names_per_dataset(db)
    datasets_per_family={}
    for dataset in dataset_families:
        for family in dataset[1]:
            if family not in datasets_per_family:
                datasets_per_family[family]=set()
            datasets_per_family[family].add(dataset[0])
    return datasets_per_family
def __get_datasets_per_family_thresh(db,at_least=2):
    '''
    Helper function similar to __get_family_names_per_dataset, but instead datasets per family
    Arguments:
        db : open connection to database
        at_least: number of datasets that has to be present in the family
        e.g. db = dataset.connect(f'sqlite:///{db_path}', row_type=stuf)
    Return:
        datasets_per family : dict{family:set(datasets)})
        e.g. {'Adoxaceae': {'Fossil', 'Leaves'},...
    '''

    datasets_per_family = __get_datasets_per_family
    datasets_per_family_thresh={}
    for family in datasets_per_family:
        if  len(datasets_per_family[family])>=at_least:
            datasets_per_family_thresh[family]=datasets_per_family[family]
    return datasets_per_family_thresh

def __get_datasets_per_family_with(db,include_dataset='Fossil'):
    """ Datasets that share families
    Arguments:
        db : open connection to database
        include_dataset: dataset that must be included
        e.g. db = dataset.connect(f'sqlite:///{db_path}', row_type=stuf)
    Return:
        datasets_per family : dict{family:set(datasets)})
        e.g. {'Adoxaceae': {'Fossil', 'Leaves'},...
    """
    datasets_per_family = __get_datasets_per_family
    datasets_per_family_with={}
    for family in datasets_per_family:
        if include_dataset in datasets_per_family[family]:
            datasets_per_family_with[family]=datasets_per_family[family]
    return datasets_per_family_with

def __get_file_ext_per_dataset(db, file_ext='all', summarize=False):
    '''
    Function to get the number of files in each dataset, grouped by file extension.
    Can also return full file path lists stored in the same dict format:

        result[dataset][file_ext] = value

    Arguments:
        db :
            open connection to database
        file_ext, str :
            if file_ext=='all', return all filetypes.
            Otherwise limit to just the specified extension.
        summarize, bool :
            if True: return dict containing a summary of the quantity of files per ext
            if False: return dict containing full list of paths for each ext
    '''
    file_ext_whitelist = ['jpg','tif','tiff', 'png','all']

    get_file_ext = lambda x : os.path.splitext(x)[-1][1:]
    filter_by_ext = lambda x, pattern : x.str.match(pattern)

    dataset_names = [name['dataset'] for name in db['dataset'].distinct('dataset')]
    file_ext_path_dict = {}

    for name in dataset_names:
        file_ext_path_dict[name] = {}

        dataset_rows = pd.DataFrame(db['dataset'].distinct('id','path',dataset=name))
        file_extensions = list(map(get_file_ext,dataset_rows['path'].values))
        unique_ext = np.unique(file_extensions)

        if (file_ext in file_ext_whitelist):
            if (file_ext!='all'):
                unique_ext=[file_ext]

            for ext in unique_ext:
                pattern = '.*\.' + ext
                matched_rows = filter_by_ext(dataset_rows['path'], pattern)

                if summarize:
                    file_ext_path_dict[name][ext] = len(dataset_rows[matched_rows])
                    print(name, ext, file_ext_path_dict[name][ext])
                else:
                    file_ext_path_dict[name][ext] = dataset_rows[matched_rows]
    return file_ext_path_dict


# for k, v in results.items():
#     ext_keys = list(v.keys())[0]
#     cols = [col for col in v[ext_keys].keys()]

#     num_samples = [v[ext_keys][cols[0]].shape]
#     print(k, ext_keys, cols, num_samples)

#             import re
#             re.findall('*.jpg', dataset_rows['path'])


#         return



###############################################################################################################
###############################################################################################################

def summarize_db(db):
    '''
    Combines helper functions to summarize key info about the data in opened database, db.
    '''
    summary = {'Database column keys':db['dataset'].columns,
               'distinct datasets':[name['dataset'] for name in db['dataset'].distinct('dataset')],
               'Number of distinct families':__get_num_families_per_dataset(db),
               'Number of rows in db': len(db['dataset'])}
    print(summary)
    return summary

def load(file):
    '''
    Helper function to load a json file.

    Arguments:
        - file : Path to the json file
    Return:
        - dictionary with the json
    '''
    with open(file,encoding='utf-8') as infile:
        inside=json.load(infile)
    return inside

def save(file,obj):
    '''
    Helper function to save a json file.

    Arguments:
        - file : Path to the json file
        - obj : python object to save in json format.
    Return:
        - dictionary with the json
    '''
    with open(file, 'w') as outfile:
        json.dump(obj, outfile)

def image_checker(p):
    try:
        image = cv2.resize(cv2.imread(p),(229,229))
        return True
    except:
        print(f'Problem with image path {p}, not valid.')
        return False


def flattenit(pyobj, keystring =''):
    '''
    Function to flatten a dictionary

    Arguments:
        -pyobj python object
        -keystring : String that is being looked for

    Returns :
        - flattened structure
    '''

    if type(pyobj) is dict:
        if (type(pyobj) is dict):
            keystring = keystring + "_" if keystring else keystring
            for k in pyobj:
                yield from flattenit(pyobj[k], keystring + k)

        elif (type(pyobj) is list):
            for lelm in pyobj:
                yield from flatten(lelm, keystring)
    else:
        yield keystring, pyobj






from time import perf_counter
##########################

class TimeLog:

    def __init__(self,
                 name='',
                 start_time=0,
                 end_time=0,
                 duration=0,
                 result=None
                 ):
        self.name = ''
        self.log = {'start_time':start_time,
                 'end_time':end_time,
                 'duration':duration}

        self.result = result

    def start(self):
        self.log['start_time'] = perf_counter()
    def stop(self):
        self.log['end_time'] = perf_counter()
        self.log['duration'] = self.log['end_time'] - self.log['start_time']

    def timeit(self, func, name=None, *args, **kwargs):
        if name:
            self.name = name
        # start_time = perf_counter()
        self.start()
        self.result = func(*args, **kwargs)
        # end_time = perf_counter()
        self.stop()

        if 'verbose' in kwargs.keys():
            print(self)

        return self.__dict__

    def __repr__(self):
        return '\n'.join([json.dumps(self.__dict__[key],indent=4) for key in ['names','logs','results']])


class TimeLogs(TimeLog):

    def __init__(self):
        super().__init__(self)

        self.names = []
        self.logs = {'start_time':[],
                 'end_time':[],
                 'duration':[]}

        self.results = []

        self.__records = []


    def timeit(self, func, name=None, *args, **kwargs):

        new_record = super().timeit(func=func, *args, name=name, **kwargs)

        self.__records.append(new_record)

        self.__dict__.update(new_record)
        self.commit_new_log()


    def commit_new_log(self):
        '''
        Run right after executing self.timeit() to commit new individual log values to logs collection.

        Returns
        -------
        None.

        '''
        self.names.append(self.name)
        for k, v in self.logs.items():
            v.append(self.log[k])
        self.results.append(self.result)

        assert len(self)>0


    def __len__(self):
        num_names=len(self.names)

        _len = [num_names,
                *(len(_log) for _log in self.logs.values()),
                len(self.results)
                ]
        __all_equal__ = [l==num_names for l in _len]
        if np.all(__all_equal__):
            return num_names

        else:
            raise "[ERROR]: lengths inconsistent"
