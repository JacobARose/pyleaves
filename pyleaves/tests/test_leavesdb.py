'''
Functions for testing the validity of data entries in leavesdb.db

'''
import dataset
import os
import pandas as pd
from stuf import stuf

from pyleaves import leavesdb


def validate_image(image_path, file_ext_whitelist=['.jpg','.tif','.png']):
    '''
    Function for performing a set of checks on a single file specified by image_path.

    Currently performs checks:
        1) verify image_path refers to a valid file
        2) verify that image_path ends with an extension that is included in file_ext_whitelist,
           i.e. verify that image_path is a valid *image* file


    Arguments:
        image_path, str:
            absolute path to image file to be validated
        file_ext_whitelist, list(str):
            List of allowable file extensions to consider valid

    Return:


    '''
    validated = (image_path, False)
    if not os.path.isfile(image_path):
        return validated

    for ext in file_ext_whitelist:
        if image_path.endswith(ext):
            validated=(image_path,True)
            return validated

    if validated[1]==False:
        print('Found invalid filepath: ', image_path)
    return validated




def test_db_validate_images(db_path=None):

    if db_path is None:
        db_path = leavesdb.db_utils.init_local_db()

    db_URI = f'sqlite:///{db_path}'
    db = dataset.connect(db_URI)

    data = pd.DataFrame(db['dataset'].all())
    data_by_dataset = data.groupby(by='dataset')
    data_by_dataset_dict = {k:v for k,v in data_by_dataset}

    validated_datasets = {}
    for dataset_name, dataset_rows in data_by_dataset:
        print(dataset_name, dataset_rows.shape)

        validated_data = dataset_rows['path'].apply(validate_image)
        # pd.DataFrame().apply
        validated_datasets.update({dataset_name:validated_data})

        
        
        
        
        
        
        
        
        
        
        
        
        