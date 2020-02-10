'''
Functions for testing the validity of data entries in leavesdb.db

'''
import argparse
import cv2
from PIL.Image import Image
import imageio
import dataset
import imghdr
from more_itertools import chunked, unzip, collapse
import numpy as np
import os
import pandas as pd
import swifter
from stuf import stuf

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from PIL import Image


from pyleaves import leavesdb
from pyleaves.utils import ensure_dir_exists

VALIDATION_RESULTS_PATH = os.path.expanduser('~/tests/test_leavesdb_validation_results.csv')


db_path = '/home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db'

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)


def dask_validate_image(image_path, image_id, dataset_name='na'):
    '''
    Function for performing a set of checks on a single file specified by image_path.

    Adds additional step of checking validity of image read from disk

    Currently performs checks:
        1) verify image_path refers to a valid file
        2) verify that image_path ends with an extension that is included in file_ext_whitelist,
           i.e. verify that image_path is a valid *image* file


    Arguments:
        image_path, str:
            absolute path to image file to be validated
        file_ext_whitelist, list(str):
            List of allowable file extensions to consider valid
        dataset_name, str:
            Default: 'na' for not-applicable
            optional str to reference dataset that file belongs to

    Return:
        validated, tuple(str,str,bool):
            tuple that contains (dataset_name, image_path, True or False), where False indicates an invalid image.

    '''
    
    validated = [image_id, dataset_name, image_path, False]    
    
    if image_id%100==0:
        print(image_id)
    
    try:
        img = imageio.imread(image_path)
        assert (type(img) in [imageio.core.util.Array, np.ndarray]), "invalid_image_type: "+str(type(img))
        validated= [image_id, dataset_name, image_path, True]
    except Image.DecompressionBombWarning as dw:
        print(dw)
        print(f'Suspect image_id = {image_id}, image path = {image_path}')
    except Image.DecompressionBombError as de:
        print(de)
        print(f'Suspect image_id = {image_id}, image path = {image_path}')
    except:
        print("Unexpected error:", sys.exc_info())
        print('EXCEPTION WHILE READING image # ', image_id, ' at path: ',image_path)
    finally:
        return validated
    
def dask_validate_batch(image_info_list, dataset_name='na'):
    results = []
    for image_path, image_id in image_info_list:
        results.append(dask_validate_image(image_path, image_id, dataset_name=dataset_name))
    return results

def test_db_dask_validate_images(db_path=None):

    ensure_dir_exists(os.path.dirname(VALIDATION_RESULTS_PATH))
    if os.path.isfile(VALIDATION_RESULTS_PATH):
        os.remove(VALIDATION_RESULTS_PATH)
    if db_path is None:
        db_path = leavesdb.db_utils.init_local_db()
    db_URI = f'sqlite:///{db_path}'
    db = dataset.connect(db_URI)

    data = pd.DataFrame(db['dataset'].all())
    data_by_dataset = data.groupby(by='dataset')
    data_by_dataset_dict = {k:v for k,v in data_by_dataset}
 
    #client = Client(threads_per_worker=10, n_workers=1)
    
    validated_datasets = pd.DataFrame(columns=['image_id','dataset','path','valid'])
    validated_datasets.to_csv(VALIDATION_RESULTS_PATH, mode='w', index=False)
    batch_size = 64
    
    dataset_name='Leaves'; dataset_rows = data_by_dataset_dict['Leaves'][:2000]
    if True:
#     for dataset_name, dataset_rows in data_by_dataset:
#         if dataset_name == 'Fossil':
#             continue
        print('validating dataset: ', dataset_name, ', shape=', dataset_rows.shape)
        
        dd = dask.delayed
        
        image_paths = list(dataset_rows['path'])
        num_paths = len(image_paths)
        image_ids = list(range(num_paths))
        
        testing_data = [dd(row) for row in zip(image_paths,image_ids)]
#         testing_data = list(zip(image_paths,image_ids))

#         validate = dask.delayed(dask_validate_image)
#         results = [validate(row[0],row[1],dataset_name) for row in testing_data]    
        
        chunked_data = list(chunked(testing_data, batch_size))
        testing_data = chunked_data #dask.delayed(chunked_data, nout=len(chunked_data))
#         testing_data = dask.delayed(list(chunked(testing_data, batch_size)))
        validate = dask.delayed(dask_validate_batch, nout=1)
        results = [validate(row,dataset_name) for row in testing_data]
        
        with ProgressBar():
            validated_results = dask.compute(*results)
        
        validated_data = pd.DataFrame(collapse(validated_results, levels=1))
        validated_data.columns = ['image_id','dataset','path','valid']
        validated_datasets = pd.concat([validated_datasets, validated_data], ignore_index=True)
        validated_data.to_csv(VALIDATION_RESULTS_PATH, mode='a', index=False)
    return validated_datasets

###################################################################

# def thorough_validate_image(image_path, image_id, file_ext_whitelist=['jpg','tif','tiff','png'], dataset_name='na'):
#     '''
#     Function for performing a set of checks on a single file specified by image_path.

#     Adds additional step of checking validity of image read from disk

#     Currently performs checks:
#         1) verify image_path refers to a valid file
#         2) verify that image_path ends with an extension that is included in file_ext_whitelist,
#            i.e. verify that image_path is a valid *image* file


#     Arguments:
#         image_path, str:
#             absolute path to image file to be validated
#         file_ext_whitelist, list(str):
#             List of allowable file extensions to consider valid
#         dataset_name, str:
#             Default: 'na' for not-applicable
#             optional str to reference dataset that file belongs to

#     Return:
#         validated, tuple(str,str,bool):
#             tuple that contains (dataset_name, image_path, True or False), where False indicates an invalid image.

#     '''
    
#     validated = [image_id, dataset_name, image_path, False]    
# #     if not os.path.isfile(image_path):
# #         print(f'FILE NOT FOUND: {image_path}')
# #         return validated
        
#     image_ext = os.path.splitext(image_path)[-1].strip('.')
#     if (imghdr.what(image_path) not in file_ext_whitelist) or (image_ext not in file_ext_whitelist):
#         print(f'file mislabeled. bytes: {imghdr.what(image_path)}, filename ext:{image_ext}')
#         return validated
        
#     try:
#         img = imageio.imread(image_path)
#         assert (type(img) in [imageio.core.util.Array, np.ndarray]), "invalid_image_type: "+str(type(img))
#         validated= [image_id, dataset_name, image_path, True]
#         print(image_id, validated[-1])
#     except:
#         print("Unexpected error:", sys.exc_info())
#         print('EXCEPTION WHILE READING image # ', image_id, ' at path: ',image_path)
#     finally:
#         return validated




# def test_db_validate_images(db_path=None, parallel=True):

#     ensure_dir_exists(os.path.dirname(VALIDATION_RESULTS_PATH))
#     if os.path.isfile(VALIDATION_RESULTS_PATH):
#         os.remove(VALIDATION_RESULTS_PATH)
    
#     if db_path is None:
#         db_path = leavesdb.db_utils.init_local_db()

#     db_URI = f'sqlite:///{db_path}'
#     db = dataset.connect(db_URI)

#     data = pd.DataFrame(db['dataset'].all())
#     data_by_dataset = data.groupby(by='dataset')
#     data_by_dataset_dict = {k:v for k,v in data_by_dataset}
    
#     file_ext_whitelist=['jpg','jpeg','tif','tiff','png']
#     mode='w'
#     validated_datasets = pd.DataFrame(columns=['dataset','path','valid'])
 
#     dataset_name='Leaves'
#     dataset_rows = data_by_dataset_dict['Leaves']
#     for dataset_name, dataset_rows in data_by_dataset:
#         if dataset_name == 'Fossil':
#             continue
#         print('validating dataset: ', dataset_name, ', shape=', dataset_rows.shape)

#         image_paths = list(dataset_rows['path'])
#         num_paths = len(image_paths)        
#         image_ids = list(range(num_paths))
        
#         testing_data = pd.DataFrame(list(zip(image_paths,image_ids)),columns=['image_path','image_id'])
        
#         if parallel:
#             print('running parallel with swifter')
# #             validated_data = testing_data.swifter.apply((lambda row: print(row)), raw=True, reduce=False, convert_dtype=False)
# #             validated_data = testing_data.swifter.progress_bar(False).allow_dask_on_strings(enable=True).apply(lambda row: thorough_validate_image(row['image_path'], row['image_id'], file_ext_whitelist, dataset_name), axis=1, broadcast=False)
#         else:
#             validated_data = []
#             image_paths = list(dataset_rows['path'])
#             num_paths = len(image_paths)
#             for i, image_path in enumerate(image_paths):
#                 result = thorough_validate_image(image_path, i,
#                                         file_ext_whitelist=file_ext_whitelist,
#                                         dataset_name=dataset_name)
            
#                 validated_data.append((i,*result))
#                 print(i, 'out of',num_paths, result[-1])
            
#         validated_data = validated_data.apply(pd.Series)
#         validated_data.columns = ['dataset','path','valid']
#         validated_datasets = pd.concat([validated_datasets, validated_data], ignore_index=True)
#         validated_data.to_csv(VALIDATION_RESULTS_PATH, mode=mode)
#         mode='a'
#     return validated_datasets
        
        
        
        
        
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--db_path', default='default', type=str,help="Path corresponding to leavesdb.db file to test. If 'default', then runs init_local_db() with no args.")
    args = parser.parse_args()
    
    if args.db_path == 'default':
        args.db_path=None
    
    print('RUNNING FULL TEST')
    

    test_db_validate_images(db_path=args.db_path) #parallel=False)#validation_func=validation_func)
        
        
        
        
        