'''
Functions for testing the validity of data entries in leavesdb.db

'''
import cv2
from PIL.Image import Image
import dataset
import imghdr
import numpy as np
import os
import pandas as pd
import swifter
from stuf import stuf

from pyleaves import leavesdb
from pyleaves.utils import ensure_dir_exists

VALIDATION_RESULTS_PATH = os.path.expanduser('~/tests/test_leavesdb_validation_results.csv')


def validate_image(image_path, file_ext_whitelist=['jpg','tif','tiff','png'], dataset_name='na'):
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
        dataset_name, str:
            Default: 'na' for not-applicable
            optional str to reference dataset that file belongs to

    Return:
        validated, tuple(str,str,bool):
            tuple that contains (dataset_name, image_path, True or False), where False indicates an invalid image.

    '''
    print('inside_validate_image')
    validated = (dataset_name, image_path, False)
    
    if not os.path.isfile(image_path):
        return validated

    for ext in file_ext_whitelist:
        if image_path.endswith(ext):
            validated=(dataset_name, image_path, True)
#             return validated

    with open(image_path,'rb'):
        im = Image.load(image_path)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close()
    validated=(dataset_name, image_path, True)
#     cv2.imread(image_path)
        
    if validated[-1]==False:
        print('Found invalid filepath: ', image_path)
    return validated



def thorough_validate_image(image_path, file_ext_whitelist=['jpg','tif','tiff','png'], dataset_name='na'):
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
#     print('inside thorough_validate_image')
    validated = [dataset_name, image_path, False]
#     print(image_path, dataset_name)
#     if True:
#         return validated

    if not os.path.isfile(image_path):
        print(validated)
        return validated
#     try:
    if True:
#         print('inside thorough_validate_image')
        validated = [dataset_name, image_path, False]
        if not os.path.isfile(image_path):
            print('not file')
            return validated

#         valid_ext=False
        
        image_ext = os.path.splitext(image_path)[-1].strip('.')
        
        if (image_ext not in file_ext_whitelist):
            print(f'ext = {image_ext}')
#             print('image_file = ', image_path)
#             print('(image_ext not in file_ext_whitelist) = ',(image_ext not in file_ext_whitelist))
            print('filename contains invalid ext')
            return validated
        
        if imghdr.what(image_path) not in file_ext_whitelist:
#             print('file_ext_whitelist = ',file_ext_whitelist)
            print(f'file mislabeled. bytes: {imghdr.what(image_path)}, filename ext:{image_ext}')
            return validated
        
                        
#         assert valid_ext, ' '.join(['invalid extension',ext,f' file: {image_path}'])
#         print('valid ', ext)
#         with open(image_path, 'rb') as f:
#             check_last2bytes = f.read()[-2:]
#         if check_last2bytes != b'\xff\xd9' :
#             print('image incomplete, last 2 bytes = ',check_last2bytes)
#             return validated
        
#         img = cv2.imread(image_path)
#         assert (type(img)==np.ndarray), "invalid_image_type"+str(type(img))
        validated= [dataset_name, image_path, True]
#         assert (validated[-1]==True), "for some reason invalidated"
#         print('valid')
        return validated

    
    
    
# def parallelize_dataframe(df, func, n_cores=4):
#     df_split = np.array_split(df, n_cores)
#     pool = Pool(n_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df






def test_db_validate_images(db_path=None, validation_func=validate_image, parallel=True):

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
    
    file_ext_whitelist=['jpg','jpeg','tif','tiff','png']
    mode='w'
    validated_datasets = pd.DataFrame(columns=['dataset','path','valid'])
    for dataset_name, dataset_rows in data_by_dataset:
        print('validating dataset: ', dataset_name, ', shape=', dataset_rows.shape)

        if parallel:
            print('running parallel with swifter')
            validated_data = dataset_rows['path'].swifter.apply(thorough_validate_image, #validation_func,
                                                                file_ext_whitelist=file_ext_whitelist,
                                                                dataset_name=dataset_name)
#             validated_data = dataset_rows['path'].swifter.allow_dask_on_strings().set_dask_scheduler(scheduler="processes").apply(thorough_validate_image, #validation_func,
#                                                                 file_ext_whitelist=file_ext_whitelist,
#                                                                 dataset_name=dataset_name)
        else:
            validated_data = dataset_rows['path'].apply(validation_func,
                                                        file_ext_whitelist=file_ext_whitelist,
                                                        dataset_name=dataset_name)
        
        validated_data = validated_data.apply(pd.Series)
        validated_data.columns = ['dataset','path','valid']
        
        validated_datasets = pd.concat([validated_datasets, validated_data], ignore_index=True)
#         validated_datasets.update({dataset_name:validated_data})

        validated_data.to_csv(VALIDATION_RESULTS_PATH, mode=mode)
        mode='a'
    return validated_datasets
        
        
        
        
        
if __name__ == "__main__":
    
    run_full_test = True
    
    
    if run_full_test:
        print('RUNNING FULL TEST')
        validation_func = thorough_validate_image
    else:
        print('NOT RUNNING FULL TEST')
        validation_func = validate_image
    
#     %time 
    test_db_validate_images() #parallel=False)#validation_func=validation_func)
        
        
        
        
        