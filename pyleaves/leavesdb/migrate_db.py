# @Author: Jacob A Rose
# @Date:   Mon, May 4th 2020, 8:36 pm
# @Email:  jacobrose@brown.edu
# @Filename: migrate_db.py
'''

Functions for duplicating, archiving, and converting database assets, including raw source files as well as SQLite db files.
'''



from tqdm import tqdm
from os.path import isfile
import shutil
import pandas as pd
import time

from pyleaves.utils.img_utils import DaskCoder, CorruptJPEGError




def duplicate_raw_dataset(data :pd.DataFrame, omitted_rows: list=[]):
    """
    Uses shutil.copy2 to duplicate a sequence of files at a new location to be as close to the original files as possible.

    # TODO Check if file hash remains the same, if so, then can be used as a test for successful duplication.

    Parameters
    ----------
    data : pd.DataFrame
        Contains all necessary info to duplicate the files. For each file to be duplicated, must have the corresponding file paths in the
        source_path and target_path columns, respectively. The choice of how to determine the best target path must be made prior to this function.
    omitted_rows : list
        Optional list of omitted sample rows. Unsuccessful copy attempts will be logged here and returned from function.

    Returns
    -------
    pd.DataFrame
        duplicated_data: Same format as input data, only contains successful samples.
    list
        omitted_rows: list of dataframes, containing unsuccessful rows.

    """
    data = data.copy()

    file_not_found = []
    copy_errors = []
    for i, row in tqdm(data.iterrows()):
        try:
            if isfile(row.target_path):
                continue
            shutil.copy2(row.source_path, row.target_path)
            assert isfile(row.target_path)
        except FileNotFoundError as e:
            print(str(e))
            file_not_found.append(row)
            print(f'total {len(file_not_found)+len(copy_errors)} files not found so far')
        except AssertionError as e:
            print(str(e))
            copy_errors.append(row)
            print(f'total {len(file_not_found)+len(copy_errors)} files not found so far')

    if len(file_not_found):
        file_not_found_df = pd.concat(file_not_found,axis=1).T
        data = data[~data.index.isin(file_not_found_df.index)]
        omitted_rows.append({'data':file_not_found_df,
                             'reason_omitted':'source file not found'})

    if len(copy_errors):
        copy_errors_df = pd.concat(copy_errors,axis=1).T
        data = data[~data.index.isin(copy_errors_df.index)]
        omitted_rows.append({'data':copy_errors_df,
                             'reason_omitted':'Copying original raw file unsuccessful'})

    return data, omitted_rows


def process_raw_dataset2jpg(data : pd.DataFrame,  target_absolute_root : str, dataset_name : str, log_path : str=None):
    """
    Uses DaskCoder to iterate through a sequence of files and duplicate/convert each file into JPEG format

    Parameters
    ----------
    data : pd.DataFrame
        Contains all necessary info to process the files. For each file to be processed, must have the corresponding file paths in the
        source_path and converted_target_path columns, respectively.
    target_absolute_root : str
        Location of the root dir towards which files will be directed. Assumes they will be organized in subdirectories by family name.
    dataset_name : str
        Name of the dataset to be recorded in the 'dataset' column of the database

    Returns
    -------
    pd.DataFrame
        data: List of successful file paths. # TODO Reformat to output same format as input data, only containing successful samples.
    list
        corrupt_jpegs: list of file paths referencing unsuccessful rows.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>



    """

    data = data.assign(label=data.family, dataset = dataset_name)
    # data.dataset = dataset_name
    num_files = len(data)

    print(f'[BEGINNING] copying {num_files} from {dataset_name}')
    start_time = time.perf_counter()

    coder = DaskCoder(data,
                      target_absolute_root,
                      columns={'source_path':'source_path',
                               'target_path':'converted_target_path',
                               'label':'family'})
    data = coder.execute_conversion(coder.input_dataset)

    num_files_kept = len(data)
    print(f'[FINISHED] copying {num_files_kept} from {dataset_name} in {time.perf_counter()-start_time:.2f} sec, originally started with {num_files}')

    #TRACK AND LOG FAILED FILE PATHS
    corrupt_jpegs = CorruptJPEGError.export_log(filepath=log_path)
    print(f'{len(corrupt_jpegs)} corrupt_jpegs')

    return data, corrupt_jpegs



    # new_dataset_paths = list(new_dataset_paths)
    # num_files_kept = len(new_dataset_paths)
    # print(f'[FINISHED] copying {num_files_kept} from {dataset_name} in {time.perf_counter()-start_time:.2f} sec, originally started with {num_files}')
    #
    # #TRACK AND LOG FAILED FILE PATHS
    # failed_paths = list(set(CorruptJPEGError.get_failed_paths()))
    # print(f'{len(failed_paths)} FAILED PATHS')
    #
    # # TODO Must make CorruptJPEGError format conform to dataframe columns
    # if len(failed_paths):
    #     corrupt_jpegs = pd.DataFrame(failed_paths, axis=1).T
    #     print(f'total of {len(corrupt_jpegs)} corrupted jpegs:')
    #     corrupt_jpegs
    # return new_dataset_paths, corrupt_jpegs
