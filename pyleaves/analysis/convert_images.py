'''

Command line script for Batch converting image file formats and copying to new location

Primarily used for standardizing all images into JPG format for future loading into tf.data.Dataset


>> python ./pyleaves/pyleaves/analysis/convert_images.py -t /media/data/jacob/Fossil_Project/src_data

'''
import argparse
import numpy as np
import pandas as pd
import dataset
from functools import partial
import os
import sys
from stuf import stuf
import time

from pyleaves.leavesdb.db_manager import dict2json, build_db_from_json, clear_duplicates_from_db
from pyleaves.utils import ensure_dir_exists
from pyleaves.analysis.img_utils import convert_to_png, convert_to_jpg, JPGCoder, DaskCoder, CorruptJPEGError
from pyleaves import leavesdb

join = os.path.join
splitext = os.path.splitext
basename = os.path.basename


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_dir', type=str, default=r'/media/data_cifs/jacob/Fossil_Project/opt_data', help='Location for saving converted image files')
    parser.add_argument('-ext', '--target_ext', type=str, default='jpg', help="Image format to save all image copies as. Must be either 'png' or 'jpg'.")
    parser.add_argument('-prefix', '--json_prefix', type=str, default=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources', help='Location for saving newly created json files containing paths of converted files + all previous data')
    parser.add_argument('-json_fname', '--json_filename', type=str, default='database_json_records_JPG_format.json', help='Filename for saving newly created json files containing paths of converted files + all previous data')
    parser.add_argument('-db', '--db_path', type=str, default='/home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db', help='Filepath of source db to use for locating images')

    args = parser.parse_args()

#     class Args:
#         target_dir=r'/media/data_cifs/jacob/Fossil_Project/opt_data'
#         target_ext = 'jpg'
#         json_prefix=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources'
#         json_filename=r'database_json_records_PNG_format.json'
#         db_path=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources/leavesdb.db'
#     args = Args()

    def get_converted_image_name(row, output_dir, output_format='jpg'):
        assert output_format in ['jpg','png']
        
        filepath = row.loc['path']
        label = row.loc['label']

        filename, file_ext = splitext(basename(filepath))
        output_filepath = join(output_dir, label, filename + '.' + output_format)
        row['source_path'] = filepath
        if filepath != output_filepath:
            row['path'] = output_filepath
        return row
    
    #LOAD DataFrame from default db and return only non-duplicated file paths.
    data = clear_duplicates_from_db(db=None, local_db=args.db_path, prefix = args.json_prefix, json_filename='database_records.json')

    data_by_dataset = data.groupby(by='dataset')
    data_by_dataset_dict = {k:v for k,v in data_by_dataset} # if k not in ['Fossil']}
    
    data_records = []
    new_data_location_info = {}
    dataset_name='Leaves'; rows = data_by_dataset_dict['Leaves']
    if True:
#     for dataset_name, rows in data_by_dataset_dict.items():
        
        output_dir=join(args.target_dir,dataset_name)
        ensure_dir_exists(output_dir)
        
        if 'source_path' in rows.columns:
            rows['path'] = rows['source_path']
        
        get_converted_image_name = partial(get_converted_image_name, output_dir=output_dir, output_format=args.target_ext)
        rows.loc[:,'label'] = rows.loc[:,'family']
        data_df = rows.apply(get_converted_image_name,axis=1)
        data_records.extend(data_df.to_dict('records'))
        
        num_files = len(rows)
        print(f'[BEGINNING] copying {num_files} from {dataset_name}')    
        start_time = time.perf_counter()
        try:
            if args.target_ext == 'jpg':
#                 coder = JPGCoder(data_df, output_dir)
#                 new_dataset_paths = coder.batch_convert()
                coder = DaskCoder(data_df, output_dir)
                new_dataset_paths = coder.execute_conversion(coder.input_dataset)

            new_dataset_paths = list(new_dataset_paths)
            end_time = time.perf_counter()
            total_time = end_time-start_time
            new_data_location_info.update({dataset_name:{'data':new_dataset_paths, 'total_time':total_time, 'conversion_rate':num_files/total_time}})
            print(f'[FINISHED] copying {num_files} from {dataset_name}')
        except CorruptJPEGError:
            print(CorruptJPEGError.corrupted_files)
        except Exception as e:
            print(type(e))
            print('[EXCEPTION] ', e)
            sys.exit(0)

    print('[FINISHED] All Corrupted files->')
    print(CorruptJPEGError.corrupted_files)
    db_path = join(args.json_prefix,'leavesdb.db')
    db_json_path = join(args.json_prefix,args.json_filename)

    #Create JSON records file containing previous file info combined with new file paths
    dict2json(data_records, prefix=args.json_prefix, filename=args.json_filename)
    
    #Create new SQLite .db file from newly created JSON
    build_db_from_json(frozen_json_filepaths=[db_json_path], db_path=db_path)
        
    with open('file_conversion_log.txt', 'a') as log_file:
        log_file.write('Database: '+db_path+'\n')
        log_file.write('db_json_path: '+db_json_path+'\n')
        for dataset_name, log_info in new_data_location_info.items():
            log_file.write(dataset_name+'\n')
            log_file.write(str({f'{str(k)} : {str(v)}\n' for k,v in log_info.items()})+'\n')
                                   
                                   
                                   
if __name__ == '__main__':
    
    main()