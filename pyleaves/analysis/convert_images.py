'''

Command line script for Batch converting image file formats and copying to new location

Primarily used for standardizing all images into PNG format for future loading into tf.data.Dataset


>> python ./pyleaves/pyleaves/analysis/convert_images.py -t /media/data/jacob/Fossil_Project/src_data

'''
import argparse
import pandas as pd
import dataset
from functools import partial
import os
from stuf import stuf
import time

from pyleaves.leavesdb.db_manager import dict2json, build_db_from_json
from pyleaves.utils import ensure_dir_exists
from pyleaves.analysis.img_utils import convert_to_png
from pyleaves import leavesdb

join = os.path.join
splitext = os.path.splitext
basename = os.path.basename



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_dir', type=str, default=r'/media/data/jacob/Fossil_Project/src_data', help='Location for saving converted image files')
    parser.add_argument('-prefix', '--json_prefix', type=str, default=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources', help='Location for saving newly created json files containing paths of converted files + all previous data')
    parser.add_argument('-json_fname', '--json_filename', type=str, default='database_json_records_PNG_format.json', help='Location for saving newly created json files containing paths of converted files + all previous data')

    args = parser.parse_args()

#     class Args:
#         target_dir=r'/media/data/jacob/Fossil_Project/src_data'
#         json_prefix=r'/home/jacob/pyleaves/pyleaves/leavesdb/resources'
#         json_filename=r'database_json_records_PNG_format.json'
#     args = Args()

    def get_converted_image_name(row, output_dir):
        filepath = row.loc['path']
        label = row.loc['label']

        filename, file_ext = splitext(basename(filepath))
        output_filepath = join(output_dir, label, filename+'.png')
        row['source_path'] = filepath
        if filepath != output_filepath:
            row['path'] = output_filepath
        return row

    
    
    local_db = leavesdb.init_local_db()
    db = dataset.connect(f"sqlite:///{local_db}", row_type=stuf)
#     data = leavesdb.db_query.load_all_data(db)
    data = pd.DataFrame(db['dataset'].all())

    data_by_dataset = data.groupby(by='dataset')
    data_by_dataset_dict = {k:v for k,v in data_by_dataset}
    
    data_records = []
    new_data_location_info = {}
    for dataset_name, rows in data_by_dataset_dict.items():
        
        output_dir=join(args.target_dir,dataset_name)
        ensure_dir_exists(output_dir)
        
        get_converted_image_name = partial(get_converted_image_name, output_dir=output_dir)
        rows['label'] = rows.loc[:,'family']
        data_df = rows.apply(get_converted_image_name,axis=1)

        data_records.extend(data_df.to_dict('records'))
        
#         break        
        num_files = len(rows)
        print(f'[BEGINNING] copying {num_files} from {dataset_name}')    
        start_time = time.perf_counter()
        try:
            new_dataset_paths = convert_to_png(data_df, output_dir=output_dir)
            new_dataset_paths = list(new_dataset_paths)
            end_time = time.perf_counter()
            total_time = end_time-start_time
            new_data_location_info.update({dataset_name:{'data':new_dataset_paths, 'total_time':total_time, 'conversion_rate':num_files/total_time}})
            print(f'[FINISHED] copying {num_files} from {dataset_name}')
        except Exception as e:
            print('[EXCEPTION] ', e)
            print('dir(e) = ', dir(e))

    db_path = join(args.json_prefix,'leavesdb.db')
    db_json_path = join(args.json_prefix,args.json_filename)
        
    dict2json(data_records, prefix=args.json_prefix, filename=db_json_path)
        
    build_db_from_json(frozen_json_filepaths=[db_json_path], db_path=db_path)
        
    
    with open('file_conversion_log.txt', 'a') as log_file:
        log_file.write('Database: '+db_path+'\n')
        log_file.write('db_json_path: '+db_json_path+'\n')
        
        for dataset_name, log_info in new_data_location_info.items():
            log_file.write(dataset_name+'\n')
            log_file.write(str({f'{str(k)} : {str(v)}\n' for k,v in log_info.items()})+'\n')
                                   
                                   
                                   
if __name__ == '__main__':
    
    main()