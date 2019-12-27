from pyleaves.leavesdb.db_manager import create_db, build_db_from_json
import argparse
import os
from os.path import join, abspath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default=r'pyleaves/leavesdb/resources/full_dataset_frozen.json', type=str,help='Db json file')
    parser.add_argument('--output_folder', default=r'pyleaves/leavesdb/resources' ,type=str,help='Folder Where to save the output')
    args = parser.parse_args()
    json_path = abspath(args.json_path)
    db_dir = abspath(args.output_folder)
    print('Creating folder to save db')
    os.makedirs(db_dir,exist_ok=True)
    
    db_path = join(db_dir,'leavesdb.db')


    build_db_from_json(frozen_json_filepaths=[json_path],
                       db_path=db_path)