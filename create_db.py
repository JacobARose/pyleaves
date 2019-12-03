from  pyleaves.leavesdb.db_manager import create_db
import argparse
import os 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='/media/data_cifs/irodri15/data/processed/datasets.json', type=str,help='Db json file')
    parser.add_argument('--output_folder', default='resources' ,type=str,help='Folder Where to save the output')
    args = parser.parse_args()
    json_path = args.json_path
    db_path = os.path.abspath(args.output_folder)
    print('Creating folder to save db')
    os.makedirs(db_path,exist_ok=True)
    
    create_db(json_path,db_path)