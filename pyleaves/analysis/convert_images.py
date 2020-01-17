'''

Command line script for Batch converting image file formats and copying to new location

Primarily used for standardizing all images into PNG format for future loading into tf.data.Dataset


>> python ./pyleaves/pyleaves/analysis/convert_images.py -t /media/data/jacob/Fossil_Project/src_data

'''
import argparse
import dataset
from stuf import stuf
import time

from pyleaves.analysis.img_utils import convert_to_png
from pyleaves import leavesdb


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_dir', type=str, default=r'/media/data/jacob/Fossil_Project', help='Location for saving converted image files')

    args = parser.parse_args()
    
    local_db = leavesdb.init_local_db()
    db = dataset.connect(f'sqlite:///{local_db}', row_type=stuf)
    data = leavesdb.db_query.load_all_data(db)

    data_by_dataset = data.groupby(by='dataset')
    data_by_dataset_dict = {k:v for k,v in data_by_dataset}
    
    
    new_data_location_info = {}
    for dataset_name, rows in data_by_dataset_dict.items():
        
        filepaths = list(rows['path'].values)[:100]
        labels = list(rows['family'].values)[:100]
        assert len(filepaths) == len(labels)
        print('Starting ', dataset_name, ' with ', len(filepaths), ' files')

        num_files = len(filepaths)
        start_time = time.perf_counter()
        new_dataset_paths = convert_to_png(filepaths, labels, dataset_name = dataset_name, target_dir = args.target_dir)
        end_time = time.perf_counter()
        total_time = end_time-start_time
        print(f'Finished copying {num_files} from {dataset_name} in {total_time:.3f} seconds at a rate of {num_files/total_time:.3f} images/sec')
        
        new_dataset_paths = list(new_dataset_paths)
        new_data_location_info.update({dataset_name:{'data':new_dataset_paths, 'total_time':total_time}})
        
        with open('file_conversion_log.txt', 'a') as log_file:
            for dataset_name, log_info in new_data_location_info.items():
                log_file.write(dataset_name+'\n')
                log_file.write(str({f'{str(k)} : {str(v)}\n' for k,v in log_info.items()})+'\n')
                                   
                                   
                                   
                                   
if __name__ == '__main__':
    
    main()