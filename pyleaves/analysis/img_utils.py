'''
Functions for managing images
'''

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2
import dataset
import matplotlib.pyplot as plt
from more_itertools import chunked
import numpy as np
import os
import pandas as pd
from pathos.multiprocessing import ProcessPool
from pathos.threading import ThreadPool
import sys
from stuf import stuf
from threading import Lock


import time

import tensorflow as tf
tf.compat.v1.enable_eager_execution()


AUTOTUNE = tf.data.experimental.AUTOTUNE

join = os.path.join
splitext = os.path.splitext
basename = os.path.basename

from pyleaves.utils import ensure_dir_exists
from pyleaves.config import DatasetConfig
# from pyleaves import leavesdb


def get_dataset_from_list(sample_list):
    '''
    
    '''
    samples = tf.data.Dataset.from_tensor_slices(sample_list)
    samples = samples.prefetch(1)
    return samples

# def filter_tiff(sample_filepaths):
#     filtered_dataset = sample_filepaths.map(lambda x: tf.strings.regex_full_match(x,'(.*?)\.(tif)')) #'*tif'))
    
#     return filtered_dataset

def copy_img2png(src_filepath, target_filepath, label):
    
    img = tf.io.read_file(src_filepath)
    img = tf.image.decode_image(img, channels=3)
    
    img = tf.image.encode_png(img)
    tf.io.write_file(target_filepath, img)
    return target_filepath, label
    
def copy_tiff2png(src_filepath, target_filepath, label):
    
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
    try:
        img = cv2.imread(src_filepath.decode('utf-8'), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise
    try:
        cv2.imwrite(target_filepath.decode('utf-8'), img, compression_params)
        return target_filepath, label
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise

##############################################################################

def copy_img2jpg(src_filepath, target_filepath, label):
    
    img = tf.io.read_file(src_filepath)
    img = tf.image.decode_image(img, channels=3)
    
    img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)
    tf.io.write_file(target_filepath, img)
    return target_filepath, label
    
def copy_tiff2jpg(src_filepath, target_filepath, label):
    
    try:
        img = cv2.imread(src_filepath.decode('utf-8'), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise
    try:
        cv2.imwrite(target_filepath.decode('utf-8'), img)
        return target_filepath, label
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'failed reading {src_filepath}, [ERROR] {e}')
        raise


##############################################################################
def time_ds(ds):
    start_time = time.perf_counter()
    failed_files =[]
    for i, sample in enumerate(ds):
        print(i)
        if sample==0:
            failed_files.append(i)
    end_time = time.perf_counter()
    run_time = end_time-start_time
    print(f'converted {i} files in {run_time} seconds, at rate {i/run_time} images/sec')
    return failed_files
##############################################################################



# config = DatasetConfig(dataset_name=dataset_name,
#                                target_size=(224,224),
#                                low_class_count_thresh=low_count_thresh,
#                                data_splits={'val_size':val_size,'test_size':test_size},
#                                tfrecord_root_dir=output_dir,
#                                num_shards=anum_shards)
# dataset_name = 'Fossil'
# y_col = 'family'
# local_db = leavesdb.init_local_db()
# db = dataset.connect(f'sqlite:///{local_db}', row_type=stuf)
# db_df = pd.DataFrame(leavesdb.db_query.load_data(db, y_col=y_col, dataset=dataset_name))

# #     sample_dataset = sample_dataset.map(lambda x: tf.string_split([x],sep='.'))
# #     sample_iter = iter(sample_dataset)

# #     sample = next(sample_iter)
# db_df = data_df
# src_paths = get_dataset_from_list(sample_list=list(db_df['source_path']))
# src_labels = get_dataset_from_list(sample_list=list(db_df['label']))
# target_paths = get_dataset_from_list(sample_list=list(db_df['path']))
# mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

# copied_dataset = mappings_dataset.map(lambda src, target, label: create_sample_copy(src, target, label), num_parallel_calls=AUTOTUNE)
# tiff_reader = lambda src_filepath, target_filepath, label: tf.py_func(create_tiff_sample_copy, [src_filepath, target_filepath, label],[tf.string, tf.string]) #, tf.int64)
# tiff_results = mappings_dataset.map(tiff_reader, num_parallel_calls=AUTOTUNE)
# for sample in results.take(1):
#     print(sample)
# for sample in mappings_dataset.take(1):
#     print(sample)
# time_ds(tiff_results)
##############################################################################
##############################################################################

class Coder:
    def __init__(self, data, output_dir):
        '''
        Class for managing different conversion functions depending on source and target image formats.
        '''
        self.output_ext = 'jpg'
        
        labels = set(list(data['label']))
        [ensure_dir_exists(join(output_dir,label)) for label in labels]
        
        self.indices = {
                        'tiff':np.where(data['source_path'].str.endswith('.tif')),
                        'non_tiff':np.where(~(data['source_path'].str.endswith('.tif')))
                       }
        
        self.data = {
                    'tiff':data[data['source_path'].str.endswith('.tif')],
                    'non_tiff':data[~(data['source_path'].str.endswith('.tif'))]
                    }
        
        
    def stage_dataset(self, data_df):
        src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
        src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
        target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
        mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels)).cache()
        return mappings_dataset
    
    def stage_converter(self, data_df, from_tiff=False): #, output_ext='jpg'):
        '''
        Arguments:
            data_df, pd.DataFrame:
                DataFrame containing columns ['path','sourcepath','label'] for image conversion
            from_tiff, bool:
                Indicates whether to use converters for converting from tiff images, or to use generic image converters for converting from jpg or png
            output_ext, bool:
                Default='jpg'
                Must be either 'jpg' or 'png'
        Return:
            converted_dataset, tf.data.Dataset:
                Dataset that, when iterated over, will read and write images referenced in data_df
        
        '''
        print('tiff shape = ', data_df.shape)
        print(type(data_df))
        
        output_ext = self.output_ext
        mappings_dataset = self.stage_dataset(data_df)
        
        if from_tiff:
            if output_ext=='jpg':
                img_reader = lambda src, target, label: tf.py_func(copy_tiff2jpg, [src, target, label],[tf.string, tf.string])
            elif output_ext=='png':
                img_reader = lambda src, target, label: tf.py_func(copy_tiff2png, [src, target, label],[tf.string, tf.string])        
        else:
            if output_ext=='jpg':
                img_reader = lambda src, target, label: copy_img2jpg(src, target, label)
            elif output_ext=='png':
                img_reader = lambda src, target, label: copy_img2png(src, target, label)
        
        converted_dataset = mappings_dataset.map(img_reader, num_parallel_calls=AUTOTUNE)
        return converted_dataset.prefetch(AUTOTUNE)

    def execute_conversion(self, input_dataset):
        '''
        Arguments:
            input_dataset, tf.data.Dataset:
                Dataset that has been output from the stage_converter() method, but not yet iterated through.
        Return:
            output, list:
                List of whatever information is returned by input_dataset, specified in the specific coding/conversion function
        '''
        perf_counter = time.perf_counter
        output=[]
        indices = [0]
        timelog=[perf_counter()]
        j=0
        for i, converted_data in enumerate(input_dataset):
            output.append(converted_data)
            if i%20==0:
                indices.append(i+1)
                timelog.append(perf_counter())
                idx = (indices[j],indices[j+1])
                print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
                j+=1
        return output

    
class JPGCoder(Coder):
    
    def __init__(self, data, output_dir):
        super().__init__(data, output_dir)
        
        self.output_ext='jpg'
        
    def batch_convert(self):
        
        outputs = []
                
        try:
            if self.data['non_tiff'].shape[0]>0:
                print(f"converting {self.data['non_tiff'].shape[0]} non-tiff images to jpg")
                non_tiff_staged = self.stage_converter(data_df=self.data['non_tiff'],from_tiff=False)
                outputs.extend(self.execute_conversion(non_tiff_staged))
                
            if self.data['tiff'].shape[0]>0:
                print(f"converting {self.data['tiff'].shape[0]} tiff images to jpg")
                tiff_staged = self.stage_converter(data_df=self.data['tiff'],from_tiff=True)
                outputs.extend(self.execute_conversion(tiff_staged))
            return outputs

        except Exception as e:
            print("Unexpected error:", sys.exc_info())
            print(f'[ERROR] {e}')
            raise        
        

        
        
        
        
    
def batch_convert_to_jpg(data_df, output_dir):
    '''
    Function to load a list of image files, convert to jpg format if necessary, and save to specified target dir.
    
    Arguments:
        dataset_name, str:
            Name of source dataset from which images are sourced, to be name of subdir in target root dir
        target_dir, str:
            Root directory for converted images, which will be saved in hierarchy:  
            root/
                |dataset_1/
                          |class_1/
                                  |image_1
                                  |image_2
                                  ...
    
    Return:
    
    '''
    
    labels = set(list(data_df['label']))
    [ensure_dir_exists(join(output_dir,label)) for label in labels]
    indices = list(range(len(data_df)))
    
    tiff_data = data_df[data_df['source_path'].str.endswith('.tif')]
    non_tiff_data = data_df[~(data_df['source_path'].str.endswith('.tif'))]

    
    coder = Coder()
    
    
    
    outputs = []
    try:
        if non_tiff_data.shape[0]>0:
            print(f'converting {non_tiff_data.shape[0]} non-tiff images to jpg')
            outputs.extend(convert_from_non_tiff2jpg(non_tiff_data))
        if tiff_data.shape[0]>0:
            print(f'converting {tiff_data.shape[0]} tiff images to jpg')
            outputs.extend(convert_from_tiff2jpg(tiff_data))
        return outputs
    
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'[ERROR] {e}')
        raise        
    
    
    
    
    
    
    
##############################################################################
    
def convert_from_nontiff2png(data_df):
    '''
    data_df must only contain filenames referring to non-TIFF formatted images (e.g. PNG, JPG, GIF) in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    converted_dataset = mappings_dataset.map(lambda src, target, label: copy_img2png(src, target, label), num_parallel_calls=AUTOTUNE)
    
    perf_counter = time.perf_counter
    
    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1
    
    return output
    
    
def convert_from_tiff2png(data_df):
    '''
    data_df must only contain filenames referring to TIFF formatted images in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    tiff_reader = lambda src_filepath, target_filepath, label: tf.py_func(copy_tiff2png, [src_filepath, target_filepath, label],[tf.string, tf.string])
    converted_dataset = mappings_dataset.map(tiff_reader, num_parallel_calls=AUTOTUNE)
    perf_counter = time.perf_counter
    
    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1
    return output 


def convert_from_nontiff2jpg(data_df):
    '''
    data_df must only contain filenames referring to non-TIFF formatted images (e.g. PNG, JPG, GIF) in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    converted_dataset = mappings_dataset.map(lambda src, target, label: copy_img2jpg(src, target, label), num_parallel_calls=AUTOTUNE)
    
    perf_counter = time.perf_counter
    
    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1
    
    return output
    
    
def convert_from_tiff2jpg(data_df):
    '''
    data_df must only contain filenames referring to TIFF formatted images in column 'source_path'
    '''
    src_paths = get_dataset_from_list(sample_list=list(data_df['source_path']))
    src_labels = get_dataset_from_list(sample_list=list(data_df['label']))
    target_paths = get_dataset_from_list(sample_list=list(data_df['path']))
    mappings_dataset = tf.data.Dataset.zip((src_paths, target_paths, src_labels))

    tiff_reader = lambda src_filepath, target_filepath, label: tf.py_func(copy_tiff2jpg, [src_filepath, target_filepath, label],[tf.string, tf.string])
    converted_dataset = mappings_dataset.map(tiff_reader, num_parallel_calls=AUTOTUNE)
    perf_counter = time.perf_counter
    
    output=[]
    indices = [0]
    timelog=[perf_counter()]
    j=0
    for i, converted_data in enumerate(converted_dataset):
        output.append(converted_data)
        if i%20==0:
            indices.append(i+1)
            timelog.append(perf_counter())
            idx = (indices[j],indices[j+1])
            print(f'{i+1} images at rate {((idx[1]-idx[0])/(timelog[j+1] - timelog[j])):.2f} images/second')
            j+=1
    return output 


##############################################################################

def convert_to_png(data_df, output_dir):
    '''
    Function to load a list of image files, convert to png format if necessary, and save to specified target dir.
    
    Arguments:
        dataset_name, str:
            Name of source dataset from which images are sourced, to be name of subdir in target root dir
        target_dir, str:
            Root directory for converted images, which will be saved in hierarchy:  
            root/
                |dataset_1/
                          |class_1/
                                  |image_1
                                  |image_2
                                  ...
    
    Return:
    
    '''
    
    labels = set(list(data_df['label']))
    [ensure_dir_exists(join(output_dir,label)) for label in labels]
    indices = list(range(len(data_df)))
    
    tiff_data = data_df[data_df['source_path'].str.endswith('.tif')]
    non_tiff_data = data_df[~(data_df['source_path'].str.endswith('.tif'))]

    outputs = []
    try:
        if non_tiff_data.shape[0]>0:
            print(f'converting {non_tiff_data.shape[0]} non-tiff images to png')
            outputs.extend(convert_from_nontiff2png(non_tiff_data))
        if tiff_data.shape[0]>0:
            print(f'converting {tiff_data.shape[0]} tiff images to png')
            outputs.extend(convert_from_tiff2png(tiff_data))
        return outputs
    
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'[ERROR] {e}')
        raise
        
        
def convert_to_jpg(data_df, output_dir):
    '''
    Function to load a list of image files, convert to jpg format if necessary, and save to specified target dir.
    
    Arguments:
        dataset_name, str:
            Name of source dataset from which images are sourced, to be name of subdir in target root dir
        target_dir, str:
            Root directory for converted images, which will be saved in hierarchy:  
            root/
                |dataset_1/
                          |class_1/
                                  |image_1
                                  |image_2
                                  ...
    
    Return:
    
    '''
    
    labels = set(list(data_df['label']))
    [ensure_dir_exists(join(output_dir,label)) for label in labels]
    indices = list(range(len(data_df)))
    
    tiff_data = data_df[data_df['source_path'].str.endswith('.tif')]
    non_tiff_data = data_df[~(data_df['source_path'].str.endswith('.tif'))]

    outputs = []
    try:
        if non_tiff_data.shape[0]>0:
            print(f'converting {non_tiff_data.shape[0]} non-tiff images to jpg')
            outputs.extend(convert_from_non_tiff2jpg(non_tiff_data))
        if tiff_data.shape[0]>0:
            print(f'converting {tiff_data.shape[0]} tiff images to jpg')
            outputs.extend(convert_from_tiff2jpg(tiff_data))
        return outputs
    
    except Exception as e:
        print("Unexpected error:", sys.exc_info())
        print(f'[ERROR] {e}')
        raise        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


# def convert_to_png(data_df, output_dir):
#     '''
#     Function to load a list of image files, convert to png format if necessary, and save to specified target dir.
    
#     Arguments:
#         dataset_name, str:
#             Name of source dataset from which images are sourced, to be name of subdir in target root dir
#         target_dir, str:
#             Root directory for converted images, which will be saved in hierarchy:  
#             root/
#                 |dataset_1/
#                           |class_1/
#                                   |image_1
#                                   |image_2
#                                   ...
    
    
#     Return:
    
#     '''
    
# #     output_dir = join(target_dir,dataset_name)
# #     ensure_dir_exists(output_dir)
#     labels = set(list(data_df['label']))
#     [ensure_dir_exists(join(output_dir,label)) for label in labels]
    
#     def parse_function(row):
#         src_filepath = row.source_path #.loc[:,'source_path']
#         target_filepath = row.path #.loc[:,'path']
#         label = row.label #loc[:,'label']
#         index = row.id
#         compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
#         print('Converting image ',index)
        
#         if not os.path.isfile(target_filepath):
#             img = cv2.imread(src_filepath, cv2.IMREAD_UNCHANGED)
#             cv2.imwrite(target_filepath, img, compression_params)
#             print(f'Converted image {index} and saved at {target_filepath}')
#         else:
#             print(f'image {index} already exists in converted form at: {target_filepath}')
#         return row
    
#     def parse_batch(batch):
#         print('starting batch inside')
#         print(f'batch length {len(batch)}')
# #         print('batch[0] = ',batch[0])
# #         return len(batch)
#         lock = Lock()
# #         with ThreadPoolExecutor(max_workers=12) as executor:
#         with ThreadPool(nodes=4) as executor:
#             with lock:
#                 _map = executor.imap
#                 return list(_map(parse_function, batch))
    
#     indices = list(range(len(data_df)))
#     num_workers = 1 #os.cpu_count()#//2
#     chunksize = len(indices)//num_workers
#     data = chunked([stuf(row) for _, row in data_df.iterrows()],chunksize)

#     try:

#         outputs = []
#     #     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#     #     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         with ProcessPool(nodes=num_workers) as executor:

#             outputs.extend(list(executor.map(parse_batch, data)))
#             print('done')
#         return outputs
    
#     except:
#         raise len(outputs)  


def plot_image_grid(imgs, labels = np.array([]), x_plots = 4, y_plots = 4, figsize=(15,15)):
	fig, axes = plt.subplots(y_plots, x_plots, figsize=figsize)
	axes = axes.flatten()

	num_imgs = len(imgs)
	
	if len(axes) > num_imgs:
		axes = axes[:num_imgs]
	for i, ax in enumerate(axes):
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)

		ax.imshow(imgs[i,...])
		if len(labels) >= i:
			ax.set_title(labels[i])
	plt.tight_layout()
    
    
def pad_image(img, target_size, interpolation=cv2.INTER_CUBIC):
    old_size = img.shape[:2]
    ratio = np.min(np.array(target_size)/old_size)
    new_size = tuple(np.int16(np.array(old_size)*ratio))
    
    img = cv2.resize(img,tuple(new_size)[::-1],interpolation=cv2.INTER_CUBIC)
        
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=[0, 0, 0])
    return img
    
def resize(img, target_size, padding = True, interpolation=cv2.INTER_CUBIC):
    '''
    Resize function with option to pad borders to avoid warping image aspect ratio
    '''
    if padding == True:
        img = pad_image(img, tuple(target_size), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, tuple(target_size), interpolation=cv2.INTER_CUBIC)    
    return img


def load_image(filepath, target_size=(224,224)):
    '''
    Read an image stored at filepath, and resize to target_size.
    Written to default resize with 0 padding in order to conserve aspect ratio. If alternative
    resizing or padding is desired the function can be easily refactored.

    '''
    try:
#         print(filepath)
        img = cv2.imread(filepath)
    except:
        print('[error:] ', filepath)
        return None
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    
    return img
