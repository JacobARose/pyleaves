'''
Functions for managing images
'''

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2
import matplotlib.pyplot as plt
from more_itertools import chunked
import numpy as np
import os
from pathos.multiprocessing import ProcessPool

join = os.path.join
splitext = os.path.splitext
basename = os.path.basename

from pyleaves.utils import ensure_dir_exists

# def parse_batch(batch):
#     print('outside convert function')
#     print('starting batch')
#     print(f'batch length {len(batch)}')
#     return len(batch)


def convert_to_png(image_filepaths, labels, dataset_name, target_dir=r'/media/data/jacob/Fossil_Project'):
    '''
    Function to load a list of image files, convert to png format if necessary, and save to specified target dir.
    
    Arguments:
        image_filepaths, list(str):
            List of absolute file paths for images to be copied/converted
        labels, list(str):
            Corresponding labels for each filepath in image_filepaths            
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
    
    output_dir = join(target_dir,dataset_name)
    ensure_dir_exists(output_dir)
    [ensure_dir_exists(join(output_dir,label)) for label in labels]
    
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 4] #9] 
    
    def parse_function(row):
        filepath, label, index = row
        print(index)
        filename, file_ext = splitext(basename(filepath))
        output_filepath = join(output_dir, label, filename+'.png')
        
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(output_filepath, img, compression_params)
        
        print(f'Converted image {index} and saved at {output_filepath}') #, end='\r', flush=True)
        assert os.path.isfile(output_filepath)
        return output_filepath, label
    
    def parse_batch(batch):
        print('starting batch inside')
        print(f'batch length {len(batch)}')
#         print('batch[0] = ',batch[0])
#         return len(batch)

        with ThreadPoolExecutor(max_workers=4) as executor:
            _map = executor.map
            return list(_map(parse_function, batch))
    
    
    
    indices = list(range(len(labels)))
    
    num_workers = os.cpu_count()//2
    
    chunksize = len(indices)//num_workers
    
    data = chunked(list(zip(image_filepaths, labels, indices)),chunksize)

    outputs = []
#     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
    with ProcessPool(nodes=num_workers) as executor:
        
        outputs = list(executor.imap(parse_batch, data))
        print('done')
#         outputs.extend(list(executor.imap(parse_batch, data)))
#         for chunk in data:
#             outputs.append(executor.map(parse_batch, chunk))
#             outputs.append(executor.map(parse_function, chunk))
        
#         output_paths = executor.map(parse_function, image_filepaths, labels, indices)
    
    return outputs
    
    



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
