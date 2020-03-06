'''

'''
import os

def ensure_dir_exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    if os.path.isdir(dir_path):
        return True
    else:
        return False
    
    
    
def set_visible_gpus(gpu_ids=[0]):
    from tensorflow.config import experimental
#     import pdb; pdb.set_trace()
#     gpus = experimental.get_visible_devices('GPU')
    gpus = tf.config.list_physical_devices('GPU') 

    if gpus:
#         print(gpus)
        gpu_objects = [gpus[i] for i in gpu_ids]

        
        experimental.set_visible_devices(gpu_objects, 'GPU')    
#         experimental.set_memory_growth(*gpu_objects, True)
#         experimental.set_visible_devices(gpu_objects, 'GPU')
#         experimental.set_memory_growth(*gpu_objects, True)
        
        logical_gpus = experimental.get_visible_devices('GPU')
#         experimental.set_memory_growth(*logical_gpus, True)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        print(logical_gpus)
    else:
        print("No visible GPUs found")
        
        
        
        
def __check_if_hdf5(path=''):
    _ext = ['h5','hdf5']
    
    for e in _ext:
        if path.endswith(e):
            return True
    return False

def __check_if_json(path=''):
    
    if path.endswith('json'):
        return True
    else:
        return False

    
def validate_filepath(path, file_type='json'):
    if file_type == 'json':
        return __check_if_json(path)
    if file_type in ['h5','hdf5']:
        return __check_if_hdf5(path)