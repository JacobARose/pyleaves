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
    gpus = experimental.get_visible_devices('GPU')
    if gpus:
        experimental.set_visible_devices([gpus[i] for i in gpu_ids], 'GPU')
        logical_gpus = experimental.get_visible_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    else:
        print("No visible GPUs found")