''''''
import os

def ensure_dir_exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    if os.path.isdir(dir_path):
        return True
    else:
        return False