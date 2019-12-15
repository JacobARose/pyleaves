'''
Functions for managing images
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
