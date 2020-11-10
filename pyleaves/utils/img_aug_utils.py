# @Author: Jacob A Rose
# @Date:   Sat, July 18th 2020, 8:41 pm
# @Email:  jacobrose@brown.edu
# @Filename: img_aug_utils.py


"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

import os
import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
from PIL import Image
import math
from typing import Tuple
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_SCALE_FACTOR = 0.017

_RESIZE_SIDE_MIN = 350
_RESIZE_SIDE_MAX = 512

_ROTATION_ANGLE_MIN = 0
_ROTATION_ANGLE_MAX = 4




import matplotlib.pyplot as plt
import numpy as np


def plot_top_k_batch(img, label, label_names=None, top_k=2, row=6, col=4):
    img = (img + 1.0)/2.0
    
    plt.figure(figsize=(22,int(15*row/col)))
    for j in range(row*col):
        label_idx = np.argsort(label[j,:])[::-1]
        top_k_idx = label_idx[:top_k].tolist()
        top_k_names = [label_names[k] for k in top_k_idx]
        top_k_weights = [label[j,k] for k in top_k_idx]

        text_labels = [(name,weight) for name, weight in zip(top_k_names,top_k_weights)]
        title = ',\n'.join([f'{name}={weight:.2f}' for name, weight in text_labels])

        assert sum([w for _,w in text_labels]) == 1
        
        plt.subplot(row,col,j+1)
        plt.axis('off')
        plt.imshow(img[j])
        plt.title(title)
    plt.show()


def display_batch_augmentation(data_iter: tf.data.Dataset, augmentation_function=None, label_names=None, aug_batch_size=24, top_k=2, row=6, col=4):

    all_elements = data_iter.map(lambda x,y,_: (x,y)).unbatch()
    augmented_element = all_elements.repeat().batch(aug_batch_size).map(augmentation_function)
    row = min(row,aug_batch_size//col)
    for (img,label) in augmented_element:
        plot_top_k_batch(img, label, label_names=label_names, top_k=top_k, row=row, col=col)
        break


# num_classes = config.num_classes
def cutmix(image, label, probability = 1.0, target_size=None, aug_batch_size=1, num_classes=None):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    DIM = target_size[0]
    
    imgs = []; labs = []
    for j in range(aug_batch_size):
        # DO CUTMIX WITH probability DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=probability, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast( tf.random.uniform([],0,aug_batch_size),tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0
        WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:DIM,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],num_classes)
            lab2 = tf.one_hot(label[k],num_classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(aug_batch_size,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(aug_batch_size,num_classes))
    return image2,label2



def mixup(image, label, probability = 1.0, target_size=None, aug_batch_size=1, num_classes=None):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = target_size[0]
    
    imgs = []; labs = []
    for j in range(aug_batch_size):
        # DO MIXUP WITH probability DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=probability, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,aug_batch_size),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        # MAKE CUTMIX LABEL
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],num_classes)
            lab2 = tf.one_hot(label[k],num_classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(aug_batch_size,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(aug_batch_size,num_classes))
    return image2,label2




def transform(image,label, switch = 0.5, cutmix_prob = 0.666, mixup_prob = 0.666,
              target_size=None, aug_batch_size=1, num_classes=None):
    # THIS FUNCTION APPLIES BOTH CUTMIX AND MIXUP
    DIM = target_size[0]
    
    # FOR switch PERCENT OF TIME WE DO CUTMIX AND (1-switch) WE DO MIXUP
    _cutmix = partial(cutmix, aug_batch_size=aug_batch_size, num_classes=num_classes, target_size=target_size)
    _mixup = partial(mixup, aug_batch_size=aug_batch_size, num_classes=num_classes, target_size=target_size)

    image2, label2 = _cutmix(image, label, cutmix_prob)
    image3, label3 = _mixup(image, label, mixup_prob)
    imgs = []; labs = []
    for j in range(aug_batch_size):
        P = tf.cast( tf.random.uniform([],0,1)<=switch, tf.float32)
        imgs.append(P*image2[j,]+(1-P)*image3[j,])
        labs.append(P*label2[j,]+(1-P)*label3[j,])
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image4 = tf.reshape(tf.stack(imgs),(aug_batch_size,DIM,DIM,3))
    label4 = tf.reshape(tf.stack(labs),(aug_batch_size,num_classes))
    return image4,label4

from functools import partial
def apply_cutmixup(dataset, do_aug=True, aug_batch_size=1, num_classes=None, target_size=None, batch_size=1):
    dataset = dataset.batch(aug_batch_size)
    _transform = partial(transform, aug_batch_size=aug_batch_size, num_classes=num_classes, target_size=target_size)
    if do_aug: dataset = dataset.map(_transform, num_parallel_calls=AUTO) # note we put AFTER batching
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) 
    return dataset



###################################################################


def resize_repeat(target_size: Tuple[int], training: bool):
  """ Adapted from Ivan's code located at
  https://github.com/serre-lab/tripletcyclegan/blob/1b192a631a28b03304980060057843996e5a4b14/tf2lib/data/dataset.py#L81

  Resize function that creates repetitions along the shortest image side to resize without aspect ratio distortion.


  Args:
      target_size ([type]): [description]
      training ([type]): [description]

  Returns:
      [type]: [description]
  """  
  if training:
    @tf.function
    def _map_fn(img):  # preprocessing
      img = tf.image.random_flip_left_right(img)
      maxside = tf.math.maximum(tf.shape(img)[0],tf.shape(img)[1])
      minside = tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1])
      new_img = img
        
      if tf.math.divide(maxside,minside) > 1.3:
        repeat = tf.math.floor(tf.math.divide(maxside,minside))  
        new_img = img
        if tf.math.equal(tf.shape(img)[1],minside):
          for i in range(int(repeat)):
            new_img = tf.concat((new_img, img), axis=1) 

        if tf.math.equal(tf.shape(img)[0],minside):
          for i in range(int(repeat)):
            new_img = tf.concat((new_img, img), axis=0) 
      else:
        new_img = img      
      img = tf.image.resize(new_img, tuple(target_size))
      return img
  else:
    @tf.function
    def _map_fn(img):  # preprocessing
      maxside = tf.math.maximum(tf.shape(img)[0],tf.shape(img)[1])
      minside = tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1])
      new_img = img
        
      if tf.math.divide(maxside,minside) > 1.3:
        repeat = tf.math.floor(tf.math.divide(maxside,minside))  
        new_img = img
        if tf.math.equal(tf.shape(img)[1],minside):
          for i in range(int(repeat)):
            new_img = tf.concat((new_img, img), axis=1) 

        if tf.math.equal(tf.shape(img)[0],minside):
          for i in range(int(repeat)):
            new_img = tf.concat((new_img, img), axis=0) 
      else:
        new_img = img
      img = tf.image.resize(new_img, tuple(target_size))
      return img
  return _map_fn

#######################################################################
#######################################################################












def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  height = tf.cast(height, dtype=tf.float32)
  width = tf.cast(width, dtype=tf.float32)
  smallest_side = tf.cast(smallest_side, dtype=tf.float32)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)

  new_height = tf.cast(height * scale, dtype=tf.int32)
  new_width = tf.cast(width * scale, dtype=tf.int32)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """

  height = tf.shape(image)[0]
  width = tf.shape(image)[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize(image, [new_height, new_width])
  resized_image = tf.squeeze(resized_image,[0])
  return resized_image
  # resized_image.set_shape([new_height, new_width, 3])

# def inception_preprocess(images):
#     ## Images are assumed to be [0,255]
#     images = images / 255.0
#     images = tf.subtract(images, 0.5)
#     images = tf.multiply(images, 2.0)
#     return images
#
# def inception_preprocess_leaves(images):
#     ## Images are assumed to be [0,255]
#     images = tf.compat.v1.image.rgb_to_grayscale(images)
#     images = tf.compat.v1.image.grayscale_to_rgb(images)
#     images = images / 255.0
#     images = tf.subtract(images, 0.5)
#     images = tf.multiply(images, 2.0)
#
#     return images
# def inception_preprocess_leaves_color(images):
#     ## Images are assumed to be [0,255]
#     images = images / 255.0
#     images = tf.subtract(images, 0.5)
#     images = tf.multiply(images, 2.0)
#
#     return images
#
# def denseNet_preprocess(images):
#     ## Images are assumed to be [0,255]
#     images = images / 255.0
#     images = _mean_image_subtraction(images, [0.485, 0.456, 0.406])
#     images = _std_image_normalize(images, [0.229, 0.224, 0.225])
#     return images
#
# def vgg_preprocess(images):
#     ## Images are assumed to be [0,255]
#     _R_MEAN = 123.68
#     _G_MEAN = 116.78
#     _B_MEAN = 103.94
#     images = _mean_image_subtraction(images, [_R_MEAN , _G_MEAN, _B_MEAN])
#     return images

# def preprocess_for_train(image,
#                          output_height,
#                          output_width,
#                          folder=None,
#                          resize_side_min=_RESIZE_SIDE_MIN,
#                          resize_side_max=_RESIZE_SIDE_MAX,
#                          rotation_angle_min=_ROTATION_ANGLE_MIN,
#                          rotation_angle_max=_ROTATION_ANGLE_MAX,
#                          preprocess_func='densenet'):
#   """Preprocesses the given image for training.

#   Note that the actual resizing scale is sampled from
#     [`resize_size_min`, `resize_size_max`].

#   Args:
#     image: A `Tensor` representing an image of arbitrary size.
#     output_height: The height of the image after preprocessing.
#     output_width: The width of the image after preprocessing.
#     resize_side_min: The lower bound for the smallest side of the image for
#       aspect-preserving resizing.
#     resize_side_max: The upper bound for the smallest side of the image for
#       aspect-preserving resizing.

#   Returns:
#     A preprocessed image.
#   """

#   if preprocess_func in ['inception_leaves','inception_leaves_color']:
#       resize_side = tf.random_uniform([], minval=int(output_height*1.05), maxval=int(output_height*2), dtype=tf.int32)
#       rotation_angle = tf.random_uniform([], minval=rotation_angle_min, maxval=rotation_angle_max, dtype=tf.int32)
#       image = tf.image.rot90(image, rotation_angle)
#       image = _aspect_preserving_resize(image, resize_side)
#       image = _central_crop([image], output_height, output_width)[0]
#   else:
#       resize_side = tf.random_uniform(
#           [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)
#       rotation_angle = tf.random_uniform(
#       [], minval=0, maxval=0, dtype=tf.int32)
#       image = _aspect_preserving_resize(image, resize_side)
#       image = _random_crop([image], output_height, output_width)[0]
#   image.set_shape([output_height, output_width, 3])
#   image = tf.to_float(image)
#   image = tf.image.random_flip_left_right(image)

#   if preprocess_func == 'inception_v1':
#       print('Inception Format Augmentation')
#       image = inception_preprocess(image)
#   elif preprocess_func == 'densenet':
#       print('DenseNet Format Augmentation')
#       image = denseNet_preprocess(image)
#   elif preprocess_func == 'vgg':
#       print('VGG Format Augmentation')
#       image = vgg_preprocess(image)
#   elif preprocess_func == 'inception_leaves':
#       print('Leaves preprocessing')
#       image = inception_preprocess_leaves(image)
#   elif preprocess_func == 'inception_leaves_color':
#       print('Leaves color preprocessing')
#       image = inception_preprocess_leaves_color(image)
#   return image


# def preprocess_for_eval(image, output_height, output_width, resize_side=_RESIZE_SIDE_MIN,preprocess_func='densenet'):
#   """Preprocesses the given image for evaluation.

#   Args:
#     image: A `Tensor` representing an image of arbitrary size.
#     output_height: The height of the image after preprocessing.
#     output_width: The width of the image after preprocessing.
#     resize_side: The smallest side of the image for aspect-preserving resizing.

#   Returns:
#     A preprocessed image.
#   """
#   image = _aspect_preserving_resize(image, resize_side+10)
#   image = _central_crop([image], output_height, output_width)[0]
#   image.set_shape([output_height, output_width, 3])
#   image = tf.to_float(image)

#   if preprocess_func =='inception_leaves':
#       print('Inception Leaves')
#       image = inception_preprocess_leaves(image)

#   if preprocess_func == 'inception_v1':
#       print('Inception Format Augmentation')
#       image = inception_preprocess(image)
#   elif preprocess_func == 'densenet':
#       print('DenseNet Format Augmentation')
#       image = denseNet_preprocess(image)
#   elif preprocess_func == 'vgg':
#       print('VGG Format Augmentation')
#       image = vgg_preprocess(image)

#   return image
