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
from PIL import Image
import math


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_SCALE_FACTOR = 0.017

_RESIZE_SIDE_MIN = 350
_RESIZE_SIDE_MAX = 512

_ROTATION_ANGLE_MIN = 0
_ROTATION_ANGLE_MAX = 4


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
  resized_image.set_shape([new_height, new_width, 3])
  return resized_image

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

def preprocess_for_train(image,
                         output_height,
                         output_width,
                         folder=None,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX,
                         rotation_angle_min=_ROTATION_ANGLE_MIN,
                         rotation_angle_max=_ROTATION_ANGLE_MAX,
                         preprocess_func='densenet'):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """

  if preprocess_func in ['inception_leaves','inception_leaves_color']:
      resize_side = tf.random_uniform([], minval=int(output_height*1.05), maxval=int(output_height*2), dtype=tf.int32)
      rotation_angle = tf.random_uniform([], minval=rotation_angle_min, maxval=rotation_angle_max, dtype=tf.int32)
      image = tf.image.rot90(image, rotation_angle)
      image = _aspect_preserving_resize(image, resize_side)
      image = _central_crop([image], output_height, output_width)[0]
  else:
      resize_side = tf.random_uniform(
          [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)
      rotation_angle = tf.random_uniform(
      [], minval=0, maxval=0, dtype=tf.int32)
      image = _aspect_preserving_resize(image, resize_side)
      image = _random_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  image = tf.image.random_flip_left_right(image)

  if preprocess_func == 'inception_v1':
      print('Inception Format Augmentation')
      image = inception_preprocess(image)
  elif preprocess_func == 'densenet':
      print('DenseNet Format Augmentation')
      image = denseNet_preprocess(image)
  elif preprocess_func == 'vgg':
      print('VGG Format Augmentation')
      image = vgg_preprocess(image)
  elif preprocess_func == 'inception_leaves':
      print('Leaves preprocessing')
      image = inception_preprocess_leaves(image)
  elif preprocess_func == 'inception_leaves_color':
      print('Leaves color preprocessing')
      image = inception_preprocess_leaves_color(image)
  return image


def preprocess_for_eval(image, output_height, output_width, resize_side=_RESIZE_SIDE_MIN,preprocess_func='densenet'):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side+10)
  image = _central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)

  if preprocess_func =='inception_leaves':
      print('Inception Leaves')
      image = inception_preprocess_leaves(image)

  if preprocess_func == 'inception_v1':
      print('Inception Format Augmentation')
      image = inception_preprocess(image)
  elif preprocess_func == 'densenet':
      print('DenseNet Format Augmentation')
      image = denseNet_preprocess(image)
  elif preprocess_func == 'vgg':
      print('VGG Format Augmentation')
      image = vgg_preprocess(image)

  return image
