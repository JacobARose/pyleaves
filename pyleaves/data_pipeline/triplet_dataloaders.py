# @Author: Jacob A Rose
# @Date:   Tue, May 26th 2020, 11:15 pm
# @Email:  jacobrose@brown.edu
# @Filename: triplet_dataloaders.py


# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import numpy as np

# A workaround to avoid crash because tfds may open to many files.
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000


DATASET_SPLITS = {
  'cifar10': {'train': 'train[:98%]', 'test': 'test'},
  'cifar100': {'train': 'train[:98%]', 'test': 'test'},
  'imagenet2012': {'train': 'train[:99%]', 'test': 'validation'},
}


def get_dataset_info(dataset, split, examples_per_class):
	data_builder = tfds.builder(dataset)
	original_num_examples = data_builder.info.splits[split].num_examples
	num_classes = data_builder.info.features['label'].num_classes
	if examples_per_class is not None:
		num_examples = examples_per_class * num_classes
	else:
		num_examples = original_num_examples
	return {'original_num_examples': original_num_examples,
		  'num_examples': num_examples,
		  'num_classes': num_classes}


def sample_subset(data, num_examples, num_classes,
				  examples_per_class, examples_per_class_seed):
  data = data.batch(min(num_examples, MAX_IN_MEMORY))

  data = data.as_numpy_iterator().next()

  np.random.seed(examples_per_class_seed)
  indices = [idx
			 for c in range(num_classes)
			 for idx in np.random.choice(np.where(data['label'] == c)[0],
										 examples_per_class,
										 replace=False)]

  data = {'image': data['image'][indices],
		  'label': data['label'][indices]}

  data = tf.data.Dataset.zip(
	(tf.data.Dataset.from_tensor_slices(data['image']),
	 tf.data.Dataset.from_tensor_slices(data['label'])))
  return data.map(lambda x, y: {'image': x, 'label': y},
				  tf.data.experimental.AUTOTUNE)


def get_data(dataset, mode,
			 repeats, batch_size,
			 resize_size, crop_size,
			 mixup_alpha,
			 examples_per_class, examples_per_class_seed,
			 num_devices,
			 tfds_manual_dir):

  split = DATASET_SPLITS[dataset][mode]
  dataset_info = get_dataset_info(dataset, split, examples_per_class)

  data_builder = tfds.builder(dataset)
  data_builder.download_and_prepare(
   download_config=tfds.download.DownloadConfig(manual_dir=tfds_manual_dir))
  data = data_builder.as_dataset(
	split=split,
	decoders={'image': tfds.decode.SkipDecoding()})
  decoder = data_builder.info.features['image'].decode_example

  if (mode == 'train') and (examples_per_class is not None):
	data = sample_subset(data,
						 dataset_info['original_num_examples'],
						 dataset_info['num_classes'],
						 examples_per_class, examples_per_class_seed)

  def _pp(data):
	im = decoder(data['image'])
	if mode == 'train':
	  im = tf.image.resize(im, [resize_size, resize_size])
	  im = tf.image.random_crop(im, [crop_size, crop_size, 3])
	  im = tf.image.flip_left_right(im)
	else:
	  # usage of crop_size here is intentional
	  im = tf.image.resize(im, [crop_size, crop_size])
	im = (im - 127.5) / 127.5
	label = tf.one_hot(data['label'], dataset_info['num_classes'])
	return {'image': im, 'label': label}

  data = data.cache()
  data = data.repeat(repeats)
  if mode == 'train':
	data = data.shuffle(min(dataset_info['num_examples'], MAX_IN_MEMORY))
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  def _mixup(data):
	beta_dist = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
	beta = tf.cast(beta_dist.sample([]), tf.float32)
	data['image'] = (beta * data['image'] +
					 (1 - beta) * tf.reverse(data['image'], axis=[0]))
	data['label'] = (beta * data['label'] +
					 (1 - beta) * tf.reverse(data['label'], axis=[0]))
	return data

  if mixup_alpha is not None and mixup_alpha > 0.0 and mode == 'train':
	data = data.map(_mixup, tf.data.experimental.AUTOTUNE)

  # Shard data such that it can be distributed accross devices
  def _shard(data):
	data['image'] = tf.reshape(data['image'],
							   [num_devices, -1, crop_size, crop_size, 3])
	data['label'] = tf.reshape(data['label'],
							   [num_devices, -1, dataset_info['num_classes']])
	return data
  if num_devices is not None:
	data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.prefetch(1)




#################################################################
#################################################################
#################################################################
#################################################################


import os
import sys
import keras
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import Sequence
from torchvision.transforms import *



class DataLoader(Sequence):
    def __init__(self, txt, batch_size, tag='train',transforms=None):
        self.data, self.labels = self._parsing(txt)
        self.tag = tag
        self.transforms = transforms
        self.label_set = set(self.labels)
        self.label2index = { i:list(np.where(np.array(self.labels)==i)[0]) for i in self.label_set}
        self.batch_size = batch_size
        if tag != 'train':
            '''
            依次遍历整个数据集，第i个数据，作为它的正pair，随机从所有该类别内选取一个，负样本随机选取一个异类，再随机选取其中一个样本。
            '''
            triplets = [[i,
                        random.choice(self.label2index[self.labels[i]]),
                        random.choice(self.label2index[random.choice(list(self.label_set - set([self.labels[i]])))])
                        ]
                        for i in tqdm(range(len(self.data)))]



            self.test_triplet = triplets

    def __getitem__(self, idx):
        samples = range(idx * self.batch_size, (idx+1) * self.batch_size)
        # print('samples',samples)
        q_imgs, p_imgs, n_imgs = [], [], []

        for index in samples:
            if self.tag == 'train':
                img1 = self._read(self.data[index])
                label = self.labels[index]

                postive_index = index
                while postive_index == index:
                    postive_index = random.choice(self.label2index[label])
                img2 = self._read(self.data[postive_index])

                negtive_label = random.choice(list(self.label_set - set([label])))
                negtive_index = random.choice(self.label2index[negtive_label])
                img3 = self._read(self.data[negtive_index])



            else:
                sample = self.test_triplet[index]
                # print(sample)
                img1 = self._read(self.data[sample[0]])
                img2 = self._read(self.data[sample[1]])
                img3 = self._read(self.data[sample[2]])
                # print(img1.shape, img2.shape, img3.shape)

            if self.transforms:
                img1 = self.transforms(img1)
                img2 = self.transforms(img2)
                img3 = self.transforms(img3)

            q_imgs.append(np.array(img1))
            p_imgs.append(np.array(img2))
            n_imgs.append(np.array(img3))

        return [np.array(q_imgs).astype('float32') / 255, np.array(p_imgs).astype('float32') / 255, np.array(n_imgs).astype('float32') / 255], None


    def _read(self, path):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if w != h:
            MAX = max(w, h)
            img = ImageOps.expand(img, border=(0, 0, MAX - w, MAX - h), fill=0)

        return img



    def _parsing(self, txt):
        paths = []
        labels = []
        with open(txt, 'r') as f:
            f = f.readlines()
            for i in f:
                path, label = i.strip().split(' ')
                paths.append(path)
                labels.append(int(label))
        return paths, labels


    def __len__(self):
        if self.tag == 'train':
            return len(self.data) // self.batch_size
        else:
            return len(self.test_triplet) // self.batch_size



if __name__ == '__main__':
    txt_path = sys.argv[1]
    testTrans = Compose([Resize(size=(224,224))])

    data = dataLoader(txt_path, 32, 'train', transforms=testTrans)

    data = iter(data)
    [i,j,k],m = next(data)
    print(i.shape, j.shape, k.shape)
    # a = GetSampling(main_path)
    # for i,j in data:
    # print(i.shape)
