import tensorflow as tf
import numpy as np
import sys
import cv2
import math as m
import time
import os
import glob
from scipy.misc import imread, imresize 
import scipy.misc as sp
from PIL import Image
from NCHW_hs_model import NCHW_hs_model
import h5py




class data_input_NCHW(object):

	def __init__(self, data_dir, subset='train', input_from='tfrecord', epoch=1, batch_size = 32):
		self.data_dir = data_dir
		self.img_id = 0
		self.epoch = epoch
		self.batch_size = batch_size
		if subset in ['train', 'validate', 'eval']:
			self.subset = subset
		else:
			raise ValueError('Invalid data subset')

		if input_from in ['tfrecord', 'hdf5', 'raw_data']:
			self.input_from = input_from
		else:
			raise ValueError('Invalid source of data')

	def get_file_name(self):
		if self.subset == 'eval':
			return [os.path.join(self.data_dir, "original_test_images/")]

		if self.input_from == 'tfrecord':
			if self.subset == 'train':
				return [os.path.join(self.data_dir,"Baidu_train_2.tfrecord")]
			elif self.subset == 'validate':
				return  [os.path.join(self.data_dir,"Baidu_val_2.tfrecord")]
		
		if self.input_from == 'hdf5' and self.subset == 'train':
			return os.path.join(self.data_dir,"train.hdf5")
		if self.input_from == 'hdf5' and self.subset == 'validate':
			return  os.path.join(self.data_dir, "val.hdf5")
		if self.input_from == 'raw_data' and self.subset == 'train':
			self.image_path = os.path.join(self.data_dir,"clean_images/images/")
			self.mask_path = os.path.join(self.data_dir,"clean_images/profiles/")
			return 
		if  self.input_from == 'raw_data' and self.subset == 'validate':
			self.image_path = os.path.join(self.data_dir,"clean_images/validate_image/")
			self.mask_path = os.path.join(self.data_dir,"clean_images/validate_profiles/")
			return


	def make_batch(self):
		if self.input_from == "tfrecord":
			filename = self.get_file_name()
			dataset = tf.data.TFRecordDataset(filename).repeat()
			dataset = dataset.map(self.parser_tf, num_parallel_calls=12 )

		elif self.input_from == "raw_data":
			self.get_file_name()
			self.img_name = os.listdir(self.image_path)
			dataset = tf.data.Dataset.from_generator(self.parser_raw, (tf.float32, tf.float32),((3, 48, 48), (48, 48)))

		elif self.input_from == "hdf5":
			data_file = self.get_file_name()
			data = h5py.File(data_file ,mode='r')
			self.X = data['img']
			self.Y = data['mask']
			dataset = tf.data.Dataset.from_generator(self.parser_hdf5, (tf.float32, tf.float32),((3,48, 48), (48, 48))).repeat(self.epoch+1)

		dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=False)
		dataset = dataset.repeat(self.epoch+1)	
		dataset = dataset.batch(self.batch_size)
		dataset = dataset.prefetch(2)
		iterator = dataset.make_one_shot_iterator()
		image_batch, mask_batch = iterator.get_next()	
		return image_batch, mask_batch

		
	def parser_tf(self, serialized_example):
        	features = tf.parse_single_example(serialized_example, 
				features={ 'train/mask': tf.FixedLenFeature([], tf.string),
                			'train/image': tf.FixedLenFeature([], tf.string)})
        	_mask = features['train/mask']
        	image = features['train/image']

        	images_decoded = tf.decode_raw(image, tf.float32)
        	images_reshaped = tf.reshape(images_decoded, [3, 48, 48])

        	masks_decoded = tf.decode_raw(_mask, tf.float32)
        	masks_reshaped = tf.reshape(masks_decoded, [48, 48])
        	return images_reshaped, masks_reshaped

	def parser_raw(self):
		if self.img_id >= (self.num_examples_per_epoch(self.subset) - self.batch_size):
			self.img_id = 0
		for i in range(self.batch_size):
                #Preparing and preprocessing images
			image1 = cv2.imread(os.path.join(self.image_path, self.img_name[self.img_id]))
			image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
			image1 = cv2.resize(image1, (48, 48), interpolation=cv2.INTER_CUBIC)
			image1 = np.reshape(image1, (3, 48, 48)).astype(np.float32)
			image1 = np.divide(image1, 255.0)
			
			mask1 = cv2.imread(os.path.join(self.mask_path, self.img_name[self.img_id][:5]+"-profile.jpg"))
			mask1 = cv2.resize(mask1, (48, 48), interpolation=cv2.INTER_CUBIC)
			mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
			mask1 = np.reshape(mask1, (48, 48)).astype(np.float32)
			mask1 = np.divide(mask1, 255.0)
			self.img_id += 1
			yield image1, mask1


	def parser_hdf5(self):
		if self.img_id >= (self.num_examples_per_epoch(self.subset) - self.batch_size):
			self.img_id = 0
		for i in range(self.batch_size):
			img = self.X[self.img_id].reshape(3,48,48).astype(np.float32)/ 255.0
			mask = self.Y[self.img_id].reshape(48,48).astype(np.float32)/ 255.0
			self.img_id += 1
			yield img , mask

	def num_examples_per_epoch(self, subset='train'):
		if subset == 'train':
			return 4848
		elif subset == 'validate':
			return 539
		elif subset == 'eval':
			return 1315
		else :
			raise ValueError('Invalid subset') 
