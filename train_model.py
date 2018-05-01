import tensorflow as tf
import numpy as np
import sys
import cv2
import time
import os
import glob
from scipy.misc import imread, imresize
import scipy.misc as sp
from PIL import Image
from NCHW_hs_model import NCHW_hs_model
from NCHW_read_data import data_input_NCHW
from NHWC_hs_model import NHWC_hs_model
from NHWC_read_data import data_input_NHWC

#
class train_model(object):

	def __init__(self):
		#Parameters*****************************************
		self.lr = 0.001
		self.keep_prob = 0.5 
		self.data_dir  ="/data/cfxevents/2018-aidevcon"

	def train(self, batch, n_epochs, subset='train', source='tfrecord', inter=0, intra=0, data_format='NHWC'):
		
		if data_format=='NHWC':
			data_in = data_input_NHWC(self.data_dir, subset, source, n_epochs,  batch)
			data_in_val = data_input_NHWC(self.data_dir, "validate", source, n_epochs,  batch)

			if subset=="train":
				images, masks = data_in.make_batch()
			else:
				images, masks = data_in_val.make_batch()
			
			output = NHWC_hs_model(images, self.keep_prob)
		
		elif data_format=='NCHW':
			data_in = data_input_NCHW(self.data_dir, subset, source, n_epochs,  batch)
			data_in_val = data_input_NCHW(self.data_dir, "validate", source, n_epochs,  batch)

			if subset=="train":
				images, masks = data_in.make_batch()
			else:
				images, masks = data_in_val.make_batch()
			
			output = NCHW_hs_model(images, self.keep_prob)
		

		# Loss calculation_______________________________________________
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(output, masks)), (1,2)))
		optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

		# Accuracy Calculation Training dataset___________________________
		out_train = tf.cast(tf.round(output), tf.bool)
		gt_train = tf.cast(masks, tf.bool)
		and_train = tf.cast(tf.logical_and(out_train, gt_train), tf.float32)
		or_train = tf.cast(tf.logical_or(out_train, gt_train),tf.float32)
		pred_train = tf.reduce_mean(tf.div(tf.reduce_sum(and_train, axis=(1,2)), tf.reduce_sum(or_train, axis=(1,2))))

		#Session configuration
		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = intra
		config.inter_op_parallelism_threads = inter
		sess = tf.Session(config=config)
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess.run(init)
       
		num_batches_v = np.ceil(539.0/batch).astype(np.uint32)
		num_batches = np.ceil(4848.0/batch).astype(np.uint32)
		for epoch in range(n_epochs):
			total_loss = 0
			total_loss_v = 0
			total_acc = 0
			total_acc_v = 0
			if (epoch > 20):
				lr = 0.0001
			time1 = time.time()

			for i in range(num_batches):
				subset = "train"
			    	# Run session___________________________________________
				loss_tr, opt, acc_tr = sess.run([loss, optimizer, pred_train])
				total_loss += loss_tr
				total_acc += acc_tr

			time2 = time.time()-time1
			print('Training time Epoch %d: %f sec' % (epoch,  time2))
			#Accuracy Calculation Training dataset
			avg_loss = total_loss/num_batches
			avg_acc = (total_acc/num_batches)*100.0
			#print('Training accuracy: %f %% '%  avg_acc)
			#print('Training loss: %f'% avg_loss)
		
		    	#Accuracy & Loss Calculation Validation dataset_
			subset = "validate"
			for i in range(num_batches_v):
				loss_val, acc_val = sess.run([loss, pred_train])
				total_loss_v += loss_val
				total_acc_v += acc_val
			avg_loss_v = total_loss_v/num_batches_v
			avg_acc_v = (total_acc_v/num_batches_v)*100.0
			#print('Validation accuracy: %f %% '% avg_acc_v)
			#print('Validation loss: %f '% avg_loss_v)
		return
