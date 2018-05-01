import tensorflow as tf
import numpy as np
import math as m
import scipy.misc as sp
from PIL import Image

def NCHW_hs_model(image, keep_prob):
        
        
        #conv1
        kernel_1 = tf.Variable(tf.truncated_normal([5, 5, 3, 48], dtype=tf.float32,
                                stddev=0.01),  name='kernel_1')
        biases_1 = tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32),
                                trainable=True, name='biases_1')
        conv_1_ = tf.nn.conv2d(image, kernel_1, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_1 = tf.nn.bias_add(conv_1_, biases_1, data_format='NCHW')
        conv_1 = tf.nn.relu(out_1)

        
        #Pool1
        pool_1 = tf.nn.max_pool(conv_1,
                                ksize=[1,1, 3,3],
                                strides=[1,1,2,2],
                                padding='SAME',
				data_format='NCHW')
        

        
        #conv2
        kernel_2 = tf.Variable(tf.truncated_normal([5, 5, 48, 128], dtype=tf.float32,
                                stddev=0.01), name='kernel_2')
        biases_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                trainable=True, name='biases_2')
        conv_2_ = tf.nn.conv2d(pool_1, kernel_2, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_2 = tf.nn.bias_add(conv_2_, biases_2, data_format='NCHW')
        conv_2 = tf.nn.relu(out_2)
        
        
        #Pool2
        pool_2 = tf.nn.max_pool(conv_2,
                                ksize=[1,1,3,3],
                                strides=[1,1,2,2],
                                padding='SAME',
				data_format='NCHW')
        
        
        #conv3
        kernel_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 192], dtype=tf.float32,
                                stddev=0.01), name='kernel_3')
        biases_3 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                                trainable=True, name='biases_3')
        conv_3_ = tf.nn.conv2d(pool_2, kernel_3, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_3 = tf.nn.bias_add(conv_3_, biases_3, data_format='NCHW')
        conv_3 = tf.nn.relu(out_3)
        
        
        #conv4
        kernel_4 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32,
                                stddev=0.01), name='kernel_4')
        biases_4 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                                trainable=True, name='biases_4')
        conv_4_ = tf.nn.conv2d(conv_3, kernel_4, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_4 = tf.nn.bias_add(conv_4_, biases_4, data_format='NCHW')
        conv_4 = tf.nn.relu(out_4)
        
        
        #conv5
        kernel_5 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32,
                                stddev=0.01), name='kernel_5')
        biases_5 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                                trainable=True, name='biases_5')
        conv_5_ = tf.nn.conv2d(conv_4, kernel_5, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_5 = tf.nn.bias_add(conv_5_, biases_5, data_format='NCHW')
        conv_5 = tf.nn.relu(out_5)
       
        
        #conv6
        kernel_6 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32,
                                stddev=0.01), name='kernel_6')
        biases_6 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                                trainable=True, name='biases_6')
        conv_6_ = tf.nn.conv2d(conv_5, kernel_6, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_6 = tf.nn.bias_add(conv_6_, biases_6, data_format='NCHW')
        conv_6 = tf.nn.relu(out_6)
        
        
        #conv7
        kernel_7 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32,
                                stddev=0.01), name='kernel_7')
        biases_7 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                                trainable=True, name='biases_7')
        conv_7_ = tf.nn.conv2d(conv_6, kernel_7, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_7 = tf.nn.bias_add(conv_7_, biases_7, data_format='NCHW')
        conv_7 = tf.nn.relu(out_7)
       
        
        #conv8
        kernel_8 = tf.Variable(tf.truncated_normal([3, 3, 192, 64], dtype=tf.float32,
                                stddev=0.01), name='kernel_8')
        biases_8 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                trainable=True, name='biases_8')
        conv_8_ = tf.nn.conv2d(conv_7, kernel_8, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
        out_8 = tf.nn.bias_add(conv_8_, biases_8, data_format='NCHW')
        conv_8 = tf.nn.relu(out_8)
       
        
        #Pool3
        pool_3 = tf.nn.max_pool(conv_8,
                                ksize=[1,1,3,3],
                                strides=[1,1,2,2],
                                padding='SAME', 
				data_format='NCHW')
        
        
        #FC1     
        fc1w = tf.Variable(tf.truncated_normal([64*6*6, 1024], dtype=tf.float32,
                                                 stddev=0.01), name='fc_1')
        fc1b = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                            trainable=True, name='biases_fc1')
        fc1l = tf.nn.bias_add(tf.matmul(tf.reshape(pool_3, [-1, 6*6*64]), fc1w),
                               fc1b)
        fc1 = tf.nn.relu(fc1l)
        fc1_d = tf.nn.dropout(fc1, keep_prob)     
        
        #FC2     
        fc2w = tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32,
                                                 stddev=0.01), name='fc_2')
        fc2b = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                            trainable=True, name='biases_fc2')
        fc2l = tf.nn.bias_add(tf.matmul(fc1_d, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)
        fc2_d = tf.nn.dropout(fc2, keep_prob)
        
        #FC3     
        fc3w = tf.Variable(tf.truncated_normal([1024, 48*48], dtype=tf.float32,
                                                 stddev=0.01), name='fc_1')
        fc3b = tf.Variable(tf.constant(0.0, shape=[48*48], dtype=tf.float32),
                            trainable=True, name='biases_fc3')
        fc3l = tf.nn.bias_add(tf.matmul(fc2_d, fc3w), fc3b)
        

        #Sigmoid function
        output_ = tf.sigmoid(fc3l)
      
        output = tf.reshape(output_, [-1, 48, 48])

        return output


