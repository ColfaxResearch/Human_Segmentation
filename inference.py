import tensorflow as tf
import numpy as np
import sys
import cv2
import time
import matplotlib.pyplot as plt
import os
import glob
import requests
import shutil
from scipy.misc import imread, imresize
from NHWC_hs_model import NHWC_hs_model
import IPython.display


#Inference_________________________________________________________
def infer_validate():
    
    img_path = "/data/cfxevents/2018-aidevcon/validate-dataset-reduced/"
    mask_path = "/data/cfxevents/2018-aidevcon/validate-profiles-reduced/"
    img_name = os.listdir(img_path)
    img_id = np.random.randint(0,len(img_name))
    
    image1 = cv2.imread(os.path.join(img_path, img_name[img_id]))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image1, (48, 48), interpolation=cv2.INTER_CUBIC)
    image2 = np.reshape(image2, (1, 48, 48, 3)).astype(np.float32)  # image is ready
    
    mask1 = cv2.imread(os.path.join(mask_path, img_name[img_id][:5]+"-profile.jpg"))
    mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
    mask2 = cv2.resize(mask1, (48, 48), interpolation=cv2.INTER_CUBIC)
    mask2 = np.reshape(mask2, (1, 48, 48))
    
    return image1, image2, mask2
    

#Inference_________________________________________________________
def infer(in_image="input-image.jpg"):
    img_path = in_image
    image1 = cv2.imread(img_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image1, (48, 48), interpolation=cv2.INTER_CUBIC)
    image2 = np.reshape(image2, (1, 48, 48, 3)).astype(np.float32)  # image is ready
    return image1, image2
   


def download_image(url):
    response = requests.get(url, stream=True)
    with open('web-image.jpg', 'wb') as outfile:
        shutil.copyfileobj(response.raw, outfile)
    del response 
    return
    
def visualize(source="validate_dataset", url=" "):
    image = tf.placeholder(tf.float32, (1, 48, 48, 3))
    output = NHWC_hs_model(image, 1.0)
    output_r = tf.round(output)
    
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 12
    config.inter_op_parallelism_threads = 2
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.reset_default_graph()
    saver.restore(sess, "weights/NHWC_train_hs_val_1000.ckpt")      
    if source=="validate_dataset":
        orig_image, image_in, orig_mask = infer_validate()    
    elif source=="url":
        download_image(url)
        orig_image, image_in = infer("web-image.jpg")    
    else:
        orig_image, image_in = infer()
    image_in_norm = image_in/255.0

 
    # Run session__________________________________________
    time1 = time.time()
    output= sess.run(output_r, feed_dict={image: image_in_norm})
    time2 = time.time()-time1
    
    
    fig = plt.figure(figsize=(20,20))
    fig.add_subplot(2, 2, 1) #Original image
    plt.imshow((orig_image).astype(np.uint8))
    fig.add_subplot(2, 2, 2)
    plt.imshow((image_in[0,...]).astype(np.uint8)) #Resized image
    fig.add_subplot(2, 2, 3) #Output mask
    plt.imshow((output[0,...]*255.0).astype(np.uint8), cmap='gray')
    if source=="validate_dataset":
        fig.add_subplot(2, 2, 4)
        plt.imshow((orig_mask[0,...]).astype(np.uint8), cmap='gray') #resized gt
                
    plt.show()   
    sess.close()
    return
