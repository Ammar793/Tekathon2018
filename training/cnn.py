from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
import csv
from glob import glob
import cv2
import time
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

###################################
### Import picture files 
###################################

file_name = 'Y_Train.csv'
folder_with_images = "./train/train/"
row_count=0
with open(file_name , 'r') as csvfile:
	spamreader = csv.reader(csvfile)
	row_count = sum(1 for row in spamreader)
	
print(row_count)

n_files = row_count-1

size_image = 64

def readImages(n_files, size_image):
	#read csv file to get labels for image	
	
	print("starting reading images")
	images = [[] for x in range (2)]
	
	first = True

	with open(file_name, 'r') as csvfile:
		spamreader = csv.reader(csvfile)
		for row in spamreader:
			if first:
				first = False
				continue
			if(os.path.isfile(folder_with_images + row[0]) ):
				img = cv2.imread (folder_with_images + row[0])
				new_img = imresize(img, (size_image, size_image, 3))				
				images[int(row[1])].append (new_img)
	#print (images)
	#time.sleep(100)
	print("finished reading images")
	return images
	
#Create classifiers and train

imageSet = readImages(n_files, size_image)
chips= imageSet[0]
drinks = imageSet[1]

allX = chips.copy()
allX.extend(drinks)

ally = [0] * len(chips) + [1] *len(drinks)

allX = np.array(allX,dtype=np.uint8)
ally = np.array(ally,dtype=np.uint8)

print(allX[0])


   
###################################
# Prepare train & test samples
###################################

# test-train split   
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)


###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################
print("asasdsasd")
# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
#conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
#network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
#conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
#conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
#network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
#network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
#network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
#network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
#acc = Accuracy(name="Accuracy")







#--------

network = conv_2d(network, 32, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 128, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 5)

network = conv_2d(network, 32, 5, activation='relu')
network = max_pool_2d(network, 5)

network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy')

#--------
tf.summary.FileWriter
					 

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_chips_drinks_9.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

###################################
# Train model for 100 epochs
###################################
#model.load('model_cat_dog_6.tflearn-1089')
X = X.astype('float32')
Y = Y.astype('float32')
X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=100, run_id='model_chips_drinks_6', show_metric=True)

model.save('model_chips_drinks_6_final.tflearn')