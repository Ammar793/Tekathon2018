import numpy as np
import cv2
import os
import csv
from scipy import interp
from scipy import stats
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

#read images from folder
def readImages():
	#read csv file to get labels for image	
	print("starting reading images")
	images = []
	names = []
	first = True
	
	folder = "C:/Users/Ammar Raufi/Desktop/winter 17/computer_vision/final_proj/test/X_Test/"
	
	files = os.listdir(folder)	
	
	for i in range (0,len(files)):	
		img = cv2.imread (folder+'/'+files[i])
		img = cv2.resize(img, (64,64))	
		images.append(img)
		names.append(files[i])			
		
	print("finished reading images")
	return images,names
	

#tests neural network	
def test(filenames,img):
					
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Create extra synthetic training data by flipping & rotating images
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)


	network = input_data(shape=[None, 64, 64, 3],
						 data_preprocessing=img_prep,
						 data_augmentation=img_aug)

	# 1: Convolution layer with 32 filters, each 3x3x3
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

		
	
	model = tflearn.DNN(network, checkpoint_path='model_cat_dog_7.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
					
	model.load('model_cat_dog_6_final.tflearn')	
	
	preds_cnn=[]
	final_pred=[]
	for i in range (0,len(img)):
		
		img[i] = np.reshape(img[i],(1, 64,64,3))		
		img[i] = img[i].astype('float32')
		probs= model.predict(img[i])
		preds_cnn.append(np.argmax(probs))
		final_pred.append(np.argmax(probs))			
			
	def getint(name):
		basename = name[0].split('.')	
		return int(basename[0])	
		
		
	final_pred=[x for (y,x) in sorted(zip(filenames,final_pred),key=getint)]
	filenames=[y for (y,x) in sorted(zip(filenames,final_pred),key=getint)]
			
	with open('Results.csv', 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile,delimiter=',')
		spamwriter.writerow(["image","label"])
		for i in range (0,len(final_pred)):		
			spamwriter.writerow([filenames[i], final_pred[i]])


imageSet,image_names= readImages()		
test(image_names,imageSet)
	
