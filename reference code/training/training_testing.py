import numpy as np
import cv2
import os
import csv
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import grid_search
from scipy.signal import butter, lfilter
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib.pyplot as plt

from scipy import stats
import time
from sklearn.externals import joblib
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

#read images from respective folders
#type =1 indicates training
#type =2 indicates testing
def readImages(type):
	#read csv file to get labels for image
	
	print("starting reading images")
	images = [[] for x in range (2)]
	names = [[] for x in range (2)]
	first = True
	
	
	if (type==1):#train
		file = 'Y_Train.csv'
		folder = "./segmented/"
		#change value according to how many files you want to read, value of 1000 will read 1000 cat images and 1000 dog images
		value=5000
	else:
		file = 'Y_Test.csv'
		folder = "./segmented_test/"
		#change value according to how many files you want to read, value of 1000 will read 1000 cat images and 1000 dog images
		value=20000
	
		
	with open(file, 'r') as csvfile:
		spamreader = csv.reader(csvfile)		
		counter1 = 0
		counter2 = 0
		for row in spamreader:
			if first:
				first = False
				continue
			if(row[1]==''):
				row[1]='1'
			if(int(row[1])==0 and counter1 <=value):
				img = cv2.imread (folder+row[0])
				img = cv2.resize(img, (64,64))	
				images[int(row[1])].append(img)
				names[int(row[1])].append(row[0])
				
				counter1+=1		
			
			if(int(row[1])==1 and counter2 <=value):
				img = cv2.imread (folder+row[0])
				img = cv2.resize(img, (64,64))	
				images[int(row[1])].append(img)
				names[int(row[1])].append(row[0])
				
				counter2+=1						
			
	print(len(images[0]))
	print(len(images[1]))
	print("finished reading images")
	return images,names
	
	
#creates bow features and returns 500 cluster centers
def createbow(imgs):
	print(len(imgs))
	BOW = cv2.BOWKMeansTrainer(500)
	
	print("starting sifting")
	
	sift = cv2.xfeatures2d.SIFT_create()
	counter_fail=0
	for i in range (0,len(imgs)):
		kp, des = sift.detectAndCompute (imgs[i],None)
		if(len(kp)>0):
			BOW.add(des)
	dic = BOW.cluster()
	return dic
	

#makes histogram for each image
def make_histos(imgs,dic):
	imnum= len(imgs)
	hist = np.zeros((imnum,500))	
	sift = cv2.xfeatures2d.SIFT_create()	
	print("starting making histos")
	
	#counter_fail=0
	
	for i in range(0,imnum):
		print(i)
		kp, des = sift.detectAndCompute (imgs[i],None)

		for j in range (0,len(kp)):
			dists2 = np.zeros(500)
			for k in range (0,500):
				#calculates eucledian distance from each center
				dists2[k]=(np.linalg.norm(des[j]-dic[k]))	
			#index of minimum distance
			l=np.argmin(dists2)
			hist[i,l] += 1
			
		cv2.normalize (hist[i],hist[i])
		#print(hist[i])
	
	print("done making histos")
	return hist
	
	
#trains neural network
def train(hist,labels):
	clf = MLPClassifier(activation = 'tanh', alpha = 0.01, learning_rate_init=0.0001, hidden_layer_sizes=(400))
	clf_svc = SVC (kernel='rbf', C=1000, gamma=0.0001)
	clf_ran= RandomForestClassifier(max_depth=100, n_estimators=300)
	
	
	print(labels)
	labels2=np.array(labels)
	print(labels2)
	y = label_binarize(labels2, classes=[0,1])
	print(y.shape)
	n_classes = y.shape[1]
	print(hist.shape)
	X_train, X_test, y_train, y_test = train_test_split(hist,y, test_size=.5)
	
	#y_score = clf.fit(X_train, y_train).decision_function(X_test)
	y_score2 = clf_svc.fit(X_train, y_train).decision_function(X_test)
	#y_score3 = clf_ran.fit(X_train, y_train).decision_function(X_test)
	
	#fpr = dict()
	tpr = dict()
	roc_auc = dict()

	
	#fpr, tpr, _ = roc_curve(y_test, y_score)
	fpr2, tpr2, _ = roc_curve(y_test, y_score2)
	#fpr3, tpr3, _ = roc_curve(y_test, y_score3)
	#roc_auc = auc(fpr, tpr)
	roc_auc2 = auc(fpr2, tpr2)
	#roc_auc3 = auc(fpr3, tpr3)
	
		

	plt.figure()
	lw = 2
	#plt.plot(fpr, tpr, color='darkorange',
	#		 lw=lw, label='ROC curve MLP (area = %0.2f)' % roc_auc)
	plt.plot(fpr2, tpr2, color='green',
			 lw=lw, label='ROC curve SVM(area = %0.2f)' % roc_auc2)
			 
	#plt.plot(fpr3, tpr3, color='blue',
	#		 lw=lw, label='ROC curve Random Forest(area = %0.2f)' % roc_auc2)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")	
	plt.show()


	clf.fit(hist,labels)
	clf_svc.fit (hist,labels)
	clf_ran.fit (hist,labels)
	
	
	joblib.dump(clf_svc, ('classifier_svc_segmented.pkl'))
	joblib.dump(clf_ran, ('classifier_ran_segmented.pkl'))
	joblib.dump(clf, ('neural2_segmented.pkl'))
	

#tests neural network	
def test(hist,labels,filenames,img):



					
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

		
	clf = joblib.load('neural2_segmented.pkl')
	clf_svc = joblib.load('classifier_svc_segmented.pkl')
	clf_ran = joblib.load('classifier_ran_segmented.pkl')	
	
	model = tflearn.DNN(network, checkpoint_path='model_cat_dog_7.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
					
	model.load('model_cat_dog_6_final.tflearn')	
	
	preds = []
	preds_svc = []
	preds_ran = []
	preds_cnn=[]
	for i in range (0,hist.shape[0]):
		a=hist[i]
		a= a.reshape(1,-1)
		preds.append(clf.predict(a))
		preds_svc.append(clf_svc.predict(a))		
		preds_ran.append(clf_ran.predict(a))
		img[i] = np.reshape(img[i],(1, 64,64,3))
		
		img[i] = img[i].astype('float32')
		probs= model.predict(img[i])
		preds_cnn.append(np.argmax(probs))
		
		
		#print(clf_svc.predict(a))
		
	counter1=0
	counter2=0	
	counter3=0
	counter4=0
	counter5=0
	counter6=0
	counter7=0
	counter8=0
	counter9=0
	counter10=0
	counter11=0
	counter12=0
	final_pred=[]
	for i in range (0,hist.shape[0]):
		#print(preds[i])
		#print(labels[i])
		if(preds[i]==labels[i]):
			counter1+=1
		else:
			counter2+=1
		
		
		if(preds_svc[i]==labels[i]):
			counter3+=1
		else:
			counter4+=1		
			
		if(preds_ran[i]==labels[i]):
			counter5+=1
		else:
			counter6+=1		
			
		if(preds_cnn[i]==labels[i]):
			counter11+=1
		else:
			counter12+=1
			
		
		if((int(preds[i]) + int(preds_svc[i]) +int(preds_ran[i])) ==  3):
			final_pred.append(1)
			if(int(labels[i])==1):				
				counter9+=1
			else:
				counter10+=1
		elif (int(preds[i]) + int(preds_svc[i]) +int(preds_ran[i]) ==  0):
			final_pred.append(0)
			if(int(labels[i])==0):				
				counter9+=1
			else:
				counter10+=1
		else:
			ok = np.random.randint(1,2)
			final_pred.append(ok)
			if(ok==int(labels[i])):				
				counter9+=1
			else:
				counter10+=1
				
		print(len(final_pred))
		if((int(preds[i]) + int(preds_svc[i]) +int(preds_ran[i])) ==  2 or (int(preds[i]) + int(preds_svc[i]) +int(preds_ran[i])) ==  3 ):
			if(int(labels[i])==1):
				counter7+=1
			else:				
				counter8+=1			
		elif((int(preds[i]) + int(preds_svc[i]) +int(preds_ran[i])) ==  0 or (int(preds[i]) + int(preds_svc[i]) +int(preds_ran[i])) ==  1 ):
			if(int(labels[i])==0):
				counter7+=1
			else:
				counter8+=1
			
			
	def getint(name):
		print(name[0])
		basename = name[0].split('.')	
		return int(basename[0])

	
	final_pred=[x for (y,x) in sorted(zip(filenames,final_pred),key=getint)]
	filenames=[y for (y,x) in sorted(zip(filenames,final_pred),key=getint)]
			
	with open('Resultss.csv', 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile,delimiter=',')
		spamwriter.writerow(["image","label"])
		for i in range (0,len(preds)):		
			spamwriter.writerow([filenames[i], final_pred[i]])
			
	accuracy = (counter1 /(counter1+counter2))*100
	accuracy_svc = (counter3 /(counter3+counter4))*100
	accuracy_ran = (counter5 /(counter5+counter6))*100
	accuracy_vote = (counter7 /(counter7+counter8))*100
	accuracy_unan = (counter9 /(counter9+counter10))*100
	accuracy_cnn = (counter11 /(counter11+counter12))*100
	
	print ("mlp accuracy: %d" %accuracy)
	print("svm accuracy: %d" %accuracy_svc)
	print("random forest accuracy: %d" %accuracy_ran)
	print("voting accuracy: %d" %accuracy_vote)
	print("unanimous voting accuracy: %d" %accuracy_unan)
	print("cnn accuracy: %d" %accuracy_cnn)
	

bow_dic = []

def grid_searchh(hist,labels):
	
	clf = MLPClassifier(hidden_layer_sizes=(250))
	
	gs = grid_search.GridSearchCV(clf, param_grid={
		'learning_rate': [0.01, 0.001,0.0001,0.00001],
		'hidden0__units': [4, 8, 12],
		'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
		
	gs.fit(hist, labels)
	print(gs.best_params_)


def training():	
	global bow_dic
	imageSet,names = readImages(1)
	labels = [0] * len(imageSet[0]) + [1] *len(imageSet[1])
	#labels = [0] * 1903 + [1] *2001

	all_images = imageSet[0].copy()
	all_images.extend(imageSet[1])	
	#bow_dic = createbow(all_images)
	
	bow_dic = joblib.load('bow_dic_segmented.pkl')	
	all_histos = joblib.load('all_histos_segmented.pkl')	
	#saves bow vocab to disk
	#joblib.dump(bow_dic, ('bow_dic_segmented.pkl'))
	
	
	#cat_histos = make_histos(imageSet[0],bow_dic)
	#dog_histos = make_histos(imageSet[1],bow_dic)

	#all_histos=np.append(cat_histos, dog_histos, axis=0)
	#joblib.dump(all_histos, ('all_histos_segmented.pkl'))
	
	train(all_histos,labels)
	#grid_searchh(all_histos,labels)
	
training()

#-------------------------------------------------------------------------------------------------


def testing():
	
	#reads bow vocab from disk
	bow_dic = joblib.load('bow_dic_segmented.pkl')
	all_histos = joblib.load('all_histos_test_segmented.pkl')

	imageSet,image_names= readImages(2)	
	labels = [0] * len(imageSet[0]) + [1] *len(imageSet[1])
	#labels = [0] * 101 + [1] *101
	
	all_images = imageSet[0].copy()
	all_images.extend(imageSet[1])
	all_names = image_names[0].copy()
	all_names.extend(image_names[1])
	#bow_dic = createbow(all_images)

	#cat_histos = make_histos(imageSet[0],bow_dic)
	#dog_histos = make_histos(imageSet[1],bow_dic)

	#all_histos=np.append(cat_histos, dog_histos, axis=0)
	#joblib.dump(all_histos, ('all_histos_test_unsegmented.pkl'))
	#joblib.dump(bow_dic, ('bow_dic_test_unsegmented.pkl'))
	test(all_histos,labels,all_names,all_images)
	
testing()