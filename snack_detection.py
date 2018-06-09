import numpy as np
import cv2
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import segmentation


#contains code in relation to detecting snack

snack_image = np.zeros(5)
snack_found = False

def set_snack_found(boolean):
    global snack_found
    snack_found = boolean


def set_snack_image(image_array):
    global snack_image
    snack_image = image_array


def reset():
    set_snack_found(False)
    set_snack_image(np.zeros(5))


def successfully_found():
    set_snack_found(True)


def snack_detection_worker():
    global snack_image


    # image preprocessors for neural network input
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping & rotating images
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # setting up neural network
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

    #TODO: change checkpoint path
    model = tflearn.DNN(network, checkpoint_path='model_chips_drinks_7.tflearn', max_checkpoints=3,
                        tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')

    model.load('training/model_chips_drinks_6_final.tflearn')

    while True:
        #print (image)
        image = cv2.resize(snack_image, (64, 64))
        if (image.all() != 0):

            if (segmentation.check_if_item_present(image)):
                print ("detecting snacks")
                test(model,  image)
            else:
                print("nothing present")


# read images from folder
def readImage():
    # read csv file to get labels for image
    print("starting reading images")
    images = []
    names = []
    first = True

    folder = "C:/Users/Ammar Raufi/Desktop/winter 17/computer_vision/final_proj/test/X_Test/"

    files = os.listdir(folder)

    for i in range(0, len(files)):
        img = cv2.imread(folder + '/' + files[i])
        img = cv2.resize(img, (64, 64))
        images.append(img)
        names.append(files[i])

    print("finished reading images")
    return images, names


# tests neural network
def test(model,  image):

    img = np.reshape(image, (1, 64, 64, 3))
    img = img.astype('float32')
    prob = model.predict(img)

    print (prob)



#imageSet, image_names = readImages()
#test(image_names, imageSet)

