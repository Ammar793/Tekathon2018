import numpy as np
import cv2
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from memory import data_storage
import threading


#contains code in relation to detecting snack
snacks = data_storage.snacks
snack_image = np.zeros(5)
snack_found = False
snack = data_storage.Snack("none", 0)
stop_looking_for_snack = True
model = None
counter_chips = 0
counter_can = 0
counter_chocolate = 0

def set_snack(name):
    global snack
    snack = [x for x in snacks if x.get_name() == name]
    if(len(snack)>0):
        print(snack[0].get_name())
        snack = snack[0]
    else:
        snack = data_storage.Snack(name, 0)


def set_snack_found(boolean):
    global snack_found
    snack_found = boolean


def set_snack_image(image_array):
    global snack_image
    snack_image = image_array


def successfully_found():
    set_snack_found(True)


def load_model():
    global model
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

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy')

    #TODO: change checkpoint path
    model = tflearn.DNN(network, checkpoint_path='model_chips_drinks_chocs_canteen_cp.tflearn', max_checkpoints=3,
                        tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')

    model.load('training/model_chips_drinks_chocs_canteen.tflearn')


class snack_detection_thread(threading.Thread):

    def __init__(self):
        self._stopevent = threading.Event()
        self._sleepperiod = 0.2
        threading.Thread.__init__(self)

    def run(self):
        global snack_image
        global model
        # image preprocessors for neural network input

        while not self._stopevent.isSet( ):
            #print (image)
            image = cv2.resize(snack_image, (64, 64))
            #if (image.all()):

            print ("detecting snacks")
            test(model,  image)

            self._stopevent.wait(self._sleepperiod)

    def join(self, timeout=None):
        """ Stop the thread and wait for it to end. """
        self._stopevent.set()
        threading.Thread.join(self, timeout)


# tests neural network
def test(model,  image):
    global counter_can
    global counter_chips
    global counter_chocolate
    img = np.reshape(image, (1, 64, 64, 3))
    img = img.astype('float32')
    prob = model.predict(img)

    print (prob)
    if prob[0][0] > 0.9:
        if confirm(0):
            set_snack("chips")
            set_snack_found(True)
            reset_counters()

    elif prob[0][1] > 0.9:
        if confirm(1):
            set_snack("drink")
            set_snack_found(True)
            reset_counters()

    elif prob[0][2] > 0.9:
        if confirm(2):
            set_snack("chocolate")
            set_snack_found(True)
            reset_counters()

    else:
        set_snack_found(False)


def reset_counters():
    global counter_can
    global counter_chips
    global counter_chocolate

    counter_chips = 0
    counter_can = 0
    counter_chocolate = 0


def confirm(item):
    global counter_can
    global counter_chips
    global counter_chocolate

    if item ==0:
        counter_chips += 1
        counter_can = 0
        counter_chocolate = 0
        if counter_chips == 10:
            return True
    elif item==1:
        counter_can+=1
        counter_chips = 0
        counter_chocolate = 0
        if counter_can == 10:
            return True
    elif item==2:
        counter_chocolate +=1
        counter_chips = 0
        counter_can = 0
        if counter_chocolate == 10:
            return True
    return False


def reset():
    global counter_can
    global counter_chips
    global counter_chocolate
    global stop_looking_for_snack

    counter_chips = 0
    counter_can = 0
    counter_chocolate = 0
    set_snack_found(False)
    set_snack_image(np.zeros(5))
    set_snack("none")
    stop_looking_for_snack = True


def stop_thread():
    global stop_looking_for_snack
    stop_looking_for_snack = True

#imageSet, image_names = readImages()
#test(image_names, imageSet)

