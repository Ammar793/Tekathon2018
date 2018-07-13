# TRAINING THE NEURAL NETWORK

## In order to train the Neural Network follow the instructions below:

1. get all the picture files from each category in the same folder

2. make a csv file that lists all the image names in one column followed by 0 or 1 in the second column, where 0 means chips and 1 means soft drink. e.g

img1.jpg 0\
img2.jpg 1

means img1.jpg is a picture of chips and img2.jpg is a picture of soft drink

3. The file with the training code is located in training/cnn.py. You will need to modify the following variables;

file_name : replace this with the full path of the file above with the list of all images\
folder_with_images: replace this with folder where all the images are located

4. the output file "model_chips_drinks_6_final.tflearn" should be your trained model

5. In the application, the trained model is defined in the file "snack_detection.py", you can modify it there and then test it to see if the output is accurate.

