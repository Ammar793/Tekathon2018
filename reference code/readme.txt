Readme File

ECSE 415 - Final project
Authors: Muhammad Ammar Raufi, Joel Lat

---------------------------------------------------------------------------------


Details of our procedure followed can be found in our report which is the file 'Report.pdf'. We tried multiple techniques but out final submission is 
made using convolution neural networks. 

The file 'Results.csv' has our final predictions on the provided test set, using CNN. 

The files model_cat_dog_6_final.tflearn.* are the output of the CNN models produced using the tflearn library which is based on tensorflow. 

----------------------------------------------------------------------------------

If you want to test on another set of images, you can do so using the file 'testing.py'. You may need to install tensorflow and the tflearn libraries 
if you don't have them already. 

Change the 'folder' variable  ( on line 24 ) in the method readImages() to the folder where the testing images are.

On running the code, the 'Results.csv' file will be updated with the new predictions. 

----------------------------------------------------------------------------------

If there is a problem or you have any questions you can email muhammad.raufi@mail.mcgill.ca or joel.lat@mail.mcgill.ca 