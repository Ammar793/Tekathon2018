from PIL import Image
import pytesseract
import os
import cv2
import numpy as np
import time

img = cv2.imread("test.jpg")
#height, width = img.shape

#x=width/2
#w=100

#y=height/2.5
#h=100


#crop_img = img[y:y+h, x:x+w]

#cv2.imshow("cropped", crop_img)
#cv2.waitKey(0)

img_height, img_width, img_colors = img.shape

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

edges = cv2.Canny(img, 100, 200)

# result is dilated for marking the corners, not important
# dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
# img[dst > 0.01 * dst.max()] = [0, 0, 255]

filename = "{}.png".format(os.getpid())
#cv2.imwrite(filename, gray)
print("aasdas")
imag = Image.open("C:/Users/mammar/PycharmProjects/Hackathon/ui/test.jpg")
text = pytesseract.image_to_string(imag)
print(text)