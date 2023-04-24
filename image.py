from skimage.color import rgb2gray, rgb2hsv
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import gaussian
import cv2

import numpy as np
import math

image=io.imread('D:\Astudy\third year\second term\NN\project\dataset_sample\men\0\0_men (1).JPG')
blur = cv2.GaussianBlur(image, (3,3), 0)
    
    # Change color-space from BGR -> HSV
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# Create a binary image with where white will be skin colors and rest is black
#mask2 = cv2.inRange(hsv, np.array([0,0,0]), np.array([20,255,255]))
#     min_YCrCb = np.array([0,133,77],np.uint8)
#     max_YCrCb = np.array([235,173,127],np.uint8)
#     imageYCrCb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
#     mask2 = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
# sensitivity = 20 
# lower_bound = np.array([50 - sensitivity, 100, 60]) 
# upper_bound = np.array([50 + sensitivity, 255, 255]) 
mask2 = cv2.inRange(hsv, np.array([0, 20, 70], dtype = "uint8"), np.array([20, 255, 255], dtype = "uint8"))
io.imshow('mask',mask2)