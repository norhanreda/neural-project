import cv2
import numpy as np

# Load the image
img = cv2.imread('dataset_sample/Women/3/3_woman (3).jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate the Harris corner detector response
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Threshold the corner response
threshold = 0.01 * dst.max()
corner_mask = dst > threshold

# Draw the detected corners on the original image
img[corner_mask] = [0, 0, 255]

# Display the result
cv2.imshow('Harris Corner Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()