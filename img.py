import cv2
import numpy as np

# Read the input image
img = cv2.imread('dataset\dataset\men\0\0_men (1).JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary mask based on color
ret, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Apply morphology operations to remove noise and fill holes
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Invert the mask to obtain the background
background = cv2.bitwise_not(mask)

# Apply the mask to the original image to obtain the foreground
foreground = cv2.bitwise_and(img, img, mask=mask)

# Display the results
cv2.imshow('Input Image', img)
cv2.imshow('Foreground', foreground)
cv2.imshow('Background', background)
cv2.waitKey(0)
cv2.destroyAllWindows()