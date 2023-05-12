import cv2
import numpy as np

# Load the image
img = cv2.imread('dataset_sample/Women/3/3_woman (1).jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the image to the YCrCb color space
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Apply a skin color range filter to the YCrCb image
lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

# Replace white pixels in gray_image with corresponding pixel values in binary_mask
result_image = cv2.bitwise_and(gray, mask)

cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand detection', 800, 600)
cv2.imshow('Hand detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours in the result image
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
max_contour = max(contours, key=cv2.contourArea)

# Crop the image to the bounding box of the contour
x, y, w, h = cv2.boundingRect(max_contour)
result_image = result_image[y:y+h, x:x+w]

result_image= cv2.resize(result_image,(128,128))

# Display the result

cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand detection', 800, 600)
cv2.imshow('Hand detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
