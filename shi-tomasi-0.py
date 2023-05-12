import cv2
import numpy as np

# Load the input image
img = cv2.imread('dataset_sample/Women/3/3_woman (1).jpg')

# Convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the image to the YCrCb color space
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Apply a skin color range filter to the YCrCb image
lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

# Display the output image
cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Corners', 800, 600)
cv2.imshow('Corners', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Set the maximum number of corners to detect
max_corners = 1000

# Set the quality level of the corners
quality_level = 0.01

# Set the minimum distance between the corners
min_distance = 2

# Detect the corners using the Shi-Tomasi algorithm
corners = cv2.goodFeaturesToTrack(mask, max_corners, quality_level, min_distance)
print(corners.flatten())
print(len(corners.flatten()))

# Draw the detected corners on the input image
for corner in corners:
    x, y = corner.ravel()
    x=int(x)
    y=int(y)
    cv2.circle(img, (x, y), 5, (0, 0, 255), 5)

# Display the output image
cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Corners', 800, 600)
cv2.imshow('Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()