import cv2
import numpy as np

# Load the image
img = cv2.imread('sample_data/Women/2/2_woman (5).jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to the grayscale image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply a median blur to the thresholded image to remove noise
blur = cv2.medianBlur(thresh, 7)

# Convert the image to the YCrCb color space
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Apply a skin color range filter to the YCrCb image
lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

# # Combine the thresholded image and the skin color mask
# hand = cv2.bitwise_and(blur, blur, mask=mask)


sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(mask, None)
sift_img = cv2.drawKeypoints(mask, keypoints, None, (255, 0, 0), 4)
print(descriptors.shape)

# Display the result
cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand detection', 800, 600)
cv2.imshow('Hand detection', sift_img)
cv2.waitKey(0)
cv2.destroyAllWindows()