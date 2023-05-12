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

# Display the result

cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand detection', 800, 600)
cv2.imshow('Hand detection', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key = len)

min_x, min_y, w, h = cv2.boundingRect(contour)
new_img = np.zeros((h, w), dtype=np.uint8)

contour = contour - [min_x, min_y]
cv2.drawContours(new_img, [contour], 0, 255, -1)

new_img= cv2.resize(new_img,(128,128))


# Display the result

cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand detection', 800, 600)
cv2.imshow('Hand detection', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()