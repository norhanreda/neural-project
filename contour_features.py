# import cv2
# import numpy as np

# # Load input image and convert to grayscale
# img = cv2.imread('dataset_sample/Women/3/3_woman (1).jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Convert the image to the YCrCb color space
# ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# # Apply a skin color range filter to the YCrCb image
# lower_skin = np.array([0, 135, 85])
# upper_skin = np.array([255, 180, 135])
# mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

# # Display the result

# cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand detection', 800, 600)
# cv2.imshow('Hand detection', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Threshold the image and find contours
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contour = max(contours, key = len)


# # Extract features from contours and convex hull
# # Calculate area and perimeter
# area = cv2.contourArea(contour)
# perimeter = cv2.arcLength(contour, True)

# # Calculate bounding box and aspect ratio
# x, y, w, h = cv2.boundingRect(contour)
# aspect_ratio = float(w) / h

# # Calculate minimum enclosing circle
# (x, y), radius = cv2.minEnclosingCircle(contour)
# center = (int(x), int(y))
# radius = int(radius)

# # Calculate convex hull and solidity
# hull = cv2.convexHull(contour)
# hull_area = cv2.contourArea(hull)
# solidity = float(area) / hull_area

# # Calculate extent
# rect_area = w * h
# extent = float(area) / rect_area

# # Draw features on the output image
# cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
# cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
# cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# cv2.circle(img, center, radius, (255, 255, 0), 2)

# # Display output image with features
# cv2.namedWindow('Features', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Features', 800, 600)
# cv2.imshow('Features', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the input image and convert it to grayscale
img = cv2.imread('dataset_sample/Women/3/3_woman (1).jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

# # Threshold the grayscale image to obtain a binary mask
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find the contours of the binary mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
largest_contour = max(contours, key=cv2.contourArea)

# Find the convex hull of the largest contour
hull = cv2.convexHull(largest_contour, returnPoints=False)

# Find the convexity defects of the largest contour
defects = cv2.convexityDefects(largest_contour, hull)

print(defects.shape)


# Draw the convex hull and defects on the input image
for i in range(defects.shape[0]):
    s, e, f, d = defects[i][0]
    start = tuple(largest_contour[s][0])
    end = tuple(largest_contour[e][0])
    far = tuple(largest_contour[f][0])
    cv2.line(img, start, end, (0, 255, 0), 2)
    cv2.circle(img, far, 5, (0, 0, 255), -1)

cv2.drawContours(img, [largest_contour], -1, (255, 0, 0), 2)

# Display the output image
cv2.namedWindow('Convexity Defects', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Convexity Defects', 800, 600)
cv2.imshow('Convexity Defects', img)
cv2.waitKey(0)
cv2.destroyAllWindows()