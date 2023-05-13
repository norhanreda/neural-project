import cv2
import numpy as np

# Load the image
lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])
image = cv2.imread('dataset_sample/Women/3/3_woman (1).jpg')

############################################# colored mask ########################################################
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
new_img=cv2.bitwise_and(image,image, mask=mask)

# to cut
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contour = max(contours, key = len)
# x, y, w, h = cv2.boundingRect(contour)
# new_img = new_img[y:y+h, x:x+w]

new_img= cv2.resize(new_img,(128,128))


############################################# gray mask ########################################################
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
new_img = cv2.bitwise_and(gray, mask)

# to cut
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contour = max(contours, key = len)
# x, y, w, h = cv2.boundingRect(contour)
# new_img = new_img[y:y+h, x:x+w]

new_img= cv2.resize(new_img,(128,128))

############################################# binary mask ########################################################
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
new_img = cv2.inRange(ycrcb, lower_skin, upper_skin)

# to cut
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contour = max(contours, key = len)
# x, y, w, h = cv2.boundingRect(contour)
# new_img = new_img[y:y+h, x:x+w]

new_img= cv2.resize(new_img,(128,128))

# to display
# cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hand detection', 800, 600)
# cv2.imshow('Hand detection', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

