import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern


import cv2

# Define the directory where the hand gesture images are stored
# dataset_dir = "dataset\Woman"
dataset_dir = "dataset_sample\Women"

labels = []
features=[]

# Define the HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

# range
lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])

for sub_dir in os.listdir(dataset_dir):
    sub_dir_path = os.path.join(dataset_dir, sub_dir)
    if not os.path.isdir(sub_dir_path):
        continue

    # Iterate through each image file in the subdirectory
    for file_name in os.listdir(sub_dir_path):
        if not file_name.endswith(".JPG"):
            continue
        image_path = os.path.join(sub_dir_path, file_name)

        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the image to the YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Apply a skin color range filter to the YCrCb image
        lower_skin = np.array([0, 135, 85])
        upper_skin = np.array([255, 180, 135])
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Replace white pixels in gray_image with corresponding pixel values in binary_mask
        result_image = cv2.bitwise_and(gray, mask)

        # Find contours in the result image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Crop the image to the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)
        result_image = result_image[y:y+h, x:x+w]

        result_image= cv2.resize(result_image,(128,128))


        #Initialize HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        #Compute HOG features
        hog_features = hog.compute(result_image)
        features.append(hog_features)

        print(sub_dir)
        labels.append(sub_dir)
        

features = np.array(features)
labels = np.array(labels) 
print(labels.shape)
print(features.shape)


# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

print('Shape of train_images:', train_features.shape)
print('Shape of train_labels:', train_labels.shape)
print('Shape of test_images:', test_features.shape)
print('Shape of test_labels:', test_labels.shape)


# # Create an AdaBoost classifier with decision tree base estimator
# clf = AdaBoostClassifier(n_estimators=300, random_state=42)

# # Fit the classifier to the training data
# clf.fit(train_features, train_labels)

# predicted_labels = clf.predict(test_features)

# # Compute the accuracy of the SVM classifier
# accuracy = accuracy_score(test_labels, predicted_labels)

# print("Accuracy: {:.4f}%".format(accuracy * 100))

#############################################################################

# Train a Support Vector Machine (SVM) classifier
svm_classifier = svm.SVC(kernel="linear")
svm_classifier.fit(train_features, train_labels)

# Predict the labels of the test set using the trained SVM classifier
predicted_labels = svm_classifier.predict(test_features)

# Compute the accuracy of the SVM classifier
accuracy = accuracy_score(test_labels, predicted_labels)

print("Accuracy: {:.4f}%".format(accuracy * 100))

#############################################################################

# # Create a KNN classifier with k=3
# knn = KNeighborsClassifier(n_neighbors=3)

# # Fit the model using the training data
# knn.fit(train_features, train_labels)

# # Make predictions on the testing data
# predicted_labels = knn.predict(test_features)

# # Compute the accuracy of the knn classifier
# accuracy = accuracy_score(test_labels, predicted_labels)

# print("Accuracy: {:.4f}%".format(accuracy * 100))

