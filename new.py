import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
import numpy as np
from skimage import feature
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
# Define the directory where the hand gesture images are stored
dataset_dir = "dataset_sample\Women"
images = []
labels = []
descriptors = []
features=[]
# Define the HOG parameters
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
num_orientations = 9
for sub_dir in os.listdir(dataset_dir):
    sub_dir_path = os.path.join(dataset_dir, sub_dir)
    if not os.path.isdir(sub_dir_path):
        continue

    # Iterate through each image file in the subdirectory
    for file_name in os.listdir(sub_dir_path):
        if not file_name.endswith(".JPG"):
            continue
        image_path = os.path.join(sub_dir_path, file_name)

        # Load the image and compute its HOG features
        image = np.asarray(Image.open(image_path).convert("L"))
        # image = cv2.resize(image, (600,400))
        sift = cv2.SIFT_create()
        # surf = cv2.xfeatures2d.SURF_create(128)

        num_channels = image.shape[-1]

        # Convert the image to grayscale if it has three channels
        if num_channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Already grayscale
        kp, des = sift.detectAndCompute(gray, None)

        if des is not None:
         mean=np.mean(des,axis=0)
         descriptors.append(mean)
         labels.append(sub_dir)
        hog_features = feature.hog(image, pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block,
                                orientations=num_orientations)

    # Add the HOG features and label to the lists
        features.append(hog_features)
# descriptors = np.vstack(descriptors)
        # descriptors.append(des)
descriptors = np.array(descriptors)
features = np.array(features)
total=np.concatenate((descriptors, features), axis=1)
print('hog',features)
print('hof shape',features.shape)
# surf_des=np.array(surf_des)
labels = np.array(labels) 
# print(surf_des.shape)
# descriptors = descriptors.reshape(descriptors.shape[0], descriptors.shape[1])
# descriptors = np.reshape(descriptors, (len(labels), -1))
   

print('sift shape',descriptors.shape)
print(labels.shape)
print('sift',descriptors)
# for image in images:
#     kp, des = sift.detectAndCompute(image, None)
#     descriptors.append(des)
# descriptors = np.array(descriptors)
# labels = np.array(labels)
# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    total, labels, test_size=0.25, random_state=42)

print('Shape of train_images:', train_features.shape)
print('Shape of train_labels:', train_labels.shape)
print('Shape of test_images:', test_features.shape)
print('Shape of test_labels:', test_labels.shape)


# Train a Support Vector Machine (SVM) classifier
svm_classifier = svm.SVC(kernel="linear")
svm_classifier.fit(train_features, train_labels)

# Predict the labels of the test set using the trained SVM classifier
predicted_labels = svm_classifier.predict(test_features)

# Compute the accuracy of the SVM classifier
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# # Split data into training and testing sets
# train_descriptors, test_descriptors, train_labels, test_labels = train_test_split(
#     descriptors, labels, test_size=0.2, random_state=42)

# # Train the SVM classifier
# clf = svm.SVC(kernel='linear')
# clf.fit(train_descriptors, train_labels)

# # Load new test images
# test_images = []
# test_dir_path = 'test_images'
# for filename in os.listdir(test_dir_path):
#     img = cv2.imread(os.path.join(test_dir_path, filename))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     test_images.append(gray)

# # Extract SIFT features from test images
# test_descriptors = []
# for image in test_images:
#     kp, des = sift.detectAndCompute(image, None)
#     test_descriptors.append(des)

# test_descriptors = np.array(test_descriptors)

# # Classify test images using SVM classifier
# predicted_labels = clf.predict(test_descriptors)

# # Print predicted labels
# print(predicted_labels)

# # Evaluate accuracy on test set
# accuracy = accuracy_score(test_labels, predicted_labels)
# print("Accuracy:", accuracy)
