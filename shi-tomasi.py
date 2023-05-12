import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib
import cv2

# Define the directory where the hand gesture images are stored
# dataset_dir = "dataset\Woman"
dataset_dir = "dataset_sample\Women"

labels = []
features=[]


lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])

max_corners = 500
quality_level = 0.01
min_distance = 10

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
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        corners = cv2.goodFeaturesToTrack(mask, max_corners, quality_level, min_distance)

        print(len(corners.flatten()))

        features.append(corners.flatten())
        print(sub_dir)
        labels.append(sub_dir)
        

features = np.array(features)
labels = np.array(labels) 
print(labels.shape)

# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

print('Shape of train_images:', train_features.shape)
print('Shape of train_labels:', train_labels.shape)
print('Shape of test_images:', test_features.shape)
print('Shape of test_labels:', test_labels.shape)


# Train a Support Vector Machine (SVM) classifier
svm_classifier = svm.SVC(kernel="linear")
svm_classifier.fit(train_features, train_labels)

# # Save the trained classifier to a file
# joblib.dump(svm_classifier, 'svm_classifier.joblib')

# Predict the labels of the test set using the trained SVM classifier
predicted_labels = svm_classifier.predict(test_features)

# Compute the accuracy of the SVM classifier
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy: {:.4f}%".format(accuracy * 100))