
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
# Define the HOG parameters
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
num_orientations = 9

# Initialize the lists for storing the HOG features and labels
features = np.empty((0, 128), dtype=np.float32)
labels = []
data=[]
# Iterate through each subdirectory of the dataset directory
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
        sift = cv2.SIFT_create()
        num_channels = image.shape[-1]

        # Convert the image to grayscale if it has three channels
        if num_channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Already grayscale

        keypoints, descriptors = sift.detectAndCompute(gray, None)
        # lbp = cv2.LBP_create()
        # features = lbp.compute(image)
        # radius = 3
        # n_points = 8 * radius
        # lbp = local_binary_pattern(image, n_points, radius, 'uniform')
        # features = lbp.ravel()

        # data.append(features.flatten())
        # labels.append(sub_dir)
        # hog_features = feature.hog(image, pixels_per_cell=pixels_per_cell,
        #                            cells_per_block=cells_per_block,
        #                            orientations=num_orientations)

        # Add the HOG features and label to the lists
        # features.append(hog_features)
        features=np.append(features,descriptors, axis=0)
        
        labels.append(sub_dir)

# Convert the features and labels to numpy arrays

# features = np.array(data)
# features = np.array(features)
labels = np.array(labels)
print(features.shape)
print(labels.shape)
# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = svm.SVC(kernel="linear")
svm_classifier.fit(train_features, train_labels)

# Predict the labels of the test set using the trained SVM classifier
predicted_labels = svm_classifier.predict(test_features)

# Compute the accuracy of the SVM classifier
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))