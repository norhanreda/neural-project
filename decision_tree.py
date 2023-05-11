import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import skimage.io as io
import os
import numpy as np
from skimage import feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
# Define the directory where the hand gesture images are stored
dataset_dir = "dataset\dataset\Woman"
# dataset_dir = "dataset_sample\Women"
images = []
labels = []
descriptors = []
features=[]
arr=[]
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
        # image = np.asarray(Image.open(image_path))
        image = cv2.imread(image_path)
        image= cv2.resize(image,(128,128))
        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('mask', 800, 600)
        # cv2.imshow('mask', image) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # image = cv2.resize(image, (256, 256))

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Apply a skin color range filter to the YCrCb image
        lower_skin = np.array([0, 135, 85])
        upper_skin = np.array([255, 180, 135])
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('mask', 800, 600)
        # cv2.imshow('mask', mask) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
         # Define HOG parameters
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        # Compute HOG features
        hog_features = hog.compute(mask)
        # print(hog_features.shape)
        # print(hog_features)
        features.append(hog_features)
        print(sub_dir)
        labels.append(sub_dir)
        
        
        
# descriptors = np.vstack(descriptors)
        # descriptors.append(des)
# descriptors = np.array(descriptors)
features = np.array(features)
# total=np.concatenate((descriptors, features), axis=1)
# print('hog',features)
# print('hof shape',features.shape)
# surf_des=np.array(surf_des)
labels = np.array(labels) 
# print(surf_des.shape)
# descriptors = descriptors.reshape(descriptors.shape[0], descriptors.shape[1])
# descriptors = np.reshape(descriptors, (len(labels), -1))
   

# print('sift shape',descriptors.shape)
print(labels.shape)
# print('sift',descriptors)
# for image in images:
#     kp, des = sift.detectAndCompute(image, None)
#     descriptors.append(des)
# descriptors = np.array(descriptors)
# labels = np.array(labels)
# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

print('Shape of train_images:', train_features.shape)
print('Shape of train_labels:', train_labels.shape)
print('Shape of test_images:', test_features.shape)
print('Shape of test_labels:', test_labels.shape)


# Train a decision tree classifier
clf = DecisionTreeClassifier(max_depth=1000, criterion='entropy')
clf.fit(train_features, train_labels)

# Predict the labels of the test set using the trained SVM classifier
predicted_labels =  clf.predict(test_features)

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
