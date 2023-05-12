import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
# Define the directory where the hand gesture images are stored
dataset_dir = "dataset\dataset\Woman"
# dataset_dir = "dataset_sample\Women"

labels = []
features=[]

# Define the HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

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
        image= cv2.resize(image,(128,128))

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key = len)

        min_x, min_y, w, h = cv2.boundingRect(contour)
        new_img = np.zeros((h, w), dtype=np.uint8)

        contour = contour - [min_x, min_y]
        cv2.drawContours(new_img, [contour], 0, 255, -1)

        new_img= cv2.resize(new_img,(128,128))

        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        # Compute HOG features
        hog_features = hog.compute(new_img)
        features.append(hog_features)
        print(sub_dir)
        labels.append(sub_dir)
        

features = np.array(features)
labels = np.array(labels) 
print(labels.shape)

# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=22)

print('Shape of train_images:', train_features.shape)
print('Shape of train_labels:', train_labels.shape)
print('Shape of test_images:', test_features.shape)
print('Shape of test_labels:', test_labels.shape)


# Train a Support Vector Machine (SVM) classifier
clf1 = svm.SVC(probability=True)
clf2 = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
clf3= RandomForestClassifier(n_estimators=200)
# Create the soft voting classifier by combining the three base classifiers
voting_clf = VotingClassifier(estimators=[('svc', clf1), ('mlp', clf2),('rdf', clf3)], voting='soft')

# Train the voting classifier on the training data
voting_clf.fit(train_features, train_labels)

# Evaluate the performance of the voting classifier on the testing data
accuracy = voting_clf.score(test_features, test_labels)
print("Accuracy:", accuracy*100)
# svm_classifier.fit(train_features, train_labels)

# Save the trained classifier to a file
# joblib.dump(svm_classifier, 'svm_classifier.joblib')

# # Predict the labels of the test set using the trained SVM classifier
# predicted_labels = svm_classifier.predict(test_features)

# # Compute the accuracy of the SVM classifier
# accuracy = accuracy_score(test_labels, predicted_labels)
# print("Accuracy: {:.4f}%".format(accuracy * 100))