from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib
import cv2
import time

# Load the saved classifier from the file
# clf = joblib.load('svm_classifier.joblib')
import os
from PIL import Image

# Set the directory path that contains the images
directory = "data"
features=[]

# Define the HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])

clf = joblib.load('svm_classifier.joblib')
# Loop through all the files in the directory
with open('results.txt', 'w') as r: 
    with open('time.txt', 'w') as t:
        for filename in os.listdir(directory):
            # Check if the file is an image
            if filename.endswith(".JPG") or filename.endswith(".png"):
                # Load the image file using Pillow
                img_path = os.path.join(directory, filename)
                image = cv2.imread(img_path)
                image= cv2.resize(image,(128,128))
                # Record the start time
                start_time = time.time()

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
                predicted_labels = clf.predict(hog_features.reshape(1, -1))
                # Record the end time
                end_time = time.time()
                # Calculate the execution time
                execution_time = end_time - start_time
                print(f"Execution time: {execution_time:.2f} seconds")

                r.write(predicted_labels[0]+'\n')
                t.write(f"{execution_time:.3f}"+'\n')
            
        # features.append(hog_features)

        # print(img)

# features = np.array(features)
# Load the saved classifier from the file

# Predict the labels of the test set using the trained SVM classifier
# predicted_labels = clf.predict(features)
# Print the predicted labels
# print(predicted_labels)
# with open('results.txt', 'w') as f:
#     for i in range(0,len(predicted_labels)):

        # Write a string to the file
        # f.write(predicted_labels[i]+'\n')

# Compute the accuracy of the SVM classifier
# accuracy = accuracy_score(test_labels, predicted_labels)
# print("Accuracy: {:.4f}%".format(accuracy * 100))