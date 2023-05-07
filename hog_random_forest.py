import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import cv2

# Define the directory where the hand gesture images are stored
dataset_dir = "dataset\Woman"
# dataset_dir = "sample_data\Women"

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

# # Generate some random classification data
# X, y = make_classification(n_samples=1821, n_features=10, n_informative=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

print('Shape of train_images:', X_train.shape)
print('Shape of train_labels:', y_train.shape)
print('Shape of test_images:', X_test.shape)
print('Shape of test_labels:', y_test.shape)

# Create a Random Forest Classifier with 100 trees
rfc = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model on the training set
rfc.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rfc.predict(X_test)

# Evaluate the performance of the model
accuracy = rfc.score(X_test, y_test) * 100
print(f"Accuracy: {accuracy}")

