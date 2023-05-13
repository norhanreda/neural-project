import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib
from sklearn.decomposition import PCA
import cv2

# Define the directory where the hand gesture images are stored
dataset_dir = "dataset\Woman"
# dataset_dir = "dataset_sample\Women"
# dataset_dir = "our_data"

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
        if not file_name.endswith(".JPG") and not file_name.endswith(".jpg") and not file_name.endswith(".jpeg"):
            continue
        image_path = os.path.join(sub_dir_path, file_name)
        image = cv2.imread(image_path)

        # colored mask
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        new_img=cv2.bitwise_and(image,image, mask=mask)
        new_img= cv2.resize(new_img,(128,128))
        # x, y, w, h = cv2.boundingRect(contour)
        # new_img = new_img[y:y+h, x:x+w]
        
        # gray mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        new_img=cv2.bitwise_and(image,image, mask=mask)
        result_image = cv2.bitwise_and(gray, mask)
        # x, y, w, h = cv2.boundingRect(contour)
        # new_img = new_img[y:y+h, x:x+w]
        new_img= cv2.resize(new_img,(128,128))

        # binary mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contour = max(contours, key = len)
        # min_x, min_y, w, h = cv2.boundingRect(contour)
        # new_img = np.zeros((h, w), dtype=np.uint8)
        # contour = contour - [min_x, min_y]
        # cv2.drawContours(new_img, [contour], 0, 255, -1)
        new_img= cv2.resize(new_img,(128,128))


        # cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Hand detection', 800, 600)
        # cv2.imshow('Hand detection', new_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        

        


        # cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Hand detection', 800, 600)
        # cv2.imshow('Hand detection', new_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # cv2.namedWindow('Hand detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Hand detection', 800, 600)
        # cv2.imshow('Hand detection', new_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        # Compute HOG features
        hog_features = hog.compute(new_img)
        # print(hog_features.shape)

        # hog_features = hog_features.reshape(108,-1)
        # print(hog_features.shape)

        # pca = PCA(n_components=108)
        # hog_features = pca.fit_transform(hog_features)
        # print(hog_features.shape)

        # features.append(hog_features.flatten())

        features.append(hog_features)
        print(sub_dir)
        labels.append(sub_dir)
        

features = np.array(features)
print(features.shape)
print(features)
# features=features.reshape(features.shape[1],features.shape[0])
# print(features.shape)
labels = np.array(labels) 
print(labels.shape)

# pca = PCA(n_components=50,svd_solver='auto')
# features = pca.fit_transform(features)
# print(features.shape)

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
joblib.dump(svm_classifier, 'hog_try.joblib')

# Predict the labels of the test set using the trained SVM classifier
predicted_labels = svm_classifier.predict(test_features)

# Compute the accuracy of the SVM classifier
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy: {:.4f}%".format(accuracy * 100))