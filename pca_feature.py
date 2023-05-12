# from sklearn.decomposition import PCA
# import cv2
# import numpy as np

# # read the input image
# img = cv2.imread('dataset_sample/Women/3/3_woman (3).jpg')
# img= cv2.resize(img,(128,128))
# # convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # reshape the image into a 1D array
# X = gray.reshape(-1, 1)

# # apply PCA to the image
# pca = PCA(n_components=1)
# X_pca = pca.fit_transform(X)

# # reshape the PCA features back into an image
# img_pca = X_pca.reshape(gray.shape[0], gray.shape[1], -1)

# # display the output images
# cv2.imshow('Input Image', img)
# cv2.imshow('PCA Image', img_pca)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from sklearn.decomposition import PCA
import cv2
import numpy as np
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import skimage.io as io
import os
import numpy as np
from skimage import feature
from sklearn import svm
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key = len)

        min_x, min_y, w, h = cv2.boundingRect(contour)
        new_img = np.zeros((h, w), dtype=np.uint8)

        contour = contour - [min_x, min_y]
        cv2.drawContours(new_img, [contour], 0, 255, -1)

        new_img= cv2.resize(new_img,(128,128))
        
        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('mask', 800, 600)
        # cv2.imshow('mask', mask) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
         # Define HOG parameters
        # win_size = (64, 64)
        # block_size = (16, 16)
        # block_stride = (8, 8)
        # cell_size = (8, 8)
        # nbins = 9
        # # Initialize HOG descriptor
        # hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        # # Compute HOG features
        # hog_features = hog.compute(mask)
        # print(hog_features.shape)
        # print(hog_features)
            # reshape the image into a 1D array
        X = mask.reshape(-1, 1)

        # apply PCA to the image
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X)

        # reshape the PCA features back into an image
        img_pca = X_pca.reshape(mask.shape[0], mask.shape[1], -1).flatten()
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        # Compute HOG features
        hog_features = hog.compute(mask)
        print(len(hog_features))
        hog_features=np.append( hog_features,X_pca)
        print(len(hog_features))
        features.append( hog_features)
        # features.append(img_pca)
        print(sub_dir)
        labels.append(sub_dir)
        
        
        


features = np.array(features)

labels = np.array(labels) 

print(labels.shape)
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.1, random_state=22)

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

