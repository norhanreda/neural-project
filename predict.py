
import os
import cv2
import time
import joblib
import numpy as np
import re
# Set the directory path that contains the images
directory = "data"
# Define the HOG parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

# Skin ranges
lower_skin = np.array([0, 135, 85])
upper_skin = np.array([255, 180, 135])

# Load the trained model 
clf = joblib.load('svm_all_n_remove.joblib')

# Loop through all the files in the directory
with open('results.txt', 'w') as r: 
    with open('time.txt', 'w') as t:

        # Get a list of all files in the directory
        files = os.listdir(directory)
        # new_files = []
        # for filename in files:
        #     # Remove parentheses from the filename
        #     new_filename = re.sub(r'\(|\)', '', filename)

        #     # Rename the file with the new filename
        #     os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

        #     # Append the new filename to the list
        #     new_files.append(new_filename)
        # Sort the files alphabetically
        sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else -1)

        for filename in sorted_files:

            # Check if the file is an image
            if filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):

                # Load the image file
                img_path = os.path.join(directory, filename)

                # Read the image
                image = cv2.imread(img_path)

                # Record the start time
                start_time = time.time()

                # image pre-processing -> Binary Mask
                image= cv2.resize(image,(128,128))
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                new_img = cv2.inRange(ycrcb, lower_skin, upper_skin)
                contours, _ = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

                # Predict the label
                predicted_labels = clf.predict(hog_features.reshape(1, -1))

                # Record the end time
                end_time = time.time()

                # Calculate the execution time
                execution_time = end_time - start_time

                # Write in files
                r.write(predicted_labels[0]+'\n')
                t.write(f"{execution_time:.3f}"+'\n')

                print(img_path,' ', predicted_labels[0])