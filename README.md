## 📝 Table of Contents

- [About ](#about-)
- [Description ](#Description-)
- [Accuracies ](#Accuracies-)
- [Contributors ](#contributors-)

## About <a name = "about"></a>

In the Hand Gesture Recognition System project, we aim to develop a comprehensive machine learning pipeline capable of accurately classifying hand gestures into six digits (0 to 5). Our system is designed to handle variations in lighting effects and hand poses, ensuring robust performance in real-world scenarios.

## Description <a name = "Description"></a>
The project includes the following key modules:
- Data Acquisition: To train and evaluate our system, we collect a dataset of hand gesture images. We capture images containing a single hand with different gestures, ensuring diversity in lighting conditions and hand poses. The dataset serves as the foundation for subsequent stages of the pipeline.
- Preprocessing: Preprocess the images to enhance the quality and normalize them. Common preprocessing techniques include resizing the images to a consistent size, converting them to grayscale, applying image filters, and normalizing pixel values.
- Feature Extraction: Here we used different features and calculate their efficiency in detecting digits, These features serve as input for the subsequent classification module.
- Classification Model: We train a machine learning model to classify hand gestures into the six digit categories (0 to 5). Various classification algorithms can be employed, such as Support Vector Machines (SVMs), or Random Forests. The model is trained using the preprocessed images and their corresponding labels.
- Training and Evaluation: We split the dataset into training and validation sets. The model is trained using the training set, and the hyperparameters are fine-tuned to optimize performance. The validation set is used to evaluate the model's accuracy, precision, recall, and F1-score. Iterative training and evaluation help improve the model's performance.

## Accuracies <a name = "Accuracies"></a>
### HOG 80.0439% classifier: SVM
  `hog_cut.py`
  - resize image (128,128)
  - use mask after YCrCb conversion
  - get max contour 
  - draw the hand on new image to cut some of unwanted background
  - resize new image (128,128)

### HOG 80.0439% classifier: SVM and adaboost
  `adaboost.py`

### HOG 78.728% -> classifier: Random forest
  `hog_random_forest.py`
  - same as `hog_cut.py` 

### HOG 78.28947368421053% -> classifier: MLP
  `mlp.py`
  - same as `hog_cut.py` 
### HOG 75.43859649122807% -> soft voting (classifier: MLP , RDF , SVM)
  `soft.py`
  - same as `hog_cut.py` 
### HOG 72.37%
  `hog_opencv.py`
  - resize image (128,128)
  - use mask after YCrCb conversion

### HOG & LBP 63.4%
  `lbp.py`
  - resize image (128,128)
  - use mask after YCrCb conversion

### PCA & HOG 54.17%
  `pca_feature.py`
  - resize image (128,128)
  - use mask after YCrCb conversion

### HOG  50.22% -> classifier: decision tree
  `decison_tree.py`
  - resize image (128,128)
  - use mask after YCrCb conversion
  - get max contour 
  - draw the hand on new image to cut some of unwanted background
  - resize new image (128,128)

### sift  44% -> classifier: svm
  `.py`

## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/asmaaadel0" target="_black">
    <img src="https://avatars.githubusercontent.com/u/88618793?s=400&u=886a14dc5ef5c205a8e51942efe9665ed8fd4717&v=4" width="150px;" alt="Asmaa Adel"/>
    <br />
    <sub><b>Asmaa Adel</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/Samaa-Hazem2001" target="_black">
    <img src="https://avatars.githubusercontent.com/u/82514924?v=4" width="150px;" alt="Asmaa Adel"/>
    <br />
    <sub><b>Samaa Hazem</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/norhanreda" target="_black">
    <img src="https://avatars.githubusercontent.com/u/88630231?v=4" width="150px;" alt="norhan reda"/>
    <br />
    <sub><b>Norhan reda</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/Hoda233" target="_black">
    <img src="https://avatars.githubusercontent.com/u/77369927?v=4" width="150px;" alt="HodaGamal"/>
    <br />
    <sub><b>HodaGamal</b></sub></a>
    </td>
  </tr>
 </table>

