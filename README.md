# Deep Feature Extraction with Iterative Neighborhood Component Analysis (INCA)
This repository contains a Python script for deep feature extraction using 8 pretrained Convolutional Neural Networks (CNNs) and performing feature selection with Iterative Neighborhood Component Analysis (INCA). The selected features are then used to train a Support Vector Machine (SVM) classifier.

## Features

- *Deep Feature Extraction*: Utilizes 8 different pretrained CNNs to extract deep features from images.
- *Feature Selection*: Implements Iterative Neighborhood Component Analysis (INCA) to select the most informative features.
- *Cross-Validation*: Uses Stratified K-Fold cross-validation to evaluate the performance of k-NN classifier during feature selection.
- *SVM Training*: Trains a Support Vector Machine (SVM) classifier with the selected features.
- *Model Saving*: Saves the trained SVM model and the selected features for future use.

## CNN Models Used

- ResNet50
- ResNet101
- MobileNetV2
- Xception
- EfficientNetB0
- DenseNet201
- InceptionV3
- InceptionResNetV2

## How It Works

### 1. Feature Extraction
The script extracts deep features from images using the specified layers of the pretrained CNN models. Each image is processed to generate a feature vector.

### 2. Data Preparation
- *Image Loading*: The script loads image files from the current directory.
- *Label Extraction*: Assumes the label is the first character of the file name.
- *Normalization*: The extracted features are normalized.

### 3. Feature Selection
- *SelectKBest*: Initially reduces the feature space using mutual information.
- *Iterative Neighborhood Component Analysis (INCA)*: Further refines the feature selection using cross-validation with k-NN classifier.

### 4. Model Training
- *SVM Classifier*: Trains an SVM classifier with polynomial kernel using the selected features.

### 5. Model Saving
The trained SVM model and the selected features are saved for future use.

## How to Use

1. *Clone the Repository*:
   sh
   git clone https://github.com/abdulvahapmutlu/deep-feature-extraction-inca.git
   cd deep-feature-extraction-inca
   

2. *Prepare Your Image Files*: Place your .jpg, .jpeg, and .png image files in the repository directory.

3. *Install Dependencies*:
   sh
   pip install numpy tensorflow scikit-learn joblib
   

4. *Run the Script*:
   sh
   python deep_eight_cnn.py

