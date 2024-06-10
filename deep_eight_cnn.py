import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, ResNet101, MobileNetV2, Xception, EfficientNetB0,
    DenseNet201, InceptionV3, InceptionResNetV2
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Define model configuration
models_info = [
    (ResNet50, 'avg_pool'),
    (ResNet101, 'avg_pool'),
    (MobileNetV2, 'global_average_pooling2d'),
    (Xception, 'avg_pool'),
    (EfficientNetB0, 'top_activation'),
    (DenseNet201, 'avg_pool'),
    (InceptionV3, 'avg_pool'),
    (InceptionResNetV2, 'avg_pool')
]

# Function to extract features using the model
def extract_features(model_class, layer_name, img_path):
    model = model_class(weights='imagenet', include_top=True)
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    model_layer = model.get_layer(layer_name).output
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model_layer)
    features = feature_extractor.predict(img_data)
    return features.flatten()

# List of image files
image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

X = []
y = []

for file in image_files:
    try:
        label = int(file[0])  # Assuming the label is the first character of the file name
        y.append(label)
        file_features = []

        for model_class, layer in models_info:
            features = extract_features(model_class, layer, file)
            file_features.extend(features)

        X.append(file_features)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the features
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + np.finfo(float).eps)

# Feature selection
selector = SelectKBest(mutual_info_classif, k=1000)
X_selected = selector.fit_transform(X, y)

# Cross-validation for k-NN
kf = StratifiedKFold(n_splits=5)
knn = KNeighborsClassifier(n_neighbors=1, metric='cityblock')

losses = []
for train_index, test_index in kf.split(X_selected, y):
    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn.fit(X_train, y_train)
    loss = 1 - knn.score(X_test, y_test)
    losses.append(loss)

min_loss = min(losses)
best_features = selector.get_support(indices=True)

# Select best features
X_best = X[:, best_features[:int(min_loss * len(best_features))]]  # Use proportionate to min_loss

# Train SVM with the selected features
svm = SVC(kernel='poly', degree=3, gamma='auto', C=1, decision_function_shape='ovo')
svm.fit(X_best, y)

# Save the model and selected features
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(best_features, 'selected_features.pkl')

print("Processing complete and models saved.")