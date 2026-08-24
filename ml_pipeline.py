import os
import sys

import joblib

from sklearn.model_selection import train_test_split

sys.path.append("src")

from preprocessing import load_image_dataset

from feature_extraction import extract_features

from classifiers import (
    train_random_forest,
    train_logistic_regression,
    train_knn,
    save_classifier
)

from evaluate import evaluate_classifier


CNN_MODEL = "models/se_cnn.keras"

RF_MODEL = "models/random_forest.pkl"

LR_MODEL = "models/logistic_regression.pkl"

KNN_MODEL = "models/knn.pkl"


print("=" * 60)
print("CNN + MACHINE LEARNING PIPELINE")
print("=" * 60)


# --------------------------------------------
# LOAD DATA
# --------------------------------------------

X, y = load_image_dataset(
    "data"
)

print(
    "\nDataset:",
    X.shape
)


# --------------------------------------------
# SPLIT DATA
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------
# CNN FEATURE EXTRACTION
# --------------------------------------------

print("\nExtracting CNN features...")

X_train_features = extract_features(
    CNN_MODEL,
    X_train
)

X_test_features = extract_features(
    CNN_MODEL,
    X_test
)


print(
    "Feature shape:",
    X_train_features.shape
)


# --------------------------------------------
# RANDOM FOREST
# --------------------------------------------

print("\nTraining Random Forest...")

rf = train_random_forest(
    X_train_features,
    y_train
)

rf_results = evaluate_classifier(
    rf,
    X_test_features,
    y_test
)

save_classifier(
    rf,
    RF_MODEL
)


# --------------------------------------------
# LOGISTIC REGRESSION
# --------------------------------------------

print("\nTraining Logistic Regression...")

lr = train_logistic_regression(
    X_train_features,
    y_train
)

lr_results = evaluate_classifier(
    lr,
    X_test_features,
    y_test
)

save_classifier(
    lr,
    LR_MODEL
)


# --------------------------------------------
# KNN
# --------------------------------------------

print("\nTraining KNN...")

knn = train_knn(
    X_train_features,
    y_train
)

knn_results = evaluate_classifier(
    knn,
    X_test_features,
    y_test
)

save_classifier(
    knn,
    KNN_MODEL
)


print("\n================================")
print("PIPELINE COMPLETED")
print("================================")

print("\nModels saved:")
print(RF_MODEL)
print(LR_MODEL)
print(KNN_MODEL)