import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

sys.path.append("src")

from preprocessing import load_image_dataset
from cnn_model import build_se_cnn


DATA_PATH = "data"

MODEL_PATH = "models/se_cnn.keras"

OUTPUT_PATH = "outputs/plots"


os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


print("=" * 60)
print("ASD DETECTION - SE CNN")
print("=" * 60)


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("\nLoading dataset...")

X, y = load_image_dataset(
    DATA_PATH
)

print(
    "Dataset shape:",
    X.shape
)

print(
    "Labels shape:",
    y.shape
)


# ------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------
# BUILD MODEL
# ------------------------------------------------

print("\nBuilding SE-CNN...")

model = build_se_cnn(
    input_shape=(128, 128, 1)
)

model.summary()


# ------------------------------------------------
# TRAIN
# ------------------------------------------------

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,

    validation_split=0.2,

    epochs=20,

    batch_size=16,

    verbose=1
)


# ------------------------------------------------
# EVALUATION
# ------------------------------------------------

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------

model.save(
    MODEL_PATH
)

print(
    "\nModel saved:",
    MODEL_PATH
)


# ------------------------------------------------
# TRAINING GRAPH
# ------------------------------------------------

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("SE-CNN Training Accuracy")

plt.legend()

plt.savefig(
    f"{OUTPUT_PATH}/training_accuracy.png"
)

plt.close()


print("\nTraining completed.")