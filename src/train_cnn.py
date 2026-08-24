import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = "data/processed"
DATASET_DIR = "data/dataset"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SHAPE = (61, 73, 61)
BATCH_SIZE = 2
EPOCHS = 30


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(csv_file):

    df = pd.read_csv(csv_file)

    X = []
    y = []

    for _, row in df.iterrows():

        file_id = str(row["FILE_ID"]).strip()
        label = int(row["LABEL"])

        file_path = os.path.join(
            PROCESSED_DIR,
            f"{file_id}_label{label}.npy"
        )

        if not os.path.exists(file_path):
            print("Missing:", file_path)
            continue

        volume = np.load(file_path).astype(np.float32)

        # Verify shape
        if volume.shape != IMG_SHAPE:
            print(
                f"Unexpected shape for {file_id}: "
                f"{volume.shape}"
            )
            continue

        X.append(volume)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Add channel dimension
    X = X[..., np.newaxis]

    return X, y


# ============================================================
# ATTENTION BLOCK
# ============================================================

def attention_block(x):

    attention = layers.Conv3D(
        filters=1,
        kernel_size=1,
        activation="sigmoid",
        padding="same"
    )(x)

    x = layers.Multiply()([x, attention])

    return x


# ============================================================
# BUILD 3D CNN + ATTENTION MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(*IMG_SHAPE, 1)
    )

    # --------------------------------------------------------
    # CNN BLOCK 1
    # --------------------------------------------------------

    x = layers.Conv3D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling3D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # CNN BLOCK 2
    # --------------------------------------------------------

    x = layers.Conv3D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling3D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # ATTENTION
    # --------------------------------------------------------

    x = attention_block(x)


    # --------------------------------------------------------
    # CNN BLOCK 3
    # --------------------------------------------------------

    x = layers.Conv3D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling3D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.5
    )(x)


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)


    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )


    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

X_train, y_train = load_dataset(
    os.path.join(
        DATASET_DIR,
        "train.csv"
    )
)

print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

X_val, y_val = load_dataset(
    os.path.join(
        DATASET_DIR,
        "validation.csv"
    )
)

print("Validation data:", X_val.shape)
print("Validation labels:", y_val.shape)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test, y_test = load_dataset(
    os.path.join(
        DATASET_DIR,
        "test.csv"
    )
)

print("Test data:", X_test.shape)
print("Test labels:", y_test.shape)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding 3D CNN + Attention model...")

model = build_model()

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            "best_cnn_attention.keras"
        ),
        monitor="val_loss",
        save_best_only=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# TEST
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# PREDICTION
# ============================================================

probabilities = model.predict(
    X_test
)

predictions = (
    probabilities.flatten() >= 0.5
).astype(int)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Control",
            "Autism"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "best_cnn_attention.keras"
)

model.save(model_path)

print("\nModel saved:")
print(model_path)