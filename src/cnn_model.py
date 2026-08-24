import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model

from se_block import squeeze_excite_block


def build_se_cnn(
    input_shape=(128, 128, 1)
):

    inputs = layers.Input(
        shape=input_shape
    )

    # =========================================
    # CNN BLOCK 1
    # =========================================

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # =========================================
    # CNN BLOCK 2
    # =========================================

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # =========================================
    # SE ATTENTION
    # =========================================

    x = squeeze_excite_block(x)

    # =========================================
    # CNN BLOCK 3
    # =========================================

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # =========================================
    # FEATURE EXTRACTION
    # =========================================

    x = layers.GlobalAveragePooling2D()(x)

    feature_vector = layers.Dense(
        128,
        activation="relu",
        name="feature_vector"
    )(x)

    x = layers.Dropout(
        0.5
    )(feature_vector)

    # =========================================
    # ASD PREDICTION
    # =========================================

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="prediction"
    )(x)

    # =========================================
    # MODEL
    # =========================================

    model = Model(
        inputs=inputs,
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model