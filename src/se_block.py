import tensorflow as tf

from tensorflow.keras import layers


def squeeze_excite_block(
    inputs,
    reduction_ratio=16
):
    """
    Squeeze-and-Excitation attention block.

    It learns which feature channels
    are more important.
    """

    channels = inputs.shape[-1]

    # -----------------------------------------
    # SQUEEZE
    # -----------------------------------------

    x = layers.GlobalAveragePooling2D()(inputs)

    # -----------------------------------------
    # EXCITATION
    # -----------------------------------------

    x = layers.Dense(
        max(channels // reduction_ratio, 1),
        activation="relu"
    )(x)

    x = layers.Dense(
        channels,
        activation="sigmoid"
    )(x)

    # Convert to:
    # (batch, 1, 1, channels)

    x = layers.Reshape(
        (1, 1, channels)
    )(x)

    # -----------------------------------------
    # FEATURE RECALIBRATION
    # -----------------------------------------

    output = layers.Multiply()(
        [inputs, x]
    )

    return output