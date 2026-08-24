import numpy as np
from tensorflow.keras.models import Model, load_model


def create_feature_extractor(model_path):
    """
    Creates a model that extracts the
    128-dimensional feature vector
    from the trained SE-CNN.
    """

    model = load_model(model_path)

    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer(
            "feature_vector"
        ).output
    )

    return feature_extractor


def extract_features(model_path, X):
    """
    Extract CNN features from images.
    """

    extractor = create_feature_extractor(
        model_path
    )

    features = extractor.predict(
        X,
        verbose=0
    )

    return features