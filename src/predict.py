import sys

import cv2
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from preprocessing import preprocess_image

from feature_extraction import create_feature_extractor


CNN_PATH = "models/se_cnn.keras"

RF_PATH = "models/random_forest.pkl"


def predict_image(
    image_path
):

    # Load image

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError(
            "Unable to read image."
        )


    # Preprocess

    image = preprocess_image(
        image
    )


    # Add dimensions

    image = image[
        np.newaxis,
        ...,
        np.newaxis
    ]


    # Load CNN feature extractor

    extractor = create_feature_extractor(
        CNN_PATH
    )


    # Extract features

    features = extractor.predict(
        image,
        verbose=0
    )


    # Load Random Forest

    classifier = joblib.load(
        RF_PATH
    )


    # Prediction

    prediction = classifier.predict(
        features
    )[0]


    probability = classifier.predict_proba(
        features
    )[0]


    classes = {
        0: "CONTROL",
        1: "ASD"
    }


    label = classes[
        int(prediction)
    ]


    confidence = probability[
        int(prediction)
    ]


    return label, confidence


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python src/predict.py image.png"
        )

        sys.exit(1)


    image_path = sys.argv[1]


    label, confidence = predict_image(
        image_path
    )


    print("\n==============================")

    print(
        "AI PREDICTION"
    )

    print("==============================")

    print(
        "Result:",
        label
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )