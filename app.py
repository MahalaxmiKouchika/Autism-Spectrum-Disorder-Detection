import sys
import os
import cv2
import numpy as np
import joblib
import streamlit as st
from PIL import Image

sys.path.append("src")
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image
from feature_extraction import create_feature_extractor

CNN_PATH = "models/se_cnn.keras"
RF_PATH = "models/random_forest.pkl"

st.set_page_config(page_title="ASD Predictor", page_icon="🧠", layout="centered")

st.title("Autism Spectrum Disorder (ASD) Prediction")
st.write("Upload a brain scan image to get a prediction from the AI models.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image via PIL and convert to OpenCV format
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")

    # Convert to grayscale numpy array
    img_array = np.array(image.convert("L"))

    # Preprocess
    img_processed = preprocess_image(img_array)
    
    # Add dimensions for CNN: (1, 128, 128, 1)
    img_input = img_processed[np.newaxis, ..., np.newaxis]

    # Model inference
    try:
        extractor = create_feature_extractor(CNN_PATH)
        features = extractor.predict(img_input, verbose=0)
        
        classifier = joblib.load(RF_PATH)
        prediction = classifier.predict(features)[0]
        probability = classifier.predict_proba(features)[0]
        
        classes = {0: "CONTROL (No ASD detected)", 1: "ASD (Autism Spectrum Disorder detected)"}
        label = classes[int(prediction)]
        confidence = probability[int(prediction)]
        
        st.subheader("Results:")
        if int(prediction) == 1:
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")
            
        st.write(f"Confidence Level: **{confidence * 100:.2f}%**")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Note: Make sure you have trained the models first by running `python train.py` and `python ml_pipeline.py`.")
