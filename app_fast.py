import sys
import streamlit as st
import random
from PIL import Image

st.set_page_config(page_title="ASD Predictor (Fast Demo)", page_icon="⚡", layout="centered")

st.title("Autism Spectrum Disorder (ASD) Prediction")
st.write("Upload a brain scan image to get a prediction. *(Note: This is the FAST UI demo. It simulates the AI prediction to load instantly without heavy libraries.)*")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image via PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing using Simulated Fast Model...")

    # Simulate prediction
    prediction = random.choice([0, 1])
    confidence = random.uniform(0.75, 0.99)
    
    classes = {0: "CONTROL (No ASD detected)", 1: "ASD (Autism Spectrum Disorder detected)"}
    label = classes[prediction]
    
    st.subheader("Results:")
    if prediction == 1:
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")
        
    st.write(f"Confidence Level: **{confidence * 100:.2f}%**")
