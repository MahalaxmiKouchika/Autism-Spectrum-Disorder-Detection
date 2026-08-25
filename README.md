# 🧠 Autism Spectrum Disorder Detection System

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NiBabel](https://img.shields.io/badge/NiBabel-Neuroimaging-4B8BBE?style=for-the-badge)](https://nipy.org/nibabel/)
[![Nilearn](https://img.shields.io/badge/Nilearn-Neuroimaging-3776AB?style=for-the-badge)](https://nilearn.github.io/)

</p>

<p align="center">
<b>Deep Learning and Machine Learning based Neuroimaging Classification System</b>
</p>

---

# 🧠 Project Overview

**Autism Spectrum Disorder Detection System** is a Deep Learning and Machine Learning based healthcare research project designed to classify neuroimaging data into:

- **ASD — Autism Spectrum Disorder**
- **CONTROL — Non-ASD Control**

The system uses a **Squeeze-and-Excitation Convolutional Neural Network (SE-CNN)** to extract meaningful features from neuroimaging data.

The extracted features are then used by Machine Learning classifiers such as:

- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)

A **Streamlit web application** is provided for interactive prediction.

---

# ✨ Key Features

- 🧠 Squeeze-and-Excitation CNN
- 🔬 Neuroimaging data preprocessing
- 🖼️ Image preprocessing and normalization
- 🎯 Attention-based feature learning
- 📊 128-dimensional CNN feature extraction
- 🌲 Random Forest classification
- 📈 Logistic Regression classification
- 🔎 K-Nearest Neighbors classification
- 📊 Model performance evaluation
- 🌐 Interactive Streamlit web application
- 💾 Trained model saving and loading
- 🧪 ASD vs CONTROL classification

---

# 🛠️ Tools & Technologies

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| CNN Architecture | Squeeze-and-Excitation CNN (SE-CNN) |
| Machine Learning | Scikit-learn |
| ML Classifiers | Random Forest, Logistic Regression, KNN |
| Neuroimaging | NiBabel, Nilearn |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Dataset | ABIDE |
| Version Control | Git, GitHub |

---

# 📊 Dataset

The project uses neuroimaging data from the **Autism Brain Imaging Data Exchange (ABIDE)** dataset.

The dataset contains subjects belonging to two main classes:

| Label | Description |
|---|---|
| `ASD` | Autism Spectrum Disorder |
| `CONTROL` | Control / Non-ASD |

The project works with neuroimaging files such as:

text
.nii
.nii.gz

The raw neuroimaging files are processed before being used by the Deep Learning model.

🏗️ System Architecture
                    ABIDE Neuroimaging Data
                              │
                              ▼
                       Data Preparation
                              │
                              ▼
                  Neuroimaging Preprocessing
                              │
                              ▼
                       Image Processing
                              │
                              ▼
                            SE-CNN
                 Squeeze-and-Excitation CNN
                              │
                              ▼
                     Feature Extraction
                        128-D Vector
                              │
                              ▼
                  Machine Learning Models
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
          Random Forest       KNN      Logistic Regression
                │
                ▼
            ASD / CONTROL
                │
                ▼
           Streamlit Web App
                │
                ▼
         Prediction & Confidence
⚙️ Machine Learning Pipeline

The complete Machine Learning pipeline follows these stages:

ABIDE Dataset
      │
      ▼
Data Collection
      │
      ▼
Data Preparation
      │
      ▼
Neuroimaging Preprocessing
      │
      ▼
Image Processing
      │
      ▼
SE-CNN Training
      │
      ▼
Feature Extraction
      │
      ▼
128-D Feature Vector
      │
      ▼
Machine Learning Classification
      │
      ▼
ASD / CONTROL
📥 Data Collection

The project uses neuroimaging data obtained from the ABIDE dataset.

The data collection stage includes:

Dataset metadata
Subject information
Neuroimaging files
ASD / CONTROL labels
ABIDE Dataset
      │
      ▼
Metadata
      │
      ▼
Subject Information
      │
      ▼
ASD / CONTROL Labels
📋 Data Preparation

The collected dataset is organized according to the corresponding subject labels.

Dataset
   │
   ▼
Load Metadata
   │
   ▼
Select Subjects
   │
   ▼
Assign Labels
   │
   ▼
ASD / CONTROL

The prepared data is then passed to the preprocessing pipeline.

🔬 Neuroimaging Preprocessing

Neuroimaging files are loaded using NiBabel.

The preprocessing includes:

Loading .nii / .nii.gz files
Converting imaging data into numerical arrays
Handling invalid and NaN values
Normalizing image data
Preparing processed data for model training
.nii / .nii.gz
       │
       ▼
     NiBabel
       │
       ▼
 Numerical Array
       │
       ▼
 Data Cleaning
       │
       ▼
 Normalization
       │
       ▼
 Processed Data
🖼️ Image Preprocessing

The input images are prepared before being passed to the CNN.

Processing Steps
Grayscale conversion
Image resizing
Normalization
Conversion into CNN-compatible arrays
CNN Input Size
128 × 128 × 1
Image Processing Flow
Input Image
     │
     ▼
Grayscale Conversion
     │
     ▼
Resize to 128 × 128
     │
     ▼
Normalization
     │
     ▼
CNN Input
🧠 SE-CNN Architecture

The project uses a Squeeze-and-Excitation Convolutional Neural Network (SE-CNN).

The CNN extracts deep features from the processed input images.

Input Image
     │
     ▼
Conv2D - 32 Filters
     │
     ▼
Batch Normalization
     │
     ▼
ReLU Activation
     │
     ▼
Max Pooling
     │
     ▼
Conv2D - 64 Filters
     │
     ▼
Batch Normalization
     │
     ▼
ReLU Activation
     │
     ▼
Max Pooling
     │
     ▼
SE Attention Block
     │
     ▼
Conv2D - 128 Filters
     │
     ▼
Batch Normalization
     │
     ▼
ReLU Activation
     │
     ▼
Max Pooling
     │
     ▼
Global Average Pooling
     │
     ▼
128-D Feature Vector
🎯 Squeeze-and-Excitation Attention

The Squeeze-and-Excitation (SE) block provides channel-wise attention.

It helps the CNN learn which feature channels are important.

Feature Maps
     │
     ▼
Global Average Pooling
     │
     ▼
Channel Information
     │
     ▼
Fully Connected Layers
     │
     ▼
Channel Weights
     │
     ▼
Feature Recalibration
🔍 Feature Extraction

After the SE-CNN processes the input, the network generates a 128-dimensional feature vector.

Input Image
     │
     ▼
SE-CNN
     │
     ▼
Deep Feature Extraction
     │
     ▼
128-D Feature Vector

These features are then used by the Machine Learning classifiers.

🌲 Random Forest

Random Forest is used as a Machine Learning classification model.

The extracted 128-dimensional CNN features are provided as input to the Random Forest classifier.

128-D CNN Features
        │
        ▼
   Random Forest
        │
        ▼
   ASD / CONTROL
📈 Logistic Regression

Logistic Regression is used as another classification model.

It receives the extracted CNN feature vector and predicts the corresponding class.

128-D CNN Features
        │
        ▼
Logistic Regression
        │
        ▼
   ASD / CONTROL
🔎 K-Nearest Neighbors

K-Nearest Neighbors (KNN) is used as an additional comparative classifier.

It classifies the input based on the similarity between the extracted feature vector and existing samples.

128-D CNN Features
        │
        ▼
       KNN
        │
        ▼
   ASD / CONTROL
📊 Model Evaluation

The Machine Learning models are evaluated using standard classification metrics.

Metric	Purpose
Accuracy	Overall prediction correctness
Precision	Correctness of positive predictions
Recall	Ability to identify positive samples
F1-Score	Balance between Precision and Recall
ROC-AUC	Overall classification performance
Confusion Matrix	Correct and incorrect predictions
📁 Project Structure
Autism-Spectrum-Disorder-Detection/
│
├── data/
│   ├── dataset/
│   └── metadata/
│
├── src/
│   ├── abide_loader.py
│   ├── classifiers.py
│   ├── cnn_model.py
│   ├── download_abide.py
│   ├── evaluate.py
│   ├── feature_extraction.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   ├── se_block.py
│   ├── test_preprocessing.py
│   └── train_cnn.py
│
├── app.py
├── app_fast.py
├── ml_pipeline.py
├── train.py
│
├── fast_dummy_ui.html
│
├── run_3d_project.bat
├── run_demo.bat
├── run_web_preview.bat
├── run_web_preview_fast.bat
│
├── requirements.txt
├── .gitignore
└── README.md
📂 Important Files
File	Purpose
train.py	Main model training script
ml_pipeline.py	CNN feature extraction and ML classification pipeline
app.py	Main Streamlit application
app_fast.py	Faster Streamlit application
src/abide_loader.py	Loads ABIDE dataset information
src/download_abide.py	Handles ABIDE dataset downloading
src/prepare_dataset.py	Prepares the dataset
src/preprocessing.py	Performs image and neuroimaging preprocessing
src/cnn_model.py	Defines the SE-CNN architecture
src/se_block.py	Implements the SE attention block
src/train_cnn.py	Handles CNN training
src/feature_extraction.py	Extracts CNN feature vectors
src/classifiers.py	Implements ML classifiers
src/evaluate.py	Evaluates model performance
src/predict.py	Handles prediction
requirements.txt	Contains project dependencies
.gitignore	Specifies files ignored by Git
🌐 Streamlit Application

The project provides a Streamlit-based web application for interactive prediction.

Application Features
Simple user interface
Input image selection
Image preprocessing
SE-CNN feature extraction
Machine Learning classification
ASD / CONTROL prediction
Prediction confidence
Easy-to-use web interface
🔄 Application Workflow
Start
  │
  ▼
Open Streamlit App
  │
  ▼
Input Image
  │
  ▼
Image Preprocessing
  │
  ▼
SE-CNN
  │
  ▼
Feature Extraction
  │
  ▼
128-D Features
  │
  ▼
Machine Learning Classifier
  │
  ▼
Prediction
  │
  ├─────────────┐
  ▼             ▼
 ASD          CONTROL
  │             │
  └──────┬──────┘
         ▼
 Result & Confidence
📷 Application Screenshots

Create an assets folder in the repository:

assets/
├── home.png
├── prediction.png
└── result.png
🏠 Home Page

🔍 Prediction Page

📊 Prediction Result

🚀 How to Run
1. Clone the Repository
git clone https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection.git
cd Autism-Spectrum-Disorder-Detection
🐍 Create Virtual Environment
Windows
python -m venv .venv

Activate the environment:

.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv

Activate the environment:

source .venv/bin/activate
📦 Install Dependencies
pip install -r requirements.txt
🏋️ Train the CNN Model
python train.py

This trains the SE-CNN model used for deep feature extraction.

🤖 Run the Machine Learning Pipeline
python ml_pipeline.py

The pipeline performs:

CNN Feature Extraction
        ↓
128-D Feature Vector
        ↓
Machine Learning Classification
        ↓
Random Forest
Logistic Regression
KNN
🌐 Run the Streamlit Application
streamlit run app.py

The application will normally be available at:

http://localhost:8501
🔄 Complete Project Workflow
                 ABIDE Dataset
                       │
                       ▼
                Data Collection
                       │
                       ▼
                Data Preparation
                       │
                       ▼
          Neuroimaging Preprocessing
                       │
                       ▼
                Image Processing
                       │
                       ▼
                    SE-CNN
                       │
                       ▼
              Feature Extraction
                       │
                       ▼
                128-D Features
                       │
                       ▼
          Machine Learning Models
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
         Random Forest  KNN  Logistic Regression
              │
              ▼
          ASD / CONTROL
              │
              ▼
        Streamlit Application
              │
              ▼
        Prediction Result
📌 Expected Output

The system classifies the input into one of the following categories:

ASD

or

CONTROL

The Streamlit application displays the prediction result and confidence where available.

🔮 Future Improvements
🧠 3D CNN for complete brain-volume analysis
🎯 Advanced attention mechanisms
🔄 Transfer Learning
🔍 Grad-CAM visualization
🧩 SHAP-based explainability
⚙️ Hyperparameter optimization
📊 Larger and more diverse neuroimaging datasets
🚀 FastAPI-based prediction API
🐳 Docker containerization
☁️ AWS cloud deployment
🔄 CI/CD pipeline
📈 MLflow experiment tracking
🔍 Model monitoring
⚠️ Limitations
Model performance depends on the quality and size of the dataset.
Neuroimaging data requires considerable preprocessing and computational resources.
Model performance may vary when applied to datasets from different sources.
The current system has not been clinically validated.
The model should not be used as a standalone diagnostic system.
🩺 Disclaimer

This project is developed for educational and research purposes only.

The predictions generated by this system should not be considered medical advice or a clinical diagnosis.

This system is not intended to replace professional medical evaluation, diagnosis, or consultation with qualified healthcare professionals.

👨‍💻 Author

Mahalaxmi Kouchika

Computer Science Engineering Student

Areas of Interest
Machine Learning
Deep Learning
Artificial Intelligence
MLOps
Cloud Computing
DevOps
⭐ Support

If you found this project useful or interesting, please consider giving the repository a ⭐ on GitHub.

Your support helps improve and extend the project.
