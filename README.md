# 🧠 Autism Spectrum Disorder Detection System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NiBabel](https://img.shields.io/badge/NiBabel-Neuroimaging-4B8BBE?style=for-the-badge)
![Nilearn](https://img.shields.io/badge/Nilearn-Neuroimaging-3776AB?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

</p>

<p align="center">
<b>Deep Learning + Machine Learning based Neuroimaging Classification System</b>
</p>

---

# 🧠 Project Overview

**Autism Spectrum Disorder Detection System** is a **Deep Learning and Machine Learning based healthcare research project** designed to classify neuroimaging data into:

- **ASD — Autism Spectrum Disorder**
- **CONTROL — Non-ASD Control**

The system uses a **Squeeze-and-Excitation Convolutional Neural Network (SE-CNN)** to learn meaningful features from neuroimaging data.

The extracted features are then given to Machine Learning classifiers including:

- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)

A **Streamlit web application** is also provided for interactive prediction.

---

# ✨ Key Features

- 🧠 Squeeze-and-Excitation CNN for deep feature extraction
- 🔬 Neuroimaging data preprocessing
- 📊 Image preprocessing and normalization
- 🧩 Attention-based feature learning
- 📌 128-dimensional CNN feature extraction
- 🌲 Random Forest classification
- 📈 Logistic Regression classification
- 🔎 KNN classification
- 📊 Model performance evaluation
- 🌐 Interactive Streamlit web application
- 💾 Trained model saving and loading
- 🧪 ASD vs CONTROL classification
- 📋 Prediction result and confidence display

---

# 🛠️ Tools & Technologies

## 💻 Technology Stack

| Category | Technologies |
|---|---|
| **Programming Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **CNN Architecture** | Squeeze-and-Excitation CNN (SE-CNN) |
| **Machine Learning** | Scikit-learn |
| **ML Classifiers** | Random Forest, Logistic Regression, KNN |
| **Neuroimaging** | NiBabel, Nilearn |
| **Image Processing** | OpenCV, Pillow |
| **Data Processing** | NumPy, Pandas |
| **Data Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Web Application** | Streamlit |
| **Dataset** | ABIDE |
| **Version Control** | Git, GitHub |

### Major Technologies

**Python**  
Used as the primary programming language for data processing, Deep Learning, Machine Learning, and application development.

**TensorFlow / Keras**  
Used to build and train the SE-CNN model.

**Scikit-learn**  
Used for Machine Learning classifiers and model evaluation.

**NiBabel / Nilearn**  
Used for loading and processing neuroimaging data.

**OpenCV / Pillow**  
Used for image processing and preparation.

**NumPy / Pandas**  
Used for numerical operations and dataset management.

**Matplotlib / Seaborn**  
Used for visualization and model evaluation.

**Joblib**  
Used for saving and loading Machine Learning models.

**Streamlit**  
Used to create the interactive web application.

**Git / GitHub**  
Used for source-code management and project version control.

---

# 📊 Dataset

The project uses neuroimaging data from the:

**Autism Brain Imaging Data Exchange (ABIDE)**

The dataset contains samples belonging to two major classes:

| Label | Description |
|---|---|
| `ASD` | Autism Spectrum Disorder |
| `CONTROL` | Control / Non-ASD |

The project works with neuroimaging files such as:

```text
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
1. Data Collection

The project uses the ABIDE neuroimaging dataset.

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
2. Data Preparation

The dataset metadata and neuroimaging files are organized according to their corresponding labels.

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
3. Neuroimaging Preprocessing

Neuroimaging files are loaded using NiBabel.

The preprocessing pipeline performs:

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

The preprocessing includes:

Grayscale conversion
Image resizing
Normalization
Conversion into CNN-compatible arrays

The CNN input size is:

128 × 128 × 1
Image Processing Pipeline
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

The SE block introduces an attention mechanism that allows the network to learn the importance of different feature channels.

CNN Architecture
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

The SE block helps the CNN focus on important feature channels.

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

This allows the network to assign greater importance to useful features while reducing the influence of less important features.

🔍 Feature Extraction

After the SE-CNN is trained, it is used as a feature extractor.

Each processed image is converted into a:

128-Dimensional Feature Vector
Feature Extraction Workflow
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

These features are then passed to the Machine Learning classifiers.

🤖 Machine Learning Classification

The extracted CNN features are used as input to multiple Machine Learning algorithms.

             SE-CNN
                │
                ▼
       128-D Feature Vector
                │
                ▼
      Machine Learning Models
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
Random Forest   KNN   Logistic Regression
      │
      ▼
 ASD / CONTROL
🌲 Random Forest

Random Forest is used as the primary classification model.

It combines multiple decision trees to make the final classification.

128-D CNN Features
        │
        ▼
 Random Forest
        │
        ▼
 ASD / CONTROL
📈 Logistic Regression

Logistic Regression is used as a Machine Learning classification model and for comparison with other classifiers.

128-D CNN Features
        │
        ▼
Logistic Regression
        │
        ▼
 ASD / CONTROL
🔎 K-Nearest Neighbors

KNN is used as an additional comparative classifier.

128-D CNN Features
        │
        ▼
       KNN
        │
        ▼
 ASD / CONTROL
📊 Model Evaluation

The classification models are evaluated using standard Machine Learning metrics.

Metric	Purpose
Accuracy	Measures overall prediction correctness
Precision	Measures correctness of positive predictions
Recall	Measures the ability to identify positive samples
F1-Score	Provides a balance between Precision and Recall
ROC-AUC	Measures overall classification performance
Confusion Matrix	Shows correct and incorrect predictions
Evaluation Workflow
Predictions
     │
     ▼
Actual Labels
     │
     ▼
Evaluation Metrics
     │
 ┌───┼────┬────────┐
 ▼   ▼    ▼        ▼
Acc Precision Recall F1
     │
     ▼
ROC-AUC
     │
     ▼
Confusion Matrix
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
ml_pipeline.py	CNN feature extraction and Machine Learning classification pipeline
app.py	Main Streamlit application
app_fast.py	Faster Streamlit application
src/abide_loader.py	Loads ABIDE dataset information
src/download_abide.py	Handles ABIDE dataset downloading
src/prepare_dataset.py	Prepares the dataset
src/preprocessing.py	Performs image and neuroimaging preprocessing
src/cnn_model.py	Defines the SE-CNN architecture
src/se_block.py	Implements the Squeeze-and-Excitation attention block
src/train_cnn.py	Handles CNN training
src/feature_extraction.py	Extracts CNN feature vectors
src/classifiers.py	Implements Machine Learning classifiers
src/evaluate.py	Evaluates model performance
src/predict.py	Handles prediction
requirements.txt	Contains project dependencies
.gitignore	Specifies files ignored by Git
🌐 Streamlit Web Application

The project provides a Streamlit-based web application for interactive model prediction.

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
          Random Forest
                  │
                  ▼
             Prediction
                  │
             ┌────┴────┐
             ▼         ▼
            ASD      CONTROL
             │         │
             └────┬────┘
                  ▼
         Result & Confidence
📷 Application Screenshots

Create an assets folder in the project:

assets/
├── home.png
├── prediction.png
└── result.png
🏠 Home Page

🔍 Prediction Page

📊 Prediction Result

Replace the screenshot filenames with your actual image filenames if they are different.

🚀 How to Run
1. Clone the Repository
git clone https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection.git
cd Autism-Spectrum-Disorder-Detection
2. Create Virtual Environment
Windows
python -m venv .venv

Activate the environment:

.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv

Activate the environment:

source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the SE-CNN Model
python train.py

This trains the SE-CNN model required for feature extraction.

5. Run the Machine Learning Pipeline
python ml_pipeline.py

This performs:

CNN Feature Extraction
        ↓
Feature Preparation
        ↓
Machine Learning Training
        ↓
Random Forest
Logistic Regression
KNN
6. Start the Streamlit Application
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
         Random Forest KNN   Logistic Regression
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

The Streamlit application displays the prediction result along with the corresponding model confidence where available.

🔮 Future Improvements

The following improvements can be added to make the system more robust and production-ready:

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
📊 Real-time model performance monitoring
⚠️ Limitations
The performance of the model depends on the quality and size of the dataset.
Neuroimaging data requires considerable preprocessing and computational resources.
Model performance may vary when applied to datasets from different sources.
The current system has not been clinically validated.
The model should not be used as a standalone diagnostic system.
Prediction results may not generalize to every individual or population.
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
