# 🧠 Autism Spectrum Disorder Detection System

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NiBabel](https://img.shields.io/badge/NiBabel-Neuroimaging-4B8BBE?style=for-the-badge)](https://nipy.org/nibabel/)
[![Nilearn](https://img.shields.io/badge/Nilearn-Neuroimaging-3776AB?style=for-the-badge)](https://nilearn.github.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

</p>

**Deep Learning and Machine Learning-powered healthcare research application** for detecting Autism Spectrum Disorder (ASD) from neuroimaging data. The system uses a **Squeeze-and-Excitation Convolutional Neural Network (SE-CNN)** for deep feature extraction and Machine Learning classifiers for ASD classification.

---

## 🧠 Project Overview

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that can affect communication, behavior, and social interaction.

This project explores the use of **Artificial Intelligence, Deep Learning, and Neuroimaging** to classify brain imaging data into:

- **ASD**
- **CONTROL**

The system follows a hybrid approach where an **SE-CNN extracts meaningful image features**, and Machine Learning classifiers use those features for classification.

The project also provides a **Streamlit web application** for interactive prediction.

### Key Features

- 🧠 Squeeze-and-Excitation CNN
- 🔬 Neuroimaging data preprocessing
- 📊 Deep feature extraction
- 🤖 Random Forest classification
- 📈 Logistic Regression and KNN comparison
- 🌐 Streamlit web application
- 📋 Model evaluation and performance metrics
- 💾 Saved models for prediction

---

## 🏗️ System Architecture

```text
                  Neuroimaging Data
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
                  Image Processing
                         │
                         ▼
                     SE-CNN
              (Squeeze-and-Excitation)
                         │
                         ▼
                Feature Extraction
                  128-D Feature
                     Vector
                         │
                         ▼
              Machine Learning Models
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        Random Forest    KNN    Logistic Regression
              │
              ▼
        ASD / CONTROL
              │
              ▼
        Prediction Result
              │
              ▼
       Streamlit Web App
🛠️ Tech Stack
Category	Technology
Programming Language	Python
Deep Learning	TensorFlow / Keras
CNN Architecture	Squeeze-and-Excitation CNN
Machine Learning	Scikit-learn
Classifiers	Random Forest, Logistic Regression, KNN
Neuroimaging	NiBabel, Nilearn
Image Processing	OpenCV, Pillow
Data Processing	NumPy, Pandas
Visualization	Matplotlib, Seaborn
Model Persistence	Joblib
Web Application	Streamlit
Version Control	Git / GitHub
📊 Dataset

The project uses neuroimaging data from the Autism Brain Imaging Data Exchange (ABIDE).

The data is classified into two categories:

Label	Description
ASD	Autism Spectrum Disorder
CONTROL	Control / Non-ASD

The project processes neuroimaging files such as:

.nii
.nii.gz

The preprocessing pipeline reads the neuroimaging data and converts it into normalized numerical data suitable for model training.

⚙️ Machine Learning Pipeline
1. Data Preparation

The dataset metadata is loaded and subjects are organized according to their labels.

ABIDE Dataset
      ↓
Metadata
      ↓
ASD / CONTROL
      ↓
Selected Subjects
2. Neuroimaging Preprocessing

The raw NIfTI files are loaded using NiBabel.

The data is:

Loaded from .nii.gz
Converted into numerical arrays
NaN values handled
Normalized using mean and standard deviation
Saved as .npy files
.nii.gz
   ↓
NiBabel
   ↓
Numerical Data
   ↓
NaN Handling
   ↓
Normalization
   ↓
.npy
3. Image Preprocessing

For the CNN prototype, images are:

Converted to grayscale
Resized to 128 × 128
Normalized to the range [0, 1]
Input Image
     ↓
Grayscale
     ↓
128 × 128
     ↓
Normalization
     ↓
CNN Input
4. SE-CNN Feature Extraction

The project uses a Squeeze-and-Excitation CNN.

The architecture contains convolutional blocks followed by an SE attention block.

Input
  ↓
Conv2D - 32 Filters
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Max Pooling
  ↓
Conv2D - 64 Filters
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Max Pooling
  ↓
SE Attention Block
  ↓
Conv2D - 128 Filters
  ↓
Batch Normalization
  ↓
ReLU
  ↓
Max Pooling
  ↓
Global Average Pooling
  ↓
128-D Feature Vector
5. Machine Learning Classification

The extracted CNN features are used by multiple Machine Learning classifiers.

SE-CNN
   ↓
128-D Feature Vector
   ↓
┌─────────────────────────────┐
│ Machine Learning Classifiers│
├─────────────────────────────┤
│ Random Forest               │
│ Logistic Regression         │
│ KNN                         │
└─────────────────────────────┘
   ↓
ASD / CONTROL
🤖 Models Used
🧠 SE-CNN

The Squeeze-and-Excitation CNN is used to learn important patterns from the input image and generate a 128-dimensional feature vector.

The model uses:

Convolutional layers
Batch Normalization
ReLU activation
Max Pooling
SE Attention
Global Average Pooling
Dense Feature Layer
Dropout
Sigmoid output
🌲 Random Forest

Random Forest is used as one of the main classification algorithms for predicting:

ASD
CONTROL
📈 Logistic Regression

Logistic Regression is used as a comparative Machine Learning classifier.

🔍 K-Nearest Neighbors

KNN is also evaluated as an additional classification approach.

📈 Model Evaluation

The classifiers can be evaluated using standard classification metrics:

Metric	Purpose
Accuracy	Overall prediction correctness
Precision	Correctness of positive predictions
Recall	Ability to identify positive samples
F1-Score	Balance between Precision and Recall
ROC-AUC	Measures classification performance
Confusion Matrix	Shows correct and incorrect classifications
📁 Project Structure
Autism-Spectrum-Disorder-Detection/
│
├── data/
│   ├── raw/
│   ├── processed/
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
ml_pipeline.py	Feature extraction and ML classification pipeline
app.py	Main Streamlit application
app_fast.py	Faster Streamlit application
src/abide_loader.py	Loads ABIDE dataset information
src/download_abide.py	Handles dataset downloading
src/prepare_dataset.py	Prepares dataset for training
src/preprocessing.py	Handles image and NIfTI preprocessing
src/cnn_model.py	Defines the SE-CNN architecture
src/se_block.py	Implements Squeeze-and-Excitation attention
src/train_cnn.py	CNN training functionality
src/feature_extraction.py	Extracts CNN feature vectors
src/classifiers.py	Machine Learning classifiers
src/evaluate.py	Model evaluation
src/predict.py	Prediction functionality
requirements.txt	Python dependencies
.gitignore	Git ignored files
🌐 Application Workflow

The Streamlit application follows this workflow:

                  Start
                    │
                    ▼
            Open Web Application
                    │
                    ▼
              Input Image
                    │
                    ▼
            Image Preprocessing
                    │
                    ▼
             SE-CNN Model
                    │
                    ▼
          Feature Extraction
                    │
                    ▼
           Random Forest
                    │
                    ▼
             Prediction
                    │
              ┌─────┴─────┐
              ▼           ▼
             ASD       CONTROL
              │           │
              └─────┬─────┘
                    ▼
             Result Display
🖥️ Streamlit Application

The project includes a Streamlit-based interface for model inference.

The application allows the user to:

Provide an input image.
Preprocess the image.
Extract features using the SE-CNN.
Pass the features to the classifier.
Generate the predicted class.
Display the prediction result.
Prediction Classes
ASD
CONTROL
📷 Application Screenshots

Create an assets folder in the repository and place your screenshots inside it:

assets/
├── home.png
├── prediction.png
└── result.png
🏠 Home Page

🔍 Prediction Page

📊 Prediction Result

Replace the screenshot names above with your actual screenshot filenames if they are different.

🚀 How to Run
1. Clone the Repository
git clone https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection.git
cd Autism-Spectrum-Disorder-Detection
2. Create Virtual Environment
Windows
python -m venv .venv

Activate it:

.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv

Activate it:

source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

The project requirements include TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, NiBabel, Nilearn, OpenCV, Joblib, Requests, Streamlit and Pillow.

4. Train the CNN
python train.py

This trains the SE-CNN model.

5. Run the ML Pipeline
python ml_pipeline.py

This performs feature extraction and Machine Learning classification.

6. Run the Streamlit Application
streamlit run app.py

Then open:

http://localhost:8501
🔄 Complete Project Workflow
ABIDE Neuroimaging Dataset
            │
            ▼
       Data Selection
            │
            ▼
     Data Preprocessing
            │
            ▼
      Image Processing
            │
            ▼
         SE-CNN
            │
            ▼
   128-D Feature Vector
            │
            ▼
   Machine Learning Models
            │
      ┌─────┼─────┐
      ▼     ▼     ▼
     RF     LR    KNN
      │
      ▼
 ASD / CONTROL
      │
      ▼
 Streamlit App
🔮 Future Improvements
3D CNN for complete brain volumes
Advanced attention mechanisms
Transfer learning
Grad-CAM and SHAP explainability
Hyperparameter optimization
Larger and more diverse datasets
FastAPI prediction API
Docker containerization
AWS cloud deployment
CI/CD pipeline
MLflow model tracking
Model monitoring
⚠️ Limitations
Model performance depends on the quality and size of the dataset.
Neuroimaging data requires significant preprocessing and computational resources.
Predictions may not generalize to unseen populations or different datasets.
The system has not been clinically validated.
A machine learning prediction should not be treated as a medical diagnosis.
🩺 Disclaimer

This project is intended for educational and research purposes only.

The predictions generated by this system should not be considered a medical diagnosis or medical advice.

The system should not replace professional medical evaluation, clinical assessment, or consultation with qualified healthcare professionals.

👨‍💻 Author

Mahalaxmi Kouchika

Computer Science Engineering Student

Interests:
Machine Learning • Deep Learning • Artificial Intelligence • MLOps • Cloud Computing • DevOps

⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
