# 🧠 Autism Spectrum Disorder Detection System

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NiBabel](https://img.shields.io/badge/NiBabel-Neuroimaging-4B8BBE?style=for-the-badge)](https://nipy.org/nibabel/)
[![Nilearn](https://img.shields.io/badge/Nilearn-Neuroimaging-3776AB?style=for-the-badge)](https://nilearn.github.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---

## 🧠 Project Overview

**Autism Spectrum Disorder Detection System** is a **Deep Learning and Machine Learning-based healthcare research project** that analyzes brain imaging data to classify samples into **ASD** and **CONTROL** categories.

The system uses an **Squeeze-and-Excitation CNN (SE-CNN)** to extract deep features and Machine Learning classifiers to perform the final classification.

The project also provides a **Streamlit web application** for making predictions from processed brain-scan images.

### Key Features

- 🧠 SE-CNN with attention mechanism
- 🔬 Brain imaging data preprocessing
- 📊 CNN-based feature extraction
- 🤖 Random Forest, Logistic Regression and KNN classifiers
- 📈 Model evaluation using Accuracy, Precision, Recall, F1-score and AUC
- 🌐 Streamlit prediction interface
- 💾 Saved trained models for inference

---

## 🏗️ System Architecture

```text
                 Brain Imaging Data
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
                    SE-CNN
               (Attention CNN)
                         │
                         ▼
              Feature Extraction
                 128-D Vector
                         │
                         ▼
              Machine Learning
                Classifiers
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
       Random Forest     LR          KNN
             │
             ▼
        ASD / CONTROL
             │
             ▼
      Streamlit Web App
🛠️ Tech Stack
Category	Technology
Programming	Python
Deep Learning	TensorFlow / Keras
CNN	Squeeze-and-Excitation CNN
Machine Learning	Scikit-learn
Classifiers	Random Forest, Logistic Regression, KNN
Image Processing	OpenCV, Pillow
Neuroimaging	NiBabel, Nilearn
Data Processing	NumPy, Pandas
Visualization	Matplotlib, Seaborn
Model Storage	Joblib
Web Application	Streamlit
Version Control	Git / GitHub
📊 Dataset

The project uses ABIDE (Autism Brain Imaging Data Exchange) neuroimaging data.

The dataset is organized into two classes:

Label	Meaning
ASD	Autism Spectrum Disorder
CONTROL	Control / Non-ASD

The project works with neuroimaging files such as:

.nii.gz

The preprocessing pipeline converts the selected neuroimaging data into processed numerical data that can be used for model development.

⚙️ Machine Learning Pipeline
1. Data Preparation

ABIDE metadata is loaded and ASD/CONTROL subjects are selected.

ABIDE Metadata
      ↓
Select ASD + CONTROL
      ↓
Create Dataset Metadata
2. Preprocessing

Images are converted to grayscale, resized to 128 × 128, and normalized.

Input Image
    ↓
Grayscale
    ↓
128 × 128
    ↓
Normalization
3. SE-CNN Training

The SE-CNN learns important image features using convolution and attention.

Input
  ↓
Conv2D
  ↓
Pooling
  ↓
Conv2D
  ↓
SE Attention
  ↓
Conv2D
  ↓
Global Average Pooling
  ↓
128-D Feature Vector
4. Feature Extraction

The trained SE-CNN is used as a feature extractor.

Image
  ↓
SE-CNN
  ↓
128-D Feature Vector
5. Classification

The extracted features are given to three Machine Learning classifiers:

Random Forest
Logistic Regression
KNN
6. Evaluation

Models are evaluated using:

Accuracy
Precision
Recall / Sensitivity
F1 Score
ROC-AUC
Confusion Matrix
🤖 Models
SE-CNN

The project uses a Squeeze-and-Excitation CNN to learn important features from the input images.

The trained model is saved as:

models/se_cnn.keras
Machine Learning Classifiers
Model	Purpose
Random Forest	Primary classification model
Logistic Regression	Comparative classifier
KNN	Comparative classifier

Saved models:

models/
├── se_cnn.keras
├── random_forest.pkl
├── logistic_regression.pkl
└── knn.pkl
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
train.py	Trains and saves the SE-CNN
ml_pipeline.py	Extracts CNN features and trains ML classifiers
app.py	Main Streamlit prediction application
app_fast.py	Faster application version
src/abide_loader.py	Loads and prepares ABIDE metadata
src/preprocessing.py	Image and neuroimaging preprocessing
src/cnn_model.py	Defines the SE-CNN architecture
src/se_block.py	Implements the SE attention block
src/feature_extraction.py	Extracts CNN feature vectors
src/classifiers.py	Trains ML classifiers
src/evaluate.py	Calculates model evaluation metrics
src/predict.py	Prediction utilities
requirements.txt	Project dependencies
🌐 Streamlit Application

The application provides a simple interface where the user can:

Upload a processed image.
Preprocess the image.
Extract CNN features.
Run the Random Forest classifier.
Display the predicted class.
Display prediction confidence.
Upload Image
     ↓
Preprocessing
     ↓
SE-CNN Feature Extraction
     ↓
Random Forest
     ↓
ASD / CONTROL
     ↓
Confidence
📷 Application Screenshots

Add screenshots to an assets/ folder:

assets/
├── home.png
├── prediction.png
└── result.png

Then include them here:

Home Page

Prediction Result

🚀 How to Run
1. Clone Repository
git clone https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection.git

cd Autism-Spectrum-Disorder-Detection
2. Create Virtual Environment
python -m venv .venv
3. Activate Environment

Windows:

.venv\Scripts\activate

Linux / macOS:

source .venv/bin/activate
4. Install Dependencies
pip install -r requirements.txt
5. Train SE-CNN
python train.py
6. Train ML Classifiers
python ml_pipeline.py
7. Start Streamlit
streamlit run app.py

Open:

http://localhost:8501
🔮 Future Improvements
3D CNN for complete brain volumes
Advanced attention mechanisms
Transfer learning
Grad-CAM / SHAP explainability
Hyperparameter optimization
Larger ABIDE dataset
FastAPI-based prediction API
Docker deployment
AWS cloud deployment
MLOps pipeline and model monitoring
⚠️ Disclaimer

This project is intended for educational and research purposes only.

The predictions generated by this system should not be considered a medical diagnosis and should not replace professional medical evaluation.

👨‍💻 Author

Mahalaxmi Kouchika

Computer Science Engineering | Machine Learning | Deep Learning | MLOps

⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
