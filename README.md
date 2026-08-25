# 🧠 Autism Spectrum Disorder Detection System

<p align="center">
  <strong>AI-Powered Neuroimaging Classification for Autism Spectrum Disorder Detection</strong>
</p>

<p align="center">
  A deep learning and machine learning system that analyzes neuroimaging images and classifies them as <strong>ASD</strong> or <strong>CONTROL</strong> using SE-CNN and machine learning classifiers.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![NiBabel](https://img.shields.io/badge/Neuroimaging-NiBabel-blue)
![Nilearn](https://img.shields.io/badge/Neuroimaging-Nilearn-green)

</p>

<p align="center">

![ASD Detection](https://img.shields.io/badge/ASD%20Detection-AI%20Powered-purple)
![SE-CNN](https://img.shields.io/badge/Model-SE--CNN-blueviolet)
![Classification](https://img.shields.io/badge/Task-Binary%20Classification-success)
![Status](https://img.shields.io/badge/Project-Research%20%26%20Education-yellow)

</p>

---

## 🖥️ Application Preview

<p align="center">
  <img src="assets/asd-detection.png" alt="Autism Spectrum Disorder Detection Application" width="900">
</p>

---

## 🩺 Project Overview

**Autism Spectrum Disorder Detection System** is an AI-based project developed to classify neuroimaging images into two categories:

* 🔴 **ASD — Autism Spectrum Disorder**
* 🟢 **CONTROL — Non-ASD Control**

The system combines **deep learning** and **machine learning** to extract meaningful image features and perform classification.

The project uses a **Squeeze-and-Excitation Convolutional Neural Network (SE-CNN)** for feature extraction and applies machine learning classifiers to the extracted features.

A **Streamlit web application** provides an easy interface for making predictions.

---

## 🎯 Project Objective

The main objective is to develop an AI-based system capable of:

* Processing neuroimaging data
* Preprocessing input images
* Extracting important features using **SE-CNN**
* Generating a **128-dimensional feature vector**
* Classifying images into **ASD or CONTROL**
* Providing prediction confidence through a web application

---

## ✨ Key Features

* 🧠 **SE-CNN based feature extraction**
* 🔬 **Neuroimaging data processing**
* 🖼️ **Image preprocessing and normalization**
* 🎯 **Attention-based feature learning**
* 📊 **128-dimensional feature extraction**
* 🌲 **Random Forest classification**
* 📈 **Logistic Regression classification**
* 🔎 **K-Nearest Neighbors classification**
* 🌐 **Streamlit web application**
* 📌 **ASD / CONTROL prediction**
* 📊 **Prediction confidence**
* 📈 **Classification model evaluation**

---

## 🏗️ System Architecture

```text
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
                           │
                           ▼
                  Feature Extraction
                           │
                           ▼
                  128-D Feature Vector
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Random Forest      KNN    Logistic Regression
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                     ASD / CONTROL
                           │
                           ▼
                  Streamlit Web App
                           │
                           ▼
                Prediction & Confidence
```

---

## 🛠️ Technology Stack

| Technology               | Purpose                      |
| ------------------------ | ---------------------------- |
| **Python**               | Programming language         |
| **TensorFlow / Keras**   | Deep learning                |
| **SE-CNN**               | Feature extraction           |
| **Scikit-learn**         | Machine learning             |
| **Random Forest**        | Classification               |
| **Logistic Regression**  | Classification               |
| **KNN**                  | Classification               |
| **NiBabel**              | Neuroimaging data processing |
| **Nilearn**              | Neuroimaging analysis        |
| **OpenCV**               | Image processing             |
| **Pillow**               | Image handling               |
| **NumPy**                | Numerical processing         |
| **Pandas**               | Data processing              |
| **Matplotlib / Seaborn** | Visualization                |
| **Joblib**               | Model saving                 |
| **Streamlit**            | Web application              |

---

## 📊 Dataset

The project is designed to work with the **Autism Brain Imaging Data Exchange (ABIDE)** neuroimaging dataset.

### Classification Classes

| Class       | Description              |
| ----------- | ------------------------ |
| **ASD**     | Autism Spectrum Disorder |
| **CONTROL** | Non-ASD Control          |

The project supports neuroimaging files such as:

```text
.nii
.nii.gz
```

---

## ⚙️ Data Preprocessing

The input data goes through preprocessing before being passed to the model.

### Processing Steps

```text
Neuroimaging File
       ↓
Load Data
       ↓
Convert to Numerical Array
       ↓
Handle Invalid / NaN Values
       ↓
Normalize Data
       ↓
Prepare Image
       ↓
CNN Input
```

For image-based CNN processing, the input is prepared as:

```text
128 × 128 × 1
```

The image is converted to **grayscale**, resized, and normalized.

---

## 🧠 SE-CNN Model

The project uses a **Squeeze-and-Excitation Convolutional Neural Network**.

### Architecture

```text
Input Image
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
```

### Squeeze-and-Excitation Block

The SE block provides **channel-wise attention**.

```text
Feature Maps
     ↓
Global Average Pooling
     ↓
Channel Information
     ↓
Channel Weights
     ↓
Feature Recalibration
```

This allows the network to focus on more important feature channels.

---

## 🔍 Feature Extraction

The SE-CNN extracts deep features from the input image.

```text
Input Image
     ↓
SE-CNN
     ↓
Deep Feature Extraction
     ↓
128-Dimensional Feature Vector
```

The resulting **128-dimensional feature vector** is used by the machine learning classifiers.

---

## 🤖 Machine Learning Models

### 🌲 Random Forest

Used to classify the extracted CNN features into:

```text
ASD
```

or

```text
CONTROL
```

### 📈 Logistic Regression

Used as a binary classification model for ASD prediction.

### 🔎 K-Nearest Neighbors

Uses feature similarity to classify the input sample.

### Classification Flow

```text
128-D CNN Features
        │
        ├── Random Forest
        │
        ├── Logistic Regression
        │
        └── KNN
              │
              ▼
        ASD / CONTROL
```

---

## 📈 Model Evaluation

The classification models can be evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**
* **Confusion Matrix**

These metrics help evaluate how well the models distinguish between ASD and CONTROL samples.

---

## 🌐 Web Application

The project includes a **Streamlit-based web application**.

### Application Workflow

```text
Upload Image
      ↓
Image Preprocessing
      ↓
SE-CNN Feature Extraction
      ↓
128-D Features
      ↓
Machine Learning Classifier
      ↓
ASD / CONTROL
      ↓
Confidence Score
```

### Application Output

The application displays:

* **Predicted Class**
* **Confidence Score**

Possible results:

```text
ASD
```

or

```text
CONTROL
```

---

## 📁 Project Structure

```text
Autism-Spectrum-Disorder-Detection/
│
├── data/
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
├── assets/
│   └── asd-detection.png
│
├── app.py
├── app_fast.py
├── ml_pipeline.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection.git
cd Autism-Spectrum-Disorder-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Model

```bash
python train.py
```

### 6. Run the ML Pipeline

```bash
python ml_pipeline.py
```

### 7. Start the Web Application

```bash
streamlit run app.py
```

---

## 🔄 Complete Workflow

```text
Dataset
   ↓
Data Collection
   ↓
Data Preprocessing
   ↓
Image Processing
   ↓
SE-CNN
   ↓
Feature Extraction
   ↓
128-D Features
   ↓
ML Classification
   ↓
ASD / CONTROL
   ↓
Streamlit Application
   ↓
Prediction & Confidence
```

---

## 🔮 Future Improvements

* **3D CNN** for complete brain-volume analysis
* **Advanced attention mechanisms**
* **Transfer learning**
* **Grad-CAM visualization**
* **SHAP explainability**
* **Hyperparameter optimization**
* **Larger neuroimaging datasets**
* **Improved model validation**
* **Cloud deployment**

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The prediction generated by this system **should not be considered a medical diagnosis**.

It should not replace professional medical evaluation or consultation with a qualified healthcare professional.

---

## 👨‍💻 Author

### **Mahalaxmi Kouchika**

**Autism Spectrum Disorder Detection System**

---

## ⭐ Support

If you find this project useful, please support it by:

* ⭐ **Starring the repository**
* 🍴 **Forking the project**
* 🐛 **Reporting issues**
* 💡 **Sharing feedback**

### 🔗 Repository

**GitHub:**
https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection

---

<p align="center">
  <strong>🧠 Autism Spectrum Disorder Detection using AI</strong>
</p>

<p align="center">
  Built with Python • TensorFlow • Scikit-learn • SE-CNN • Streamlit
</p>
