# Autism-Spectrum-Disorder-Detection
# 🧠 Autism Spectrum Disorder Detection

### AI-Powered Autism Spectrum Disorder Detection using CNN-Based Feature Extraction and Machine Learning

**Autism Spectrum Disorder Detection** is a machine learning and deep learning-based healthcare research project designed to classify brain imaging data into **ASD (Autism Spectrum Disorder)** and **CONTROL** categories.

The system combines **medical image preprocessing, Convolutional Neural Network (CNN) feature extraction, and Random Forest classification** to build an end-to-end prediction pipeline. A **Streamlit-based web application** provides an interactive interface for running predictions.

> ⚠️ **Disclaimer:** This project is developed for educational and research purposes. It is not a clinically validated medical diagnostic system and should not be used to make medical decisions.

---

## 📌 Project Overview

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that can be studied using behavioral, clinical, and neuroimaging data.

This project explores the application of **Artificial Intelligence, Deep Learning, and Machine Learning** to neuroimaging-based ASD classification.

The system processes brain scan images, extracts meaningful representations using a CNN model, and uses a machine learning classifier to predict whether the input belongs to the **ASD** or **CONTROL** class.

### The complete pipeline:

```text
Brain Scan / Neuroimaging Data
            │
            ▼
     Data Preprocessing
            │
            ▼
    Image Normalization
            │
            ▼
       CNN Model
            │
            ▼
    Feature Extraction
            │
            ▼
   Feature Representation
            │
            ▼
 Random Forest Classifier
            │
            ▼
   ASD / CONTROL Prediction
            │
            ▼
   Prediction Probability
            │
            ▼
     Streamlit Dashboard
```

---

# ✨ Key Features

* 🧠 Brain imaging-based ASD classification
* 🔬 CNN-based deep feature extraction
* 🌲 Random Forest machine learning classifier
* 🖼️ Image preprocessing and normalization
* 📊 Prediction probability/confidence
* 🌐 Interactive Streamlit web application
* ⚙️ Modular machine learning pipeline
* 💾 Saved trained models for inference
* 🧪 Separate model training and prediction workflow
* 📁 Organized source-code structure
* 🚀 Local web application deployment

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │   Brain Scan Input  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Image Preprocessing │
                         │                     │
                         │ • Resize            │
                         │ • Grayscale         │
                         │ • Normalization     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     CNN Model       │
                         │                     │
                         │ Deep Feature        │
                         │ Extraction          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Extracted Features  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Random Forest       │
                         │ Classifier          │
                         └──────────┬──────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
               ┌─────────────┐             ┌─────────────┐
               │     ASD     │             │   CONTROL   │
               └─────────────┘             └─────────────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Prediction Result   │
                         │ + Probability      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Streamlit Dashboard │
                         └─────────────────────┘
```

---

# 🧩 How the System Works

The project consists of several major stages.

### 1. Input Data

The system accepts brain imaging data associated with ASD and control subjects.

The project uses neuroimaging data and processes the images before providing them to the deep learning model.

---

### 2. Data Preprocessing

The input image is prepared for model processing.

Typical preprocessing operations include:

* Image loading
* Image resizing
* Grayscale conversion
* Pixel normalization
* Shape transformation
* Input tensor preparation

The objective is to convert raw image data into a consistent format suitable for the CNN.

---

### 3. CNN Feature Extraction

A CNN is used to learn meaningful patterns from the input images.

Instead of directly using raw pixels for classification, the CNN transforms the image into a compact feature representation.

```text
Input Image
     ↓
Convolution Layers
     ↓
Feature Maps
     ↓
Pooling
     ↓
Deep Features
```

These extracted features are then passed to the machine learning classifier.

---

### 4. Machine Learning Classification

The extracted CNN features are provided to a **Random Forest classifier**.

The Random Forest analyzes the feature representation and predicts one of the two classes:

```text
ASD
```

or

```text
CONTROL
```

---

### 5. Prediction

The application displays the predicted class and corresponding prediction probability/confidence.

---

# 🛠️ Tech Stack

| Category              | Technology                   |
| --------------------- | ---------------------------- |
| Programming Language  | Python                       |
| Deep Learning         | TensorFlow / Keras           |
| CNN                   | Convolutional Neural Network |
| Machine Learning      | Scikit-learn                 |
| Classifier            | Random Forest                |
| Numerical Computing   | NumPy                        |
| Data Processing       | Pandas                       |
| Medical Imaging       | NiBabel                      |
| Neuroimaging Analysis | Nilearn                      |
| Image Processing      | OpenCV                       |
| Image Handling        | Pillow                       |
| Visualization         | Matplotlib                   |
| Visualization         | Seaborn                      |
| Model Persistence     | Joblib                       |
| Web Application       | Streamlit                    |
| Environment           | Python Virtual Environment   |
| Version Control       | Git / GitHub                 |

---

# 📊 Dataset

The project works with **neuroimaging data for ASD and control subjects**.

The dataset contains brain imaging samples associated with two primary classes:

| Class     | Description              |
| --------- | ------------------------ |
| `ASD`     | Autism Spectrum Disorder |
| `CONTROL` | Control / non-ASD class  |

The neuroimaging files can be processed using libraries such as:

* **NiBabel** — reading NIfTI medical imaging files
* **Nilearn** — neuroimaging data processing and analysis
* **NumPy** — numerical array operations
* **OpenCV / Pillow** — image processing

### Dataset Format

Neuroimaging data may be provided in formats such as:

```text
.nii
.nii.gz
```

The preprocessing pipeline converts the required imaging information into a format that can be consumed by the CNN.

---

# 🧠 Prediction Classes

The classification system uses two primary categories:

| Class     | Meaning                        |
| --------- | ------------------------------ |
| `ASD`     | Autism Spectrum Disorder class |
| `CONTROL` | Control / non-ASD class        |

The final classifier predicts which class the processed input most closely resembles.

---

# ⚙️ Machine Learning Pipeline

## 1. Data Collection

Neuroimaging data is collected and organized according to the corresponding class labels.

```text
Dataset
   │
   ├── ASD
   │
   └── CONTROL
```

---

## 2. Data Preprocessing

The raw imaging data is processed before training.

Operations may include:

```text
Load Image
    ↓
Validate Image
    ↓
Resize
    ↓
Normalize
    ↓
Prepare Tensor
```

---

## 3. CNN Training

The CNN learns spatial and visual representations from the processed imaging data.

```text
Input
 ↓
Convolution
 ↓
Activation
 ↓
Pooling
 ↓
Convolution
 ↓
Pooling
 ↓
Flatten / Feature Representation
```

The learned representation is used as the input to the machine learning classifier.

---

## 4. Feature Extraction

After training, the CNN is used as a feature extractor.

```text
Brain Image
     ↓
Trained CNN
     ↓
Feature Vector
```

The resulting feature vectors are stored or directly passed to the classifier.

---

## 5. Random Forest Classification

The extracted features are passed to a Random Forest model.

```text
CNN Features
      ↓
Random Forest
      ↓
Prediction
      ↓
ASD / CONTROL
```

Random Forest combines multiple decision trees to make the final classification.

---

## 6. Model Prediction

The trained models are loaded during application execution.

```text
Input
  ↓
Preprocessing
  ↓
CNN Feature Extraction
  ↓
Random Forest
  ↓
Prediction
  ↓
Probability / Confidence
```

---

# 🤖 Machine Learning Models

The main prediction pipeline uses:

### CNN

Used for:

* Learning image representations
* Extracting deep features
* Capturing spatial patterns

### Random Forest

Used for:

* Classification
* Learning from CNN-generated features
* Predicting ASD / CONTROL

---

# 💾 Trained Model Artifacts

The trained models are stored in the `models/` directory.

Example:

```text
models/
│
├── se_cnn.keras
└── random_forest.pkl
```

### `se_cnn.keras`

The trained CNN model used for deep feature extraction.

### `random_forest.pkl`

The trained Random Forest classifier used for final prediction.

---

# 📁 Repository Structure

```text
Autism-Spectrum-Disorder-Detection/
│
├── data/
│   └── Dataset and neuroimaging data
│
├── models/
│   ├── se_cnn.keras
│   └── random_forest.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── Supporting ML modules
│
├── app.py
│
├── app_fast.py
│
├── ml_pipeline.py
│
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
│
├── README.md
│
└── .gitignore
```

---

# 📂 File and Folder Description

| File / Folder              | Purpose                                        |
| -------------------------- | ---------------------------------------------- |
| `data/`                    | Contains dataset and input data                |
| `models/`                  | Stores trained ML/DL models                    |
| `src/`                     | Contains reusable preprocessing and ML modules |
| `app.py`                   | Main Streamlit application                     |
| `app_fast.py`              | Faster/optimized application interface         |
| `ml_pipeline.py`           | Machine learning processing pipeline           |
| `train.py`                 | Model training script                          |
| `fast_dummy_ui.html`       | Lightweight UI/testing interface               |
| `run_3d_project.bat`       | Windows script for running the 3D project      |
| `run_demo.bat`             | Windows demo launcher                          |
| `run_web_preview.bat`      | Windows web preview launcher                   |
| `run_web_preview_fast.bat` | Fast web preview launcher                      |
| `requirements.txt`         | Python dependencies                            |
| `.gitignore`               | Files excluded from Git                        |
| `README.md`                | Project documentation                          |

---

# 🔄 Complete Application Workflow

```text
                 START
                   │
                   ▼
          Upload / Select Image
                   │
                   ▼
          Validate Input Image
                   │
                   ▼
          Preprocess Image
                   │
                   ▼
          CNN Feature Extraction
                   │
                   ▼
          Generate Feature Vector
                   │
                   ▼
        Random Forest Classifier
                   │
                   ▼
          Generate Prediction
                   │
                   ▼
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
      ASD                   CONTROL
       │                       │
       └───────────┬───────────┘
                   ▼
          Display Prediction
                   │
                   ▼
          Display Probability
                   │
                   ▼
                  END
```

---

# 🌐 Streamlit Application

The project includes an interactive web application developed using **Streamlit**.

The application provides a simple interface for interacting with the trained machine learning pipeline without requiring users to execute individual Python scripts.

### Application Flow

```text
Open Streamlit Application
          ↓
Select / Upload Input
          ↓
Preprocess Data
          ↓
Run CNN
          ↓
Extract Features
          ↓
Run Random Forest
          ↓
Display Prediction
```

---

# 🚀 Installation and Setup

## 1. Clone the Repository

```bash
git clone https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection.git
```

Navigate into the project:

```bash
cd Autism-Spectrum-Disorder-Detection
```

---

# 🐍 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

# 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ 4. Train the Model

If training from scratch is required:

```bash
python train.py
```

The training process prepares the deep learning and machine learning components required for prediction.

---

# ⚙️ 5. Run the ML Pipeline

```bash
python ml_pipeline.py
```

This can be used to execute the project's machine learning processing workflow.

---

# 🌐 6. Run the Streamlit Application

Start the main application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🖥️ Running the Fast Application

The repository also contains an optimized application:

```bash
streamlit run app_fast.py
```

This version can be used when a faster or lighter application workflow is preferred.

---

# 📋 Requirements

The main dependencies include:

```text
Python
TensorFlow
Keras
NumPy
Pandas
Scikit-learn
OpenCV
Pillow
NiBabel
Nilearn
Matplotlib
Seaborn
Joblib
Streamlit
```

For the exact versions, refer to:

```text
requirements.txt
```

---

# 📈 Model Evaluation

A complete evaluation of the ASD classification model should consider multiple metrics.

| Metric           | Purpose                                     |
| ---------------- | ------------------------------------------- |
| Accuracy         | Overall classification correctness          |
| Precision        | Correctness of positive predictions         |
| Recall           | Ability to identify positive samples        |
| F1-Score         | Balance between precision and recall        |
| ROC-AUC          | Measures class discrimination               |
| Confusion Matrix | Shows correct and incorrect classifications |

Example evaluation workflow:

```text
Test Dataset
     ↓
Trained CNN
     ↓
Feature Extraction
     ↓
Random Forest
     ↓
Predictions
     ↓
Evaluation Metrics
```

> Actual performance values should be added to this README after final evaluation on the project's test dataset.

---

# 🔬 Experiments

The project can be used to experiment with different approaches, including:

### Deep Learning

* CNN architectures
* Different convolutional layers
* Feature extraction strategies
* Image preprocessing techniques

### Machine Learning

* Random Forest
* Logistic Regression
* Support Vector Machine
* Other classical classifiers

### Preprocessing

* Image resizing
* Normalization
* Different feature representations
* Data augmentation

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Develop an AI-based approach for ASD classification.
2. Process neuroimaging data using Python-based tools.
3. Extract meaningful image features using CNNs.
4. Use machine learning for final classification.
5. Build an interactive prediction application.
6. Create a modular and reproducible ML pipeline.
7. Explore the application of AI in healthcare research.
8. Provide a foundation for future explainable and deployable AI systems.

---

# 💡 Why This Project?

Traditional healthcare analysis can require significant expert involvement and time.

Machine learning can help researchers identify complex patterns within large datasets.

This project demonstrates how different AI components can be combined:

```text
Medical Imaging
      +
Deep Learning
      +
Machine Learning
      +
Web Application
      =
AI-Based ASD Research Platform
```

---

# 🔮 Future Enhancements

The project can be extended in several directions.

### 🧠 Deep Learning

* Attention-based CNN architecture
* 3D CNN for volumetric brain scans
* Transfer learning
* Vision Transformers
* Ensemble deep learning

### 🔍 Explainable AI

* Grad-CAM
* SHAP
* LIME
* Feature importance visualization
* Attention visualization

### 📊 Model Improvements

* Hyperparameter optimization
* Cross-validation
* Class imbalance handling
* Data augmentation
* Ensemble classifiers

### 🚀 Deployment

* Docker containerization
* REST API using FastAPI
* Cloud deployment
* AWS integration
* CI/CD pipeline
* Model versioning
* MLflow experiment tracking
* MLOps monitoring

### 📱 Application Improvements

* User authentication
* Prediction history
* Visualization dashboard
* Patient/report management
* PDF report generation
* 3D brain visualization

---

# 🔐 Responsible AI and Privacy

Because this project involves healthcare-related data, privacy and responsible AI practices are important.

Future implementations should consider:

* Secure storage of medical data
* Removal of personally identifiable information
* Dataset bias analysis
* Model fairness evaluation
* Secure API communication
* Access control
* Model monitoring
* Explainability of predictions

---

# ⚠️ Limitations

This project has several limitations:

* Model performance depends on the quality and size of the dataset.
* Neuroimaging data can be computationally expensive to process.
* The model may not generalize to different datasets or clinical environments.
* Predictions can contain false positives and false negatives.
* The system has not been clinically validated.
* Prediction confidence should not be interpreted as medical certainty.

---

# 🩺 Medical Disclaimer

> **This project is intended strictly for educational, academic, and research purposes.**

The predictions generated by this system **must not be considered a medical diagnosis or medical advice**.

The system should not replace:

* Medical professionals
* Clinical diagnosis
* Professional neurological assessment
* Standard medical testing

Any future clinical use would require extensive validation, clinical trials, regulatory approval, and evaluation by qualified healthcare professionals.

---

# 🧪 Example Prediction Flow

```text
Input
  │
  ▼
Brain Scan
  │
  ▼
Preprocessing
  │
  ▼
CNN
  │
  ▼
Deep Feature Vector
  │
  ▼
Random Forest
  │
  ▼
┌───────────────────────┐
│ Prediction: ASD       │
│ Probability: XX.XX%   │
└───────────────────────┘
```

---

# 📚 Technologies and Their Roles

| Technology        | Role in Project                     |
| ----------------- | ----------------------------------- |
| **Python**        | Core programming language           |
| **TensorFlow**    | Deep learning framework             |
| **Keras**         | CNN model development               |
| **Scikit-learn**  | Machine learning and classification |
| **Random Forest** | Final classifier                    |
| **NumPy**         | Numerical computation               |
| **Pandas**        | Data manipulation                   |
| **OpenCV**        | Image processing                    |
| **Pillow**        | Image loading and manipulation      |
| **NiBabel**       | NIfTI medical image handling        |
| **Nilearn**       | Neuroimaging processing             |
| **Matplotlib**    | Visualization                       |
| **Seaborn**       | Statistical visualization           |
| **Joblib**        | Model serialization                 |
| **Streamlit**     | Web application                     |
| **Git**           | Version control                     |
| **GitHub**        | Source-code hosting                 |

---

# 🌟 Project Highlights

```text
✔ Healthcare AI
✔ Deep Learning
✔ CNN Feature Extraction
✔ Machine Learning
✔ Random Forest
✔ Neuroimaging
✔ Python
✔ TensorFlow / Keras
✔ Scikit-learn
✔ Streamlit
✔ End-to-End ML Pipeline
✔ Research-Oriented Architecture
```

---

# 👨‍💻 Author

## Mahalaxmi Kouchika

**Computer Science Engineering Student**

### Areas of Interest

* 🤖 Machine Learning
* 🧠 Deep Learning
* 🔬 Artificial Intelligence
* ⚙️ MLOps
* ☁️ Cloud Computing
* 🚀 DevOps

---

# 🔗 Repository

**GitHub Repository:**

https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection

---

# ⭐ Support

If you found this project useful for learning, research, or experimentation, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for **educational and research purposes**.

If you want others to freely reuse, modify, and distribute the code, consider adding an appropriate open-source license such as the **MIT License**.

---

# 📌 Quick Start

For users who just want to run the application:

```bash
git clone https://github.com/MahalaxmiKouchika/Autism-Spectrum-Disorder-Detection.git

cd Autism-Spectrum-Disorder-Detection

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🚀 Project at a Glance

```text
                 AUTISM SPECTRUM
                 DISORDER DETECTION
                         │
                         ▼
                 Neuroimaging Data
                         │
                         ▼
                  Preprocessing
                         │
                         ▼
                       CNN
                         │
                         ▼
                Feature Extraction
                         │
                         ▼
                 Random Forest
                         │
                         ▼
                ┌────────┴────────┐
                ▼                 ▼
              ASD              CONTROL
                │                 │
                └────────┬────────┘
                         ▼
                  Prediction Result
                         │
                         ▼
                 Streamlit Dashboard
```

### 🔑 In one sentence

> **An end-to-end healthcare AI research system that processes neuroimaging data, extracts deep features using CNN, classifies ASD vs CONTROL using Random Forest, and provides predictions through an interactive Streamlit application.**
