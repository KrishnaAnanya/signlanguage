# Hybrid ASL Recognition System using MediaPipe and Machine Learning

A real-time American Sign Language (ASL) recognition system that combines MediaPipe hand landmark detection with Machine Learning algorithms to accurately recognize hand gestures and translate them into meaningful alphabet predictions. The project follows a lightweight and efficient approach by extracting hand landmarks instead of processing raw images directly, resulting in faster training and inference.

---

## 📖 Overview

Communication through sign language plays a crucial role for individuals with hearing and speech impairments. This project aims to bridge the communication gap by developing an intelligent ASL recognition system capable of recognizing hand gestures in real time.

The system first collects gesture images and preprocesses them using MediaPipe. Hand landmarks are extracted and converted into NumPy (`.npy`) files, which serve as feature vectors for Machine Learning model training. The trained model is then integrated with a webcam-based interface to perform real-time gesture recognition.

---

## ✨ Features

* Real-time ASL gesture recognition
* MediaPipe-based hand landmark extraction
* Efficient feature storage using NumPy arrays
* Machine Learning-based classification
* Fast and lightweight architecture
* Easy-to-train and scalable pipeline
* Real-time webcam prediction

---

## 🏗️ System Architecture

```text
Gesture Images
      │
      ▼
Dataset Collection
      │
      ▼
MediaPipe Landmark Extraction
      │
      ▼
NumPy Feature Generation (.npy)
      │
      ▼
Machine Learning Training
      │
      ▼
Trained Model
      │
      ▼
Real-Time Gesture Prediction
```

---

## 📂 Project Structure

```text
Hybrid-ASL-Recognition/
│
├── build_dataset.py
├── extract_landmarks.py
├── train_model.py
├── check.py
├── main.py
├── images/
├── README.md
└── .gitignore
```

---

## 🔍 Dataset Preprocessing

The project uses a landmark-based approach instead of directly training on images.

### Step 1: Dataset Collection

Hand gesture images corresponding to different ASL alphabets are collected and organized into classes.

### Step 2: Landmark Extraction

MediaPipe Hands detects 21 key hand landmarks from each image and extracts their spatial coordinates.

### Step 3: Feature Generation

The extracted landmark coordinates are converted into feature vectors and stored as NumPy (`.npy`) files.

Example Feature Vector:

```python
[x1, y1, z1,
 x2, y2, z2,
 ...
 x21, y21, z21]
```

These numerical representations significantly reduce computational complexity while preserving gesture information.

---

## 🧠 Machine Learning Model

The generated landmark features are used to train a Machine Learning classifier capable of distinguishing between different ASL gestures.

Training Pipeline:

```text
Landmark Features
        │
        ▼
Data Preprocessing
        │
        ▼
Train-Test Split
        │
        ▼
Machine Learning Model
        │
        ▼
Model Evaluation
        │
        ▼
Trained Classifier
```

The trained classifier learns gesture-specific patterns from hand landmark coordinates and predicts the corresponding sign during real-time execution.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Hybrid-ASL-Recognition.git
cd Hybrid-ASL-Recognition
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scikit-learn
```

---

## ▶️ Usage

### Build Dataset

```bash
python build_dataset.py
```

### Extract Landmarks

```bash
python extract_landmarks.py
```

### Train Model

```bash
python train_model.py
```

### Validate Model

```bash
python check.py
```

### Run Real-Time Recognition

```bash
python main.py
```

---

## 📸 Project Demonstration

### Dataset Collection

![Dataset Collection](images/dataset_collection.jpeg)

### Hand Landmark Detection

![Landmark Detection](images/landmark_detection.jpeg)

### Real-Time Prediction

![Prediction](images/prediction.jpeg)

---

## 💡 Future Enhancements

* Word-level sign recognition
* Sentence formation
* Deep Learning-based classification
* LSTM-based dynamic gesture recognition
* Text-to-Speech conversion
* Multi-language sign support
* Streamlit/Web deployment

---

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Scikit-Learn
* Machine Learning
* Computer Vision

---
