# 🌿 Plant Disease Classification using Deep Learning

## 📌 Project Overview

This project focuses on automatic **plant disease classification** using deep learning techniques on the **PlantVillage dataset**.

We implemented and compared three different architectures:

* 🧠 Custom Convolutional Neural Network (CNN)
* ⚡ EfficientNet (Transfer Learning)
* 🤖 DeiT Vision Transformer (State-of-the-Art)

The goal is to build a **production-ready training pipeline** and evaluate model performance for real-world agricultural applications.

---

## 🎯 Objectives

* Build a custom CNN for baseline performance
* Apply transfer learning using EfficientNet
* Fine-tune a Vision Transformer (DeiT)
* Compare model accuracy and training behavior
* Implement a production-level training pipeline
* Create an inference pipeline for real-time prediction

---

## 📂 Dataset

**Dataset Used:** PlantVillage Dataset

* 54,000+ leaf images
* 38 disease categories
* Multiple plant species

🔗 Dataset Source:
https://www.kaggle.com/datasets/emmarex/plantdisease

---

## 🏗️ Project Structure

```
PlantDisease-Classification/
│
├── train_pipeline.py
├── Predict.py
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── train.py
│   ├── early_stopping.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── visualize.py
│   │
│   └── models/
│       ├── cnn_model.py
│       ├── efficientnet_model.py
│       └── deit_model.py
│
├── notebooks/
│   └── experiment.ipynb
│
├── outputs/
│   ├── models/
│   └──plots/
│
└── README.md
```

---

## 🧠 Models Implemented

### 🔹 Custom CNN

* 3 convolutional layers
* Max pooling
* Fully connected classifier
* Baseline model for comparison

---

### 🔹 EfficientNet

* Pretrained on ImageNet
* Transfer learning approach
* Backbone frozen with trainable classifier

---

### 🔹 DeiT Vision Transformer

* Transformer-based architecture
* State-of-the-art performance
* Fine-tuned classification head

---

## ⚙️ Training Pipeline Features

The project includes a **production-level training pipeline** with:

* Train/Validation/Test splitting
* Early stopping to prevent overfitting
* Learning rate scheduling
* Automatic checkpoint saving
* Config-driven training
* GPU/CPU auto detection
* Reproducibility via seed control

---

## 📊 Model Performance Comparison on Test Dataset

| Model        | Accuracy |
| ------------ | -------- |
| CNN          | ~85.85%     |
| EfficientNet | ~91.83%     |
| DeiT         | ~93.54%     |

*(Results may vary depending on training setup)*

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Train Model

```bash
python train_pipeline.py
```

---

### 3️⃣ Run Prediction

```bash
python Predict.py --image sample.jpg --model deit
```

Available models:

* cnn
* efficientnet
* deit

---

## 📈 Sample Output

```
Prediction Result:
Model Used: DEIT
Predicted Class: Tomato Early Blight
```

---

## 🔬 Technologies Used

* Python
* PyTorch
* torchvision
* timm (Vision Transformers)
* scikit-learn
* matplotlib

---

## 🌍 Applications

* Smart agriculture systems
* Automated disease detection
* Crop monitoring using UAV imagery
* Precision farming solutions

---

## 🚀 Future Improvements

* Deploy as a web application
* Add real-time camera prediction
* Integrate with mobile apps
* Perform multi-disease detection
* Use advanced augmentation techniques

---

## 👨‍💻 Author

**Prashanth Reddy**

Deep Learning & AI Enthusiast
Focus Areas: Computer Vision, MLOps, Agricultural AI

---

## ⭐ If You Like This Project

Please consider giving it a **⭐ on GitHub**!
