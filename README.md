# 🫁 Lung Cancer Detection using Hybrid CNN–Transformer

This project presents a deep learning-based approach for **automated lung cancer classification** using CT scan images. The model combines the strengths of **Convolutional Neural Networks (CNNs)** and **Transformer architectures** to achieve high accuracy and robust performance.

---

## 🚀 Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection is critical for improving survival rates. This project proposes a **Hybrid CNN–Transformer model** to classify lung CT images into three categories:

- **Normal**
- **Benign**
- **Malignant**

The model leverages:
- CNN (ResNet18) → for local feature extraction  
- Transformer Encoder → for global context understanding  

---

## 🧠 Model Architecture

The proposed architecture consists of:

1. **Input CT Image (224×224)**
2. **ResNet18 Backbone (Feature Extraction)**
3. **1×1 Convolution (Dimensionality Reduction)**
4. **Transformer Encoder (Global Attention)**
5. **Fully Connected Layer (Classification)**

---

## 📊 Results

| Metric        | Score |
|--------------|------|
| Accuracy     | 98.00% |
| Precision    | 98.24% |
| Recall       | 96.39% |
| F1-Score     | 97.23% |

🔹 **Malignant Recall: 100% (No false negatives)**  
➡️ Critical for clinical applications

---

## 📁 Dataset

- Dataset: IQ-OTHNCCD Lung Cancer Dataset  
- Source: Kaggle  
- Classes:
  - Normal
  - Benign
  - Malignant

---

## 🛠️ Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## ⚙️ Installation

```bash
git clone https://github.com/sadashiv139/lung_cancer_detection.git
cd lung_cancer_detection
pip install -r requirements.txt
