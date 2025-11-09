
# ♻️ EcoVision — CNN-Based Waste Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/TensorFlow-2.12-orange.svg" alt="TensorFlow Version" />
  <img src="https://img.shields.io/badge/Keras-2.12-red.svg" alt="Keras Version" />
  <img src="https://img.shields.io/badge/OpenCV-4.8-green.svg" alt="OpenCV Version" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle-lightblue.svg" alt="Dataset" />
  <img src="https://img.shields.io/badge/Accuracy-92%25-success.svg" alt="Model Accuracy" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="License" />
  <img src="https://img.shields.io/github/last-commit/Siddhubollam9/CNN_Waste_Classification" alt="Last Commit" />
</p>

---

## 🧠 Project Overview

**EcoVision** is a **Convolutional Neural Network (CNN)** based deep learning model that classifies waste into **Organic (O)** and **Recyclable (R)** categories.  
This helps automate waste segregation, reduce landfill contamination, and improve recycling efficiency.

The model leverages TensorFlow and Keras to train on the **Waste Classification dataset from Kaggle**, achieving **92% test accuracy**.

---

## 🚀 Features

- CNN model trained from scratch for binary waste classification.  
- Real-time image prediction support using OpenCV.  
- Data augmentation for better generalization.  
- Visualization of accuracy/loss curves and confusion matrix.  
- Extendable for multi-class waste categories.  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.10+ |
| **Frameworks** | TensorFlow, Keras |
| **Image Processing** | OpenCV, NumPy |
| **Visualization** | Matplotlib |
| **Dataset Source** | Kaggle Waste Classification Dataset |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Siddhubollam9/CNN_Waste_Classification.git
cd CNN_Waste_Classification

# (Optional) Create a virtual environment
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
````

---

## 🧠 Model Training

```bash
# Run the training script
python train.py
```

You can adjust hyperparameters (epochs, batch size, learning rate) in the script as needed.

---

## 📊 Model Performance

| Class              | Precision | Recall | F1-Score | Support |
| :----------------- | :-------: | :----: | :------: | :-----: |
| **O (Organic)**    |    0.97   |  0.88  |   0.92   |    34   |
| **R (Recyclable)** |    0.88   |  0.97  |   0.92   |    30   |
| **Accuracy**       |           |        | **0.92** |    64   |
| **Macro Avg**      |    0.92   |  0.92  |   0.92   |    64   |
| **Weighted Avg**   |    0.93   |  0.92  |   0.92   |    64   |

---

### 🔍 Confusion Matrix

<p align="center">
  <img src="<img width="798" height="557" alt="Screenshot 2025-11-09 115918" src="https://github.com/user-attachments/assets/9b95bbc9-a83d-4874-b857-89681402fd5a" />
" alt="Confusion Matrix" width="450"/>
</p>

### 💬 Interpretation

* **Accuracy = 92%**, meaning 9 out of 10 predictions are correct.
* **Recyclable recall = 0.97**, showing the model detects recyclables very effectively.
* **Organic precision = 0.97**, meaning very few recyclable items are misclassified as organic.
* **Confusion Matrix:** Only 5 misclassifications total (4 → R, 1 → O).

---

## 📈 Example Predictions

You can test the model with your own images:

```bash
python predict.py --image path_to_image.jpg
```

Expected output:

```
Predicted: Recyclable
Confidence: 0.94
```

---

## 🧩 Future Improvements

* Integrate **Grad-CAM** for model explainability.
* Deploy a **Streamlit web app** for user-friendly predictions.
* Expand dataset to include **metal, glass, and paper** categories.
* Experiment with **transfer learning** (VGG16 / ResNet50) to boost performance.

---

## 📜 License

This project is licensed under the **MIT License** — free for personal and commercial use.

---

## 🤝 Contributors

* **[Siddhubollam9](https://github.com/Siddhubollam9)** – Core Developer
* **Bhuvan Kodikonda** – Model Evaluation & Documentation

---

## 🌍 About

**EcoVision** aims to promote sustainability through AI-driven waste management.
A step toward a cleaner, smarter planet 🌱.

---

```

---

Would you like me to make the **README version with dark-theme badges and logos** (Python logo, TensorFlow flame, Kaggle icon, etc.) for a more *GitHub-pro-level* appearance?  
It’ll look visually richer and more professional to recruiters and judges.
```
