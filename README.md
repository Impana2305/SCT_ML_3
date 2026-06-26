<div align="center">

# 🐱🐶 Cat vs Dog Image Classification using SVM

### SkillCraft Technology — Machine Learning Internship | Task 03

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

This project is **Task 03** of the **SkillCraft Technology Machine Learning Internship**.

The goal is to build an image classification pipeline that distinguishes between **cats 🐱** and **dogs 🐶** using a **Support Vector Machine (SVM)** with an RBF kernel, combined with PCA-based dimensionality reduction for efficient training.

---

## 🗂️ Project Structure

```
SCT_ML_3/
├── SCT_ML_Task03_Cat_vs_Dog_SVM.ipynb   ← Main notebook (Colab-ready)
├── SCT_ML_Task03_Cat_vs_Dog_SVM.py      ← Python script version
├── requirements.txt                      ← Python dependencies
└── README.md                             ← Project documentation
```

---

## 🚀 Project Workflow

```
Dataset Loading
     │
     ▼
Image Preprocessing
(Grayscale → Resize 64×64 → Normalize → HOG Feature Extraction)
     │
     ▼
Train-Test Split  (80% / 20%, stratified)
     │
     ▼
StandardScaler  →  PCA (50 components, ~40% variance)
     │
     ▼
SVM Model Training  (RBF kernel, C=10)
     │
     ▼
Model Evaluation
(Accuracy · Confusion Matrix · Classification Report)
     │
     ▼
Visualizations + Prediction on New Image
```

---

## 🧠 Technologies Used

| Library | Purpose |
|---------|---------|
| `scikit-learn` | SVM model, PCA, StandardScaler, metrics |
| `Pillow (PIL)` | Image loading, grayscale conversion, resizing |
| `NumPy` | Array operations, data handling |
| `Matplotlib` | Plots, prediction visualizations |
| `Seaborn` | Styled confusion matrix heatmaps |
| `tqdm` | Progress bars during image loading |

---

## 📊 Preprocessing Pipeline

| Step | Operation | Result |
|------|-----------|--------|
| 1 | **Grayscale** | Remove color channels |
| 2 | **Resize** to 64×64 | Uniform dimensions |
| 3 | **Normalize** ÷255 | Pixel range [0.0, 1.0] |
| 4 | **HOG Extraction** | Captures edge directions |
| 5 | **StandardScaler** | Zero-mean, unit-variance |
| 6 | **PCA** (50 components) | 40.4% variance retained |

---

## 🤖 SVM Model Configuration

```python
SVC(
    kernel      = 'rbf',    # RBF: handles non-linear image boundaries
    C           = 10,       # Regularization strength
    gamma       = 'scale',  # Kernel width = 1 / (n_features × var(X))
    probability = True,     # Enables confidence scores
    random_state= 42
)
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.69% |
| Testing Accuracy | 75.00% |
| PCA Components | 50 |
| Variance Retained | 40.4% |

> *Results vary with dataset size and random seed. Accuracy improves with more training images.*

---

## 📁 Dataset

- **Source**: [Kaggle – Dogs vs. Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **Structure**:
  ```
  PetImages/
  ├── Cat/   (12,500 images)
  └── Dog/   (12,500 images)
  ```
- **Used**: 5,000 images per class (10,000 total) (configurable via `MAX_IMAGES`)

---

## ▶️ How to Run

### Google Colab (Recommended)

1. Open `SCT_ML_Task03_Cat_vs_Dog_SVM.ipynb` in Colab
2. Run **Section 0** to install packages and download the dataset via Kaggle API
3. Run all cells sequentially

### Local Environment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SCT_ML_3.git
cd SCT_ML_3

# Install dependencies
pip install -r requirements.txt

# Download dataset manually and update DATASET_PATH in the notebook
# Then run the notebook or script:
python SCT_ML_Task03_Cat_vs_Dog_SVM.py
```

---

## 🖼️ Output Visualizations

| File | Description |
|------|-------------|
| `sample_images.png` | Grid of raw dataset samples |
| `pixel_distribution.png` | Before/after normalization histogram |
| `pca_analysis.png` | Cumulative variance + top components |
| `class_distribution.png` | Train/test class counts + pie chart |
| `confusion_matrix.png` | Raw counts + normalized heatmaps |
| `metrics_chart.png` | Per-class Precision, Recall, F1 |
| `sample_predictions.png` | 4×4 prediction grid with borders |
| `confidence_distribution.png` | Correct vs incorrect confidence histogram |
| `new_image_prediction.png` | Single image prediction with bar chart |

---

## 🎯 Key Learnings

- Image preprocessing techniques (grayscale, resize, normalize)
- Dimensionality reduction using PCA to speed up SVM
- Applying SVM (RBF kernel) to high-dimensional image data
- Evaluating models with confusion matrices and classification reports
- Building a reusable inference pipeline for new images

---

## 🔗 Future Improvements

- Use CNN (Convolutional Neural Networks) for significantly higher accuracy
- Implement data augmentation to handle class imbalance
- Build a web application to visualize results (Done! React/Vite dashboard deployed on Vercel)

---

## 👩‍💻 Author

**Machine Learning Intern** — SkillCraft Technology

---

<div align="center">
  <sub>Part of the SkillCraft Technology ML Internship Series · Task 03 of 4</sub>
</div>
