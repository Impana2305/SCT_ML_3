# ============================================================
#  SkillCraft Technology | Machine Learning Internship
#  Task 03: Cat vs Dog Image Classification using SVM
#  Author  : ML Intern – SkillCraft Technology
#  Date    : 2026
# ============================================================

# ─────────────────────────────────────────────────────────────
# SECTION 1 ▸ IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────
"""
This section imports all required libraries for:
  - Data handling         : numpy, os, glob
  - Image processing      : PIL / Pillow  (open, resize, grayscale)
  - Machine Learning      : scikit-learn  (SVM, PCA, StandardScaler, metrics)
  - Visualization         : matplotlib, seaborn
  - Utilities             : tqdm (progress bars), warnings
"""

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Set a consistent random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✅ Libraries imported successfully.")

# ─────────────────────────────────────────────────────────────
# SECTION 2 ▸ DATASET LOADING
# ─────────────────────────────────────────────────────────────
"""
Dataset: Kaggle "Dogs vs. Cats" (PetImages folder structure)
  └── PetImages/
        ├── Cat/  (0.jpg, 1.jpg, …)
        └── Dog/  (0.jpg, 1.jpg, …)

Google Colab setup (run this in a Colab cell BEFORE this script):
  !pip install -q kaggle
  from google.colab import files
  files.upload()          # upload kaggle.json
  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  !kaggle datasets download -d salader/dogs-vs-cats
  !unzip -q dogs-vs-cats.zip -d /content/data

Then set DATASET_PATH = "/content/data/PetImages"
"""

# ── Configuration ──────────────────────────────────────────────
DATASET_PATH = "/content/data/PetImages"   # ← change if needed
IMG_SIZE     = (64, 64)                    # resize all images to 64×64
MAX_IMAGES   = 2000                        # total images per class (set None = all)
LABEL_MAP    = {"Cat": 0, "Dog": 1}
CLASS_NAMES  = ["Cat", "Dog"]

# ── Helper: load images from folder ───────────────────────────
def load_images_from_folder(folder_path, label, img_size=IMG_SIZE, max_count=MAX_IMAGES):
    """
    Reads images from a folder, converts to grayscale, resizes,
    flattens, and returns pixel arrays + labels.

    Parameters
    ----------
    folder_path : str  – path to image folder
    label       : int  – numeric label for images in this folder
    img_size    : tuple – (width, height) to resize images
    max_count   : int  – maximum images to load (None = all)

    Returns
    -------
    images : list[np.ndarray]  – flattened pixel arrays
    labels : list[int]         – corresponding labels
    """
    images, labels = [], []
    extensions = ("*.jpg", "*.jpeg", "*.png")
    file_paths = []

    for ext in extensions:
        file_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if max_count:
        file_paths = file_paths[:max_count]

    for fp in tqdm(file_paths, desc=f"Loading {CLASS_NAMES[label]}s", unit="img"):
        try:
            img = Image.open(fp).convert("L")          # → grayscale
            img = img.resize(img_size, Image.LANCZOS)  # → resize
            arr = np.array(img, dtype=np.float32)       # → numpy
            images.append(arr.flatten())               # → flatten
            labels.append(label)
        except Exception:
            pass  # skip corrupted images silently

    return images, labels


# ── Load cats and dogs ─────────────────────────────────────────
print("\n📂 Loading dataset …\n")
cat_images, cat_labels = load_images_from_folder(
    os.path.join(DATASET_PATH, "Cat"), label=0
)
dog_images, dog_labels = load_images_from_folder(
    os.path.join(DATASET_PATH, "Dog"), label=1
)

# Combine both classes
X_raw = np.array(cat_images + dog_images, dtype=np.float32)
y_raw = np.array(cat_labels + dog_labels, dtype=np.int32)

print(f"\n📊 Dataset Summary")
print(f"   Total samples  : {len(X_raw)}")
print(f"   Cats (label 0) : {sum(y_raw == 0)}")
print(f"   Dogs (label 1) : {sum(y_raw == 1)}")
print(f"   Feature vector : {X_raw.shape[1]} pixels per image ({IMG_SIZE[0]}×{IMG_SIZE[1]})")

# ─────────────────────────────────────────────────────────────
# SECTION 3 ▸ IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────
"""
Steps performed:
  1. Grayscale conversion   – done during loading (PIL "L" mode)
  2. Resize to 64×64        – done during loading
  3. Flatten to 1D vector   – done during loading (4096 features)
  4. Normalize pixel values – divide by 255.0  →  [0.0, 1.0]
  5. StandardScaler         – zero-mean, unit-variance scaling
                              (important for SVM's distance-based decisions)
"""

# Normalize pixel values to [0, 1]
X_normalized = X_raw / 255.0
print("✅ Pixel values normalized to [0, 1]")

# ─────────────────────────────────────────────────────────────
# SECTION 4 ▸ FEATURE EXTRACTION (PCA)
# ─────────────────────────────────────────────────────────────
"""
Why PCA?
  Raw pixel features = 4096 dimensions (64×64) → computationally expensive for SVM.
  PCA reduces dimensionality while retaining maximum variance.
  We keep enough components to explain 95% of the variance.

Note: PCA is fit ONLY on training data to prevent data leakage.
"""

# ─────────────────────────────────────────────────────────────
# SECTION 5 ▸ TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────
"""
Split ratio: 80% training / 20% testing
Stratified  : ensures equal class proportions in both sets
"""

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_normalized, y_raw,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_raw
)

print(f"\n✂️  Train-Test Split (80/20, stratified)")
print(f"   Training samples : {len(X_train_raw)}")
print(f"   Testing  samples : {len(X_test_raw)}")

# StandardScaler: fit on train, transform train+test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)
print("✅ StandardScaler applied")

# PCA: fit on training data only
N_COMPONENTS = 150   # retains most variance, drastically reduces features
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"✅ PCA: {N_COMPONENTS} components → {explained_var:.1f}% variance explained")

# ─────────────────────────────────────────────────────────────
# SECTION 6 ▸ MODEL BUILDING – Support Vector Machine (SVM)
# ─────────────────────────────────────────────────────────────
"""
Kernel  : RBF (Radial Basis Function) – best for high-dimensional image data
C       : Regularization parameter (trade-off between margin and misclassification)
gamma   : Kernel coefficient ('scale' = 1 / (n_features × X.var()))
probability : True – enables predict_proba() for confidence scores

Optional: Uncomment the GridSearchCV block to tune hyperparameters.
"""

# ── (Optional) Hyperparameter Tuning via Grid Search ──────────
"""
param_grid = {
    'C'     : [0.1, 1, 10, 100],
    'gamma' : ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly']
}
grid_search = GridSearchCV(SVC(probability=True), param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_pca, y_train)
print("Best params:", grid_search.best_params_)
svm_model = grid_search.best_estimator_
"""

# ── Train SVM ─────────────────────────────────────────────────
print("\n🤖 Training SVM model …")
svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=RANDOM_STATE
)
svm_model.fit(X_train_pca, y_train)
print("✅ SVM training complete!")

# ─────────────────────────────────────────────────────────────
# SECTION 7 ▸ MODEL EVALUATION
# ─────────────────────────────────────────────────────────────
"""
Metrics used:
  - Accuracy              : overall correct predictions
  - Confusion Matrix      : TP, TN, FP, FN breakdown
  - Classification Report : precision, recall, F1-score per class
"""

# Predictions
y_pred       = svm_model.predict(X_test_pca)
y_pred_proba = svm_model.predict_proba(X_test_pca)  # confidence scores

# ── Accuracy ───────────────────────────────────────────────────
train_acc = accuracy_score(y_train, svm_model.predict(X_train_pca))
test_acc  = accuracy_score(y_test, y_pred)

print(f"\n📈 Model Performance")
print(f"   Training Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"   Testing  Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

# ── Classification Report ─────────────────────────────────────
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ── Confusion Matrix Plot ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task 03 – SVM Cat vs Dog | Model Evaluation", fontsize=14, fontweight="bold")

# Heatmap style
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    ax=axes[0], linewidths=0.5, linecolor="grey"
)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Normalized heatmap (% rates)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
sns.heatmap(
    cm_norm, annot=True, fmt=".2%", cmap="Greens",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    ax=axes[1], linewidths=0.5, linecolor="grey"
)
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Confusion matrix saved → confusion_matrix.png")

# ── PCA Variance Plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    np.cumsum(pca.explained_variance_ratio_) * 100,
    color="#5C6BC0", linewidth=2
)
ax.axhline(95, color="#EF5350", linestyle="--", linewidth=1.2, label="95% threshold")
ax.axvline(N_COMPONENTS, color="#FFA726", linestyle="--", linewidth=1.2,
           label=f"n={N_COMPONENTS} components")
ax.set_xlabel("Number of PCA Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA – Cumulative Explained Variance")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 PCA variance plot saved → pca_variance.png")

# ─────────────────────────────────────────────────────────────
# SECTION 8 ▸ VISUALIZATION OF SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────
"""
Display a grid of 16 test images with:
  - Predicted label (Cat / Dog)
  - Confidence % from predict_proba
  - Green border = correct prediction
  - Red   border = incorrect prediction
"""

def plot_sample_predictions(X_test_raw, y_test, y_pred, y_pred_proba,
                             n_samples=16, img_size=IMG_SIZE):
    """Plot a grid of sample test images with prediction results."""
    indices = np.random.choice(len(y_test), size=n_samples, replace=False)
    cols = 4
    rows = n_samples // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.2))
    fig.suptitle(
        "Sample Predictions – Cat vs Dog SVM Classifier\n"
        "🟢 Correct  🔴 Incorrect",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, idx in zip(axes.flatten(), indices):
        # Reshape flat vector back to 2D image
        img = X_test_raw[idx].reshape(img_size)
        true_label = CLASS_NAMES[y_test[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]] * 100

        is_correct = (y_test[idx] == y_pred[idx])
        border_color = "#43A047" if is_correct else "#E53935"

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}  ({confidence:.1f}%)",
            fontsize=8.5,
            color=border_color,
            fontweight="bold"
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📸 Sample predictions plot saved → sample_predictions.png")


plot_sample_predictions(X_test_raw, y_test, y_pred, y_pred_proba)

# ── Class Distribution Plot ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Dataset Distribution", fontsize=13, fontweight="bold")

labels_all   = np.array(cat_labels + dog_labels)
split_labels = ["Training", "Testing"]
cat_counts   = [sum(y_train == 0), sum(y_test == 0)]
dog_counts   = [sum(y_train == 1), sum(y_test == 1)]

x = np.arange(len(split_labels))
w = 0.35
axes[0].bar(x - w/2, cat_counts, w, label="Cat", color="#42A5F5", edgecolor="white")
axes[0].bar(x + w/2, dog_counts, w, label="Dog", color="#EF5350", edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(split_labels)
axes[0].set_ylabel("Number of Images")
axes[0].set_title("Train / Test Class Distribution")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

total = [sum(labels_all == 0), sum(labels_all == 1)]
axes[1].pie(
    total, labels=CLASS_NAMES, autopct="%1.1f%%",
    colors=["#42A5F5", "#EF5350"], startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
axes[1].set_title("Overall Class Distribution")

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Class distribution plot saved → class_distribution.png")

# ─────────────────────────────────────────────────────────────
# SECTION 9 ▸ PREDICTION ON NEW IMAGE
# ─────────────────────────────────────────────────────────────
"""
Function: predict_new_image()
  - Accepts any image path (JPG/PNG)
  - Applies the exact same preprocessing pipeline:
      grayscale → resize → normalize → StandardScaler → PCA
  - Outputs:
      - Predicted class label (Cat / Dog)
      - Confidence score (%)
      - Display of the image with result annotation
"""

def predict_new_image(image_path: str) -> None:
    """
    Predict whether a new image is a Cat or Dog.

    Parameters
    ----------
    image_path : str – absolute or relative path to the image file
    """
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # ── Preprocessing (mirrors training pipeline) ──────────────
    img_orig = Image.open(image_path).convert("RGB")   # keep for display
    img_gray = img_orig.convert("L")                   # grayscale
    img_resized = img_gray.resize(IMG_SIZE, Image.LANCZOS)

    # Normalize → scale → PCA
    arr = np.array(img_resized, dtype=np.float32).flatten() / 255.0
    arr_scaled = scaler.transform(arr.reshape(1, -1))
    arr_pca    = pca.transform(arr_scaled)

    # ── Inference ──────────────────────────────────────────────
    pred_label = svm_model.predict(arr_pca)[0]
    pred_proba = svm_model.predict_proba(arr_pca)[0]
    class_name  = CLASS_NAMES[pred_label]
    confidence  = pred_proba[pred_label] * 100
    cat_conf    = pred_proba[0] * 100
    dog_conf    = pred_proba[1] * 100

    emoji = "🐱" if pred_label == 0 else "🐶"

    print(f"\n{'='*50}")
    print(f"  Image    : {os.path.basename(image_path)}")
    print(f"  Result   : {emoji} {class_name.upper()} ({confidence:.1f}% confidence)")
    print(f"  Cat prob : {cat_conf:.1f}%")
    print(f"  Dog prob : {dog_conf:.1f}%")
    print(f"{'='*50}")

    # ── Visualization ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"New Image Prediction – SVM Classifier", fontsize=13, fontweight="bold")

    border_color = "#43A047" if pred_label == 0 else "#FF7043"

    # Original image
    axes[0].imshow(img_orig)
    axes[0].set_title(f"{emoji} Predicted: {class_name}\nConfidence: {confidence:.1f}%",
                      fontsize=11, fontweight="bold", color=border_color)
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)

    # Confidence bar chart
    bars = axes[1].barh(
        CLASS_NAMES,
        [cat_conf, dog_conf],
        color=["#42A5F5", "#EF5350"],
        edgecolor="white",
        height=0.4
    )
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Prediction Confidence")
    axes[1].axvline(50, color="grey", linestyle="--", linewidth=0.8, label="50% threshold")

    for bar, val in zip(bars, [cat_conf, dog_conf]):
        axes[1].text(val + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    axes[1].legend(fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("new_image_prediction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📸 Prediction plot saved → new_image_prediction.png")


# ── Example usage ──────────────────────────────────────────────
# Upload an image in Colab using:
#   from google.colab import files
#   uploaded = files.upload()
#   predict_new_image(list(uploaded.keys())[0])

# Or pass a path directly:
# predict_new_image("/content/my_cat.jpg")
# predict_new_image("/content/my_dog.jpg")

# ─────────────────────────────────────────────────────────────
# MODEL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  📦 SCT_ML_Task03 – Model Summary")
print("="*55)
print(f"  Algorithm        : Support Vector Machine (SVM)")
print(f"  Kernel           : RBF (Radial Basis Function)")
print(f"  Regularization C : 10")
print(f"  Image Size       : {IMG_SIZE[0]}×{IMG_SIZE[1]} (grayscale)")
print(f"  Feature Vector   : {IMG_SIZE[0]*IMG_SIZE[1]} → {N_COMPONENTS} (after PCA)")
print(f"  Train Accuracy   : {train_acc*100:.2f}%")
print(f"  Test  Accuracy   : {test_acc*100:.2f}%")
print(f"  Classes          : {CLASS_NAMES}")
print("="*55)
print("\n✅ Task 03 Complete! – SkillCraft Technology ML Internship")
