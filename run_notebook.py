
# ══════════════════════════════════════════════════════════════
#  SCT_ML_Task03 — Cat vs Dog SVM  |  Local Runner
#  Dataset: D:\Cat vs Dog Datasets  (cats_set / dogs_set)
# ══════════════════════════════════════════════════════════════

# ── Standard Library ──────────────────────────────────────────
import os, glob, warnings

# ── Numerical Computing ───────────────────────────────────────
import numpy as np

# ── Image Processing ──────────────────────────────────────────
from PIL import Image
from tqdm import tqdm

# ── Visualization ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Machine Learning ──────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "figure.facecolor" : "#FAFAFA",
    "axes.facecolor"   : "#FFFFFF",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
})

import sklearn
print("✅ All libraries imported successfully.")
print(f"   NumPy   : {np.__version__}")
print(f"   sklearn : {sklearn.__version__}")

# ══════════════════════════════════════════════════════════════
# SECTION 2 ▸ DATASET CONFIGURATION
# ══════════════════════════════════════════════════════════════
DATASET_PATH = r"D:\Cat vs Dog Datasets"
CAT_FOLDER   = "cats_set"
DOG_FOLDER   = "dogs_set"

IMG_SIZE    = (64, 64)
MAX_IMAGES  = 500           # 500 per class (full mini dataset)
CLASS_NAMES = ["Cat", "Dog"]
LABEL_MAP   = {"Cat": 0, "Dog": 1}

print(f"\n📁 Dataset path : {DATASET_PATH}")
print(f"   Cat folder  : {CAT_FOLDER}")
print(f"   Dog folder  : {DOG_FOLDER}")
print(f"🖼️  Image size   : {IMG_SIZE}")
print(f"🔢  Max per class: {MAX_IMAGES}")

# ══════════════════════════════════════════════════════════════
# SECTION 2 ▸ IMAGE LOADER
# ══════════════════════════════════════════════════════════════
def load_images_from_folder(folder_path, label, img_size=IMG_SIZE, max_count=MAX_IMAGES):
    images, labels, file_paths = [], [], []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        file_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    if max_count:
        file_paths = file_paths[:max_count]
    class_name = CLASS_NAMES[label]
    for fp in tqdm(file_paths, desc=f"  Loading {class_name}s", unit="img"):
        try:
            img = Image.open(fp).convert("L")
            img = img.resize(img_size, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            images.append(arr.flatten())
            labels.append(label)
        except Exception:
            pass
    return images, labels

print("\n📂 Loading dataset …\n")
cat_images, cat_labels = load_images_from_folder(os.path.join(DATASET_PATH, CAT_FOLDER), label=0)
dog_images, dog_labels = load_images_from_folder(os.path.join(DATASET_PATH, DOG_FOLDER), label=1)

X_raw = np.array(cat_images + dog_images, dtype=np.float32)
y_raw = np.array(cat_labels + dog_labels, dtype=np.int32)

print(f"\n{'─'*42}")
print(f"  Dataset loaded successfully!")
print(f"{'─'*42}")
print(f"  Total samples     : {len(X_raw):,}")
print(f"  Cat images (0)    : {sum(y_raw == 0):,}")
print(f"  Dog images (1)    : {sum(y_raw == 1):,}")
print(f"  Feature dimensions: {X_raw.shape[1]:,}  ({IMG_SIZE[0]}×{IMG_SIZE[1]} grayscale)")

# ── Sample Images Grid ─────────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
fig.suptitle("📸 Sample Images from Dataset", fontsize=14, fontweight="bold", y=1.02)
for col, idx in enumerate(np.where(y_raw == 0)[0][:8]):
    axes[0, col].imshow(X_raw[idx].reshape(IMG_SIZE), cmap="gray")
    axes[0, col].set_title("Cat", fontsize=8); axes[0, col].axis("off")
for col, idx in enumerate(np.where(y_raw == 1)[0][:8]):
    axes[1, col].imshow(X_raw[idx].reshape(IMG_SIZE), cmap="gray")
    axes[1, col].set_title("Dog", fontsize=8); axes[1, col].axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → sample_images.png")

# ══════════════════════════════════════════════════════════════
# SECTION 3 ▸ PREPROCESSING
# ══════════════════════════════════════════════════════════════
X_normalized = X_raw / 255.0
print(f"\nPreprocessing Pipeline:")
print(f"  [✔] Grayscale + Resize + Flatten → done during loading")
print(f"  [✔] Normalize /255.0  → range [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Pixel Intensity Distribution", fontsize=13, fontweight="bold")
axes[0].hist(X_raw[0], bins=50, color="#5C6BC0", edgecolor="white", linewidth=0.3)
axes[0].set_title("Before Normalization"); axes[0].set_xlim(0, 255)
axes[1].hist(X_normalized[0], bins=50, color="#26A69A", edgecolor="white", linewidth=0.3)
axes[1].set_title("After Normalization"); axes[1].set_xlim(0, 1)
plt.tight_layout()
plt.savefig("pixel_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → pixel_distribution.png")

# ══════════════════════════════════════════════════════════════
# SECTION 5 ▸ TRAIN-TEST SPLIT + SECTION 4 ▸ PCA
# ══════════════════════════════════════════════════════════════
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_normalized, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_raw
)
print(f"\n✂️  Train-Test Split (80/20, stratified)")
print(f"   Training : {len(X_train_raw):,} | Cats: {sum(y_train==0):,}  Dogs: {sum(y_train==1):,}")
print(f"   Testing  : {len(X_test_raw):,}  | Cats: {sum(y_test==0):,}   Dogs: {sum(y_test==1):,}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)
print(f"\n[✔] StandardScaler applied")

N_COMPONENTS = 100   # reduced for 500-image mini dataset
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"[✔] PCA applied  →  {N_COMPONENTS} components, {explained_var:.1f}% variance retained")

# PCA plot
cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("PCA Feature Extraction Analysis", fontsize=13, fontweight="bold")
axes[0].plot(cumulative_var, color="#5C6BC0", linewidth=2.5)
axes[0].fill_between(range(len(cumulative_var)), cumulative_var, alpha=0.1, color="#5C6BC0")
axes[0].axhline(95, color="#EF5350", linestyle="--", linewidth=1.5, label="95% threshold")
axes[0].axvline(N_COMPONENTS-1, color="#FFA726", linestyle="--", linewidth=1.5, label=f"n={N_COMPONENTS} → {explained_var:.0f}%")
axes[0].set_title("Cumulative Explained Variance"); axes[0].legend(); axes[0].grid(alpha=0.25)
axes[1].bar(range(min(40,N_COMPONENTS)), pca.explained_variance_ratio_[:40]*100, color="#26A69A", edgecolor="white")
axes[1].set_title("Top 40 Principal Components"); axes[1].grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("pca_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → pca_analysis.png")

# ══════════════════════════════════════════════════════════════
# SECTION 6 ▸ TRAIN SVM
# ══════════════════════════════════════════════════════════════
print("\n🤖 Training SVM model …  (Kernel: RBF | C=10 | gamma=scale)\n")
svm_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
svm_model.fit(X_train_pca, y_train)
print(f"✅ SVM training complete!")
print(f"   Support vectors : Cat={svm_model.n_support_[0]}, Dog={svm_model.n_support_[1]}, Total={sum(svm_model.n_support_)}")

# ══════════════════════════════════════════════════════════════
# SECTION 7 ▸ EVALUATION
# ══════════════════════════════════════════════════════════════
y_pred       = svm_model.predict(X_test_pca)
y_pred_proba = svm_model.predict_proba(X_test_pca)
train_acc    = accuracy_score(y_train, svm_model.predict(X_train_pca))
test_acc     = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("  📈 MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"  Training Accuracy : {train_acc*100:.2f}%")
print(f"  Testing  Accuracy : {test_acc*100:.2f}%")
print("="*50)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SVM Confusion Matrix — Cat vs Dog", fontsize=14, fontweight="bold")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=axes[0], annot_kws={"size":16,"weight":"bold"})
axes[0].set_title("Counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=axes[1], annot_kws={"size":14,"weight":"bold"})
axes[1].set_title("Normalized (Recall)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → confusion_matrix.png")

# Per-class metrics bar chart
metrics = {
    "Precision": precision_score(y_test, y_pred, average=None),
    "Recall"   : recall_score(y_test, y_pred, average=None),
    "F1-Score" : f1_score(y_test, y_pred, average=None),
}
x, w, colors = np.arange(2), 0.22, ["#42A5F5", "#26A69A", "#AB47BC"]
fig, ax = plt.subplots(figsize=(9, 5))
for i, (mname, vals) in enumerate(metrics.items()):
    bars = ax.bar(x + i*w, vals, w, label=mname, color=colors[i], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x+w); ax.set_xticklabels(CLASS_NAMES, fontsize=12)
ax.set_ylim(0, 1.12); ax.set_title("Per-Class Precision, Recall & F1-Score", fontsize=13, fontweight="bold")
ax.axhline(test_acc, color="#EF5350", linestyle="--", linewidth=1.2, label=f"Overall Acc: {test_acc*100:.1f}%")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("metrics_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → metrics_chart.png")

# ══════════════════════════════════════════════════════════════
# SECTION 8 ▸ SAMPLE PREDICTIONS
# ══════════════════════════════════════════════════════════════
rng = np.random.default_rng(0)
n   = 16
indices = rng.choice(len(y_test), size=n, replace=False)
fig, axes = plt.subplots(4, 4, figsize=(14, 13))
fig.patch.set_facecolor("#F5F5F5")
fig.suptitle("Sample Predictions — Cat vs Dog SVM\n🟢 Correct   🔴 Incorrect",
             fontsize=13, fontweight="bold", y=1.01)
correct = 0
for ax, idx in zip(axes.flatten(), indices):
    img        = X_test_raw[idx].reshape(IMG_SIZE)
    true_name  = CLASS_NAMES[y_test[idx]]
    pred_name  = CLASS_NAMES[y_pred[idx]]
    confidence = y_pred_proba[idx][y_pred[idx]] * 100
    is_correct = (y_test[idx] == y_pred[idx])
    border_clr = "#43A047" if is_correct else "#E53935"
    emoji      = "✅" if is_correct else "❌"
    correct   += is_correct
    ax.imshow(img, cmap="gray", aspect="auto")
    ax.set_title(f"{emoji} {pred_name}  ({confidence:.1f}%)\nTrue: {true_name}",
                 fontsize=8, color=border_clr, fontweight="bold", pad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_clr); spine.set_linewidth(3.5)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📸 Saved → sample_predictions.png  |  Correct in grid: {correct}/{n}")

# Confidence distribution
max_conf = y_pred_proba.max(axis=1) * 100
correct_mask = (y_pred == y_test)
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(max_conf[correct_mask],  bins=30, alpha=0.75, label="Correct",   color="#43A047", edgecolor="white")
ax.hist(max_conf[~correct_mask], bins=30, alpha=0.75, label="Incorrect", color="#E53935", edgecolor="white")
ax.axvline(50, color="grey", linestyle="--", linewidth=1.2)
ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Confidence (%)"); ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("confidence_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved → confidence_distribution.png")

# ══════════════════════════════════════════════════════════════
# SECTION 10 ▸ FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print()
print("╔" + "═"*54 + "╗")
print("║  📦  SCT_ML_Task03 — Final Model Summary          ║")
print("╠" + "═"*54 + "╣")
print(f"║  Algorithm      : SVM — RBF Kernel                ║")
print(f"║  C=10  |  gamma=scale  |  PCA={N_COMPONENTS} components      ║")
print("╠" + "═"*54 + "╣")
print(f"║  Total Images   : {len(X_raw):,}                            ║")
print(f"║  Training Set   : {len(X_train_raw):,}  (80%)                   ║")
print(f"║  Test Set       : {len(X_test_raw):,}   (20%)                   ║")
print("╠" + "═"*54 + "╣")
print(f"║  Train Accuracy : {train_acc*100:.2f}%                        ║")
print(f"║  Test  Accuracy : {test_acc*100:.2f}%                        ║")
print("╠" + "═"*54 + "╣")
print(f"║  Output PNGs saved to: d:\\SCT_ML_3\\              ║")
print("╚" + "═"*54 + "╝")
print()
print("✅  Task 03 Complete! — SkillCraft Technology ML Internship")
