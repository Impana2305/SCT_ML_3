"""
Fix: preserve raw pixel images for visualization alongside HOG features.
- Patch Cell 15 (idx 14): add X_test_pixels split from X_normalized
- Patch Cell 25 (idx 24): use X_test_pixels instead of X_test_raw
- Patch Cell 30 (idx 29): use X_test_pixels instead of X_test_raw
"""
import json

NOTEBOOK = r"d:\SCT_ML_3\SCT_ML_Task03_Cat_vs_Dog_SVM.ipynb"
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Fix 1: Cell 15 (idx 14) — add pixel split after HOG split ──
cell14 = nb["cells"][14]
src14 = "".join(cell14["source"])

# Insert pixel image split right after the HOG train/test split block
old_split = (
    'X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n'
    '    X_hog, y_raw,\n'
    '    test_size=0.2,\n'
    '    random_state=RANDOM_STATE,\n'
    '    stratify=y_raw         # ensures equal class ratio in each split\n'
    ')'
)
new_split = (
    'X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n'
    '    X_hog, y_raw,\n'
    '    test_size=0.2,\n'
    '    random_state=RANDOM_STATE,\n'
    '    stratify=y_raw         # ensures equal class ratio in each split\n'
    ')\n'
    '\n'
    '# Keep raw pixel images (same split order) for visualization only\n'
    'X_train_pixels, X_test_pixels, _, _ = train_test_split(\n'
    '    X_normalized, y_raw,\n'
    '    test_size=0.2,\n'
    '    random_state=RANDOM_STATE,\n'
    '    stratify=y_raw\n'
    ')'
)
src14_fixed = src14.replace(old_split, new_split)
assert src14_fixed != src14, "Cell 14 replace failed — pattern not found!"
cell14["source"] = [src14_fixed]
print("Cell 15 (idx 14) fixed: added X_test_pixels split.")

# ── Fix 2: Cell 25 (idx 24) — use X_test_pixels for visualization ──
cell24 = nb["cells"][24]
src24 = "".join(cell24["source"])
src24_fixed = src24.replace(
    "plot_sample_predictions(X_test_raw, y_test, y_pred, y_pred_proba)",
    "plot_sample_predictions(X_test_pixels, y_test, y_pred, y_pred_proba)"
)
assert src24_fixed != src24, "Cell 24 replace failed!"
cell24["source"] = [src24_fixed]
print("Cell 25 (idx 24) fixed: uses X_test_pixels for visualization.")

# ── Fix 3: Cell 30 (idx 29) — use X_test_pixels for batch prediction ──
cell29 = nb["cells"][29]
src29 = "".join(cell29["source"])
src29_fixed = src29.replace(
    "plot_batch_predictions(X_test_raw, y_test, y_pred, y_pred_proba)",
    "plot_batch_predictions(X_test_pixels, y_test, y_pred, y_pred_proba)"
).replace(
    "plot_batch_predictions(X_test_raw, y_test, y_pred, y_pred_proba, class_filter=0)",
    "plot_batch_predictions(X_test_pixels, y_test, y_pred, y_pred_proba, class_filter=0)"
).replace(
    "plot_batch_predictions(X_test_raw, y_test, y_pred, y_pred_proba, class_filter=1)",
    "plot_batch_predictions(X_test_pixels, y_test, y_pred, y_pred_proba, class_filter=1)"
)
cell29["source"] = [src29_fixed]
print("Cell 30 (idx 29) fixed: uses X_test_pixels for batch predictions.")

# ── Save ──────────────────────────────────────────────────────────
with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nNotebook saved. All visualization cells now use raw pixel images.")
