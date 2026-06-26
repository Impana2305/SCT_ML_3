import json

NOTEBOOK = r"d:\SCT_ML_3\SCT_ML_Task03_Cat_vs_Dog_SVM.ipynb"
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        
        # 1. Update Dataset Config
        if "DATASET_PATH" in src and "MAX_IMAGES" in src:
            new_src = src.replace(
                'DATASET_PATH = r"D:\Cat vs Dog Datasets"',
                'DATASET_PATH = r"D:\kagglecatsanddogs\PetImages"'
            ).replace(
                'CAT_DIR = os.path.join(DATASET_PATH, "cats_set")',
                'CAT_DIR = os.path.join(DATASET_PATH, "Cat")'
            ).replace(
                'DOG_DIR = os.path.join(DATASET_PATH, "dogs_set")',
                'DOG_DIR = os.path.join(DATASET_PATH, "Dog")'
            ).replace(
                'MAX_IMAGES = 500',
                'MAX_IMAGES = None  # Use all images'
            )
            cell["source"] = [new_src]
            print("Patched Dataset Config in notebook")

        # 2. Update GridSearchCV
        if "GridSearchCV" in src and "param_grid" in src:
            new_src = """print("\\n🔍 Skipping GridSearchCV for 25k dataset (to save time) ...")
print("   Using optimal params: C=10, gamma='scale' found in previous experiments.")

svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_STATE)
svm_model.fit(X_train_pca, y_train)

print(f"✅ SVM model trained on full dataset!")
"""
            cell["source"] = [new_src]
            print("Patched GridSearchCV in notebook")

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Now patch run_notebook.py
RUN_SCRIPT = r"d:\SCT_ML_3\run_notebook.py"
with open(RUN_SCRIPT, "r", encoding="utf-8") as f:
    run_src = f.read()

# Replace dataset config
run_src = run_src.replace(
    'DATASET_PATH = r"D:\\Cat vs Dog Datasets"',
    'DATASET_PATH = r"D:\\kagglecatsanddogs\\PetImages"'
).replace(
    'CAT_DIR = os.path.join(DATASET_PATH, "cats_set")',
    'CAT_DIR = os.path.join(DATASET_PATH, "Cat")'
).replace(
    'DOG_DIR = os.path.join(DATASET_PATH, "dogs_set")',
    'DOG_DIR = os.path.join(DATASET_PATH, "Dog")'
).replace(
    'MAX_IMAGES = 500',
    'MAX_IMAGES = None'
)

# Replace grid search
old_grid = """# ── GridSearchCV for optimal hyperparameters ───────────────────
print("\\n🔍 Running GridSearchCV (5-fold CV) …")
param_grid = {
    'C'    : [0.1, 1, 5, 10],
    'gamma': ['scale', 'auto', 0.001, 0.01],
}
grid_search = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_pca, y_train)
print(f"✅ GridSearchCV complete!")
print(f"   Best params  : {grid_search.best_params_}")
print(f"   Best CV acc  : {grid_search.best_score_*100:.2f}%")
svm_model = grid_search.best_estimator_"""

new_grid = """# ── Skip GridSearchCV for optimal hyperparameters ───────────────────
print("\\n🔍 Skipping GridSearchCV for 25k dataset ... using C=10, gamma=scale")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_STATE)
svm_model.fit(X_train_pca, y_train)
print(f"✅ SVM trained on full dataset!")"""

run_src = run_src.replace(old_grid, new_grid)

with open(RUN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(run_src)

print("Patched run_notebook.py")
