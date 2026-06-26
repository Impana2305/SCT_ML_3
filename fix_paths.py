import json
with open(r'd:\SCT_ML_3\SCT_ML_Task03_Cat_vs_Dog_SVM.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        if 'cats_set' in src or 'dogs_set' in src:
            src = src.replace('"cats_set"', '"Cat"')
            src = src.replace('"dogs_set"', '"Dog"')
            cell['source'] = [src]

with open(r'd:\SCT_ML_3\SCT_ML_Task03_Cat_vs_Dog_SVM.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed cats_set to Cat in notebook.")
