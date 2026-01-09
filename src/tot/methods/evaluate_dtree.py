# -*- coding: utf-8 -*-
"""
DTree Evaluation (with -1 treated as WRONG)
-------------------------------------------
- -1 is unclassifiable → treated as an error

- However, the following will be additionally counted:

(1) The number of samples with a true label of 0 in -1 (becomes FP)

(2) The number of samples with a true label of 1 in -1 (becomes FN)

- Accuracy is based on all samples (including -1)
"""

import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# ====== dtree json 路径 ======
PRED_PATH = Path(
    r"D:\mimic_project\Thesis_mimic_project\experiments\tree-of-thought-llm\outputs\llm_dtree_predictions.json"
)

# ====== read ======
records = []
with open(PRED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# ====== ground truth ======
y_true = np.array([r["y_true"] for r in records], dtype=int)

# ====== Obtain the original dtree prediction from final_output. ======
y_pred_raw = []
for r in records:
    out = r["final_output"].strip()
    if out == "-1":
        y_pred_raw.append(-1)
    elif out == "0":
        y_pred_raw.append(0)
    elif out == "1":
        y_pred_raw.append(1)
    else:
        # Fallback handling for "Final: x"
        if "Final:" in out:
            if "1" in out:
                y_pred_raw.append(1)
            elif "0" in out:
                y_pred_raw.append(0)
            else:
                y_pred_raw.append(-1)
        else:
            y_pred_raw.append(-1)

y_pred_raw = np.array(y_pred_raw, dtype=int)

# ====== Statistics -1 ======
uncertain_mask = (y_pred_raw == -1)
num_uncertain = uncertain_mask.sum()
total = len(y_pred_raw)

print("====================================")
print("     DTree Evaluation (RAW + ERROR)")
print("====================================")
print(f"Total samples:      {total}")
print(f"Uncertain (-1):     {num_uncertain} ({num_uncertain/total:.2%})")

# ====== Decomposing -1, does it belong to FP or FN? ======
uncertain_true_0 = ((y_pred_raw == -1) & (y_true == 0)).sum()
uncertain_true_1 = ((y_pred_raw == -1) & (y_true == 1)).sum()

print("\n--- Uncertain Error Decomposition ---")
print(f"-1 with True=0 → counted as FP : {uncertain_true_0}")
print(f"-1 with True=1 → counted as FN : {uncertain_true_1}")

# ====== Convert -1 to an incorrect prediction for use in regular binary classification evaluation. ======
y_pred_corr = y_pred_raw.copy()
y_pred_corr[y_pred_corr == -1] = 1 - y_true[y_pred_corr == -1]

print("\n=== Confusion Matrix (0/1 only) ===")
print(confusion_matrix(y_true, y_pred_corr))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_corr, digits=3))
