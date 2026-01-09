# -*- coding: utf-8 -*-
"""
Build the full dataset: full_with_text_cleaned.csv
--------------------------------------------------
Differences:
- No random sampling
- Use all positive patients + all negative patients (thy_text + rand_text)
- All other logic remains unchanged:
  * Text cleaning
  * Assign a unique text_summary per patient
  * Retrieve laboratory records
  * Add empty lab records for random patients
  * Temporal sorting
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ========== Path configuration ==========
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent.parent / "data" / "data_orig"

THYROID_TEXT_PATH = DATA_ROOT / "llm_thyroid_summary_filtered_original.csv"
RANDOM_TEXT_PATH  = DATA_ROOT / "llm_random_summary_filtered_original.csv"
LABS_PATH         = DATA_ROOT / "thyroid_labs_full_1.csv"
DIAG_PATH         = DATA_ROOT / "thyroid_diagnosis_1.csv"
OUTPUT_PATH       = DATA_ROOT / "full_with_text_cleaned.csv"   # New output name

# ========== Load data ==========
thy_text = pd.read_csv(THYROID_TEXT_PATH)
rand_text = pd.read_csv(RANDOM_TEXT_PATH)
labs = pd.read_csv(LABS_PATH)
diag = pd.read_csv(DIAG_PATH)

for df in [thy_text, rand_text, labs, diag]:
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(int)

# ========== Text cleaning ==========
def clean_text(t):
    if pd.isna(t):
        return t
    t = re.sub(r'### Diagnosis:.*?(?=###|\Z)', '', t, flags=re.DOTALL)
    return t.strip()

thy_text["llm_output"] = thy_text["llm_output"].apply(clean_text)
rand_text["llm_output"] = rand_text["llm_output"].apply(clean_text)

# ========== Assign diagnostic labels ==========
diag_ids = set(diag["subject_id"].astype(int))
thy_text["label"] = thy_text["subject_id"].isin(diag_ids).astype(int)
rand_text["label"] = 0   # Random patients are always negative

# ============================================================
# 🚀 Change: no sampling! use all patients
# ============================================================
selected_pos      = thy_text.query("label==1")["subject_id"].unique()
selected_neg_thy  = thy_text.query("label==0")["subject_id"].unique()
selected_neg_rand = rand_text["subject_id"].unique()

# Full patient set
selected_ids = list(selected_pos) + list(selected_neg_thy) + list(selected_neg_rand)
selected_set = set(selected_ids)

print(f"[INFO] Total positive patients: {len(selected_pos)}")
print(f"[INFO] Total negative (thyroid) patients: {len(selected_neg_thy)}")
print(f"[INFO] Total random negative patients: {len(selected_neg_rand)}")
print(f"[INFO] Total patients: {len(selected_set)}")

# ========== Retrieve all lab records for selected patients ==========
labs_selected = labs[labs["subject_id"].isin(selected_set)].copy()

# ========== Bind a unique text per patient ==========
thyroid_text_map = (
    thy_text.drop_duplicates(subset="subject_id")[["subject_id", "llm_output", "label"]]
    .set_index("subject_id")
    .to_dict(orient="index")
)
random_text_map = (
    rand_text.drop_duplicates(subset="subject_id")[["subject_id", "llm_output", "label"]]
    .set_index("subject_id")
    .to_dict(orient="index")
)

def get_text_and_label(sid):
    if sid in thyroid_text_map:
        return thyroid_text_map[sid]["llm_output"], thyroid_text_map[sid]["label"]
    elif sid in random_text_map:
        return random_text_map[sid]["llm_output"], 0
    else:
        return pd.NA, pd.NA

labs_selected[["text_summary", "label"]] = labs_selected["subject_id"].apply(
    lambda sid: pd.Series(get_text_and_label(sid))
)

# ========== Add empty lab rows for random patients ==========
lab_cols = [c for c in labs.columns if c != "subject_id"]
rand_ids_to_add = [sid for sid in selected_neg_rand if sid not in labs_selected["subject_id"].unique()]

rand_rows = []
for sid in rand_ids_to_add:
    r = {col: pd.NA for col in lab_cols}
    r["subject_id"] = sid
    txt, lbl = get_text_and_label(sid)
    r["text_summary"] = txt
    r["label"] = lbl
    rand_rows.append(r)

if rand_rows:
    labs_selected = pd.concat([labs_selected, pd.DataFrame(rand_rows)], ignore_index=True)

# ========== Sort and export ==========
time_col = None
for c in ["charttime", "hadm_id", "storetime"]:
    if c in labs_selected.columns:
        time_col = c
        break

if time_col:
    print(f"[INFO] Detected time column: {time_col}")
    labs_selected = labs_selected.sort_values(by=["subject_id", time_col])
else:
    print("[WARN] No time column found, sorting by subject_id")
    labs_selected = labs_selected.sort_values(by="subject_id")

# Export
labs_selected.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n[INFO] Saved to: {OUTPUT_PATH}")
print(f"[INFO] Total number of records (rows): {len(labs_selected)}")
print(f"[INFO] Number of patients: {labs_selected['subject_id'].nunique()}")
print(f"[INFO] Label distribution (patient-level):")
print(labs_selected.groupby('label')['subject_id'].nunique())
print(f"[INFO] Label distribution (row-level):")
print(labs_selected['label'].value_counts())

# Inspect an example
sample_id = labs_selected["subject_id"].iloc[0]
print(f"\n[DEBUG] Example records for patient {sample_id}:")
print(labs_selected[labs_selected["subject_id"] == sample_id][["subject_id", time_col]].head(10))
