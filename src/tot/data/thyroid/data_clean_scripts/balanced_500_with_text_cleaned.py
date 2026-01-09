# -*- coding: utf-8 -*-
"""
Construct balanced_500_with_text_cleaned.csv

-------------------------------------- The logic is completely consistent with balanced_100_with_text_cleaned:

- First, select patients, then retrieve all their examination records.

- Each patient shares a unique text_summary.

- 250 positive, 250 negative (of which 200 negative are from those who have been tested, and 50 are from those who have not been tested).

- Random patient laboratory values are empty.
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
OUTPUT_PATH       = DATA_ROOT / "balanced_500_with_text_cleaned.csv"

# ========== read data ==========
thy_text = pd.read_csv(THYROID_TEXT_PATH)
rand_text = pd.read_csv(RANDOM_TEXT_PATH)
labs = pd.read_csv(LABS_PATH)
diag = pd.read_csv(DIAG_PATH)

for df in [thy_text, rand_text, labs, diag]:
    df["subject_id"] = df["subject_id"].astype(int)

# ========== clean text ==========
def clean_text(t):
    if pd.isna(t):
        return t
    t = re.sub(r'### Diagnosis:.*?(?=###|\Z)', '', t, flags=re.DOTALL)
    return t.strip()

thy_text["llm_output"] = thy_text["llm_output"].apply(clean_text)
rand_text["llm_output"] = rand_text["llm_output"].apply(clean_text)

# ========== Labeling confirmed cases ==========
diag_ids = set(diag["subject_id"].astype(int))
thy_text["label"] = thy_text["subject_id"].isin(diag_ids).astype(int)
rand_text["label"] = 0

# ========== 病人分组 ==========
pos_ids = thy_text.query("label==1")["subject_id"].unique()
neg_ids_thy = thy_text.query("label==0")["subject_id"].unique()
neg_ids_rand = rand_text["subject_id"].unique()

np.random.seed(42)
selected_pos = np.random.choice(pos_ids, 250, replace=False)
selected_neg_thy = np.random.choice(neg_ids_thy, 200, replace=False)
selected_neg_rand = np.random.choice(neg_ids_rand, 50, replace=False)

# ========== A summary of 500 patients ==========
selected_ids = np.concatenate([selected_pos, selected_neg_thy, selected_neg_rand])
selected_set = set(selected_ids)

# ========== Retrieve the corresponding patient's laboratory records. ==========
labs_selected = labs[labs["subject_id"].isin(selected_set)].copy()

# ========== Select a unique text for each patient ==========
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

# ========== Patient-by-patient mapping of laboratory records to text and labels ==========
labs_selected[["text_summary", "label"]] = labs_selected["subject_id"].apply(
    lambda sid: pd.Series(get_text_and_label(sid))
)

# ========== Random patient replacement laboratory line ==========
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

# ========== Output (keeping all patient examinations arranged sequentially in chronological order) ==========

# Determine which column can be used for time sorting
time_col = None
for c in ["charttime", "hadm_id", "storetime"]:
    if c in labs_selected.columns:
        time_col = c
        break

if time_col:
    print(f"[INFO] Detection time column：{time_col}")
    labs_selected = labs_selected.sort_values(by=["subject_id", time_col])
else:
    print("[WARN] Time column not found, sort by subject_id")
    labs_selected = labs_selected.sort_values(by="subject_id")

# Output to file
labs_selected.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# ========== Verification output ==========
print(f"Saved to: {OUTPUT_PATH}")
print(f"[INFO] Total number of records (rows): {len(labs_selected)}")
print(f"[INFO] Number of patients: {labs_selected['subject_id'].nunique()}")
print(f"[INFO] Label distribution (patient-level):")
print(labs_selected.groupby('label')['subject_id'].nunique())
print(f"[INFO] Label distribution (row-level):")
print(labs_selected['label'].value_counts())

# Check whether the internal ordering for a single patient is correct
sample_id = labs_selected["subject_id"].iloc[0]
print(f"\n[DEBUG] Example of exam time ordering for patient {sample_id}:")
print(labs_selected[labs_selected["subject_id"] == sample_id][["subject_id", time_col]].head(10))
