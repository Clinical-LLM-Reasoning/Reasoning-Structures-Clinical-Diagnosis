import pandas as pd
from pathlib import Path
import numpy as np

### ==== Configuration ==== ###
SAMPLE_SIZE = "small"  # "mini", "small", "medium"
SAMPLE_MAP = {"mini": 20, "small": 50, "medium": 80}
TOTAL = SAMPLE_MAP[SAMPLE_SIZE]
PER_CLASS = TOTAL // 5

### ==== Paths ==== ###
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ORIG = BASE_DIR / "data_orig"

labs_path = DATA_ORIG / "thyroid_labs_full_1.csv"
diag_path = DATA_ORIG / "thyroid_diagnosis_1.csv"
random_notes_path = DATA_ORIG / "random_patients_with_notes.csv"

print("Loading CSV files...")
labs = pd.read_csv(labs_path)
diag = pd.read_csv(diag_path)
random_notes = pd.read_csv(random_notes_path)

# ======================================================
# Diagnosis table normalization: use icd_code column
# ======================================================
diag["icd_code"] = diag["icd_code"].astype(str).str.upper()

# ======================================================
# Extract TSH / T3 / T4 items from labs
# ======================================================
labs["item"] = labs["test_name"].str.upper()

# Use test_value as numeric value (float)
labs["valuenum"] = pd.to_numeric(labs["test_value"], errors="coerce")

# ======================================================
# Category A: Hyperthyroidism (TSH↓ + E05)
# ======================================================
hyper_ids = set()

# TSH < 0.27
hyper_ids |= set(
    labs[(labs["item"].str.contains("TSH")) &
         (labs["valuenum"] < 0.27)]["subject_id"]
)

# ICD E05*
hyper_ids |= set(diag[diag["icd_code"].str.startswith("E05")]["subject_id"])

hyper_ids = list(hyper_ids)
np.random.shuffle(hyper_ids)
hyper_ids = hyper_ids[:PER_CLASS]

print("A Hyper:", len(hyper_ids))

# ======================================================
# Category B: Hypothyroidism (TSH↑ + E03)
# ======================================================
hypo_ids = set()

# TSH > 4.2
hypo_ids |= set(
    labs[(labs["item"].str.contains("TSH")) &
         (labs["valuenum"] > 4.2)]["subject_id"]
)

# ICD E03*
hypo_ids |= set(diag[diag["icd_code"].str.startswith("E03")]["subject_id"])

hypo_ids = list(hypo_ids)
np.random.shuffle(hypo_ids)
hypo_ids = hypo_ids[:PER_CLASS]

print("B Hypo:", len(hypo_ids))

# ======================================================
# Category C: Borderline (mild abnormality, no diagnosis)
# ======================================================
diag_ids = set(diag["subject_id"])

border_ids = set()

# Borderline: TSH ∈ [3.0, 4.2] without diagnosis
border_ids |= set(
    labs[(labs["item"].str.contains("TSH")) &
         (labs["valuenum"].between(3.0, 4.2)) &
         (~labs["subject_id"].isin(diag_ids))]["subject_id"]
)

# Borderline low-normal: TSH ∈ [0.27, 0.40]
border_ids |= set(
    labs[(labs["item"].str.contains("TSH")) &
         (labs["valuenum"].between(0.27, 0.40)) &
         (~labs["subject_id"].isin(diag_ids))]["subject_id"]
)

border_ids = list(border_ids)
np.random.shuffle(border_ids)
border_ids = border_ids[:PER_CLASS]

print("C Borderline:", len(border_ids))

# ======================================================
# Category D: Multiple normal measurements
# ======================================================
def is_normal(group):
    for _, row in group.iterrows():
        v = row["valuenum"]
        name = row["item"]
        if pd.isna(v):
            continue
        if "TSH" in name and not (0.27 <= v <= 4.2):
            return False
        if "T3" in name and not (0.8 <= v <= 2.0):
            return False
        if "T4" in name and not (0.8 <= v <= 1.9):
            return False
    return True

multi_normal = []

for sid, g in labs.groupby("subject_id"):
    if len(g) >= 3 and is_normal(g):
        multi_normal.append(sid)

np.random.shuffle(multi_normal)
multi_normal = multi_normal[:PER_CLASS]

print("D Normal:", len(multi_normal))

# ======================================================
# Category E: Random normal
# ======================================================
random_ids = list(set(random_notes["subject_id"]) - diag_ids)
np.random.shuffle(random_ids)
random_ids = random_ids[:PER_CLASS]

print("E Random:", len(random_ids))

# ======================================================
# Output
# ======================================================
final_ids = list(set(
    hyper_ids + hypo_ids + border_ids + multi_normal + random_ids
))

print("Final count:", len(final_ids))

output_path = BASE_DIR / "data_cleaned" / f"small_eval_dataset_{TOTAL}.csv"
pd.DataFrame({"subject_id": final_ids}).to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
