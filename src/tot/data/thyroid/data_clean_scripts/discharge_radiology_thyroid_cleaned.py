# -*- coding: utf-8 -*-
"""
Deduplication and cleaning: thyroid-related discharge + radiology text data
Objectives:
  - Keep only one row per (subject_id, hadm_id)
  - Merge multiple discharge_text and radiology_text records from the same admission
  - Remove empty texts and duplicate texts
  - Output a clean file for subsequent label merging and ToT usage
"""

import pandas as pd
from pathlib import Path


# Current script directory: data/data_clean_scripts/
BASE_DIR = Path(__file__).resolve().parent

# Original data directory: data/data_orig/
DATA_ORIG = BASE_DIR.parent / "data_orig"

# Cleaned output directory: data/data_cleaned/
DATA_CLEANED = BASE_DIR.parent / "data_cleaned"
DATA_CLEANED.mkdir(parents=True, exist_ok=True)
# ========== Path configuration (modify to your actual paths) ==========
DATA_PATH = DATA_ORIG / "thyroid_patients_with_notes.csv"
OUTPUT_PATH = DATA_CLEANED / "discharge_radiology_thyroid_cleaned.csv"
CHUNK_SIZE = 200_000  # Read 200k rows per chunk, suitable for large files
# ====================================================

print("🔹 Step 1: Reading CSV in chunks ...")
chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    # Normalize column names (case, whitespace)
    chunk.columns = [c.strip().lower() for c in chunk.columns]

    # Keep required columns
    keep_cols = [c for c in ["subject_id", "hadm_id", "discharge_text", "radiology_text"] if c in chunk.columns]
    chunk = chunk[keep_cols]

    # Drop rows with missing subject_id / hadm_id
    chunk = chunk.dropna(subset=["subject_id", "hadm_id"])

    # Remove rows with no meaningful text (empty or all NaN)
    mask_valid = chunk["discharge_text"].notna() | chunk["radiology_text"].notna()
    chunk = chunk.loc[mask_valid]

    # Remove very short texts (<30 characters, likely invalid templates)
    for col in ["discharge_text", "radiology_text"]:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna("").astype(str)
            chunk.loc[chunk[col].str.len() < 30, col] = ""

    chunks.append(chunk)

print("🔹 Step 2: Merging all chunks ...")
df = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory

print(f"Raw record count: {len(df):,}")

# =================== Step 3: Deduplication ===================
print("🔹 Step 3: Deduplicating multiple rows with the same subject_id / hadm_id ...")


# Aggregate multiple texts per (subject_id, hadm_id) and remove duplicates
def merge_texts(series):
    # Remove empty strings and duplicate content
    unique_texts = list(set([t.strip() for t in series.dropna() if t.strip()]))
    return "\n---\n".join(unique_texts) if unique_texts else ""


# Group by admission
df_grouped = (
    df.groupby(["subject_id", "hadm_id"], as_index=False)
    .agg({
        "discharge_text": merge_texts,
        "radiology_text": merge_texts
    })
)

print(f"Record count after aggregation: {len(df_grouped):,}")

# =================== Step 4: Final cleaning ===================
print("🔹 Step 4: Removing completely empty texts (no discharge and no radiology) ...")
df_grouped["discharge_text"] = df_grouped["discharge_text"].fillna("").astype(str)
df_grouped["radiology_text"] = df_grouped["radiology_text"].fillna("").astype(str)
df_grouped = df_grouped.loc[
    (df_grouped["discharge_text"].str.strip() != "") |
    (df_grouped["radiology_text"].str.strip() != "")
    ]

print(f"Record count after cleaning: {len(df_grouped):,}")

# =================== Step 5: Export ===================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_grouped.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Clean file saved: {OUTPUT_PATH}")
