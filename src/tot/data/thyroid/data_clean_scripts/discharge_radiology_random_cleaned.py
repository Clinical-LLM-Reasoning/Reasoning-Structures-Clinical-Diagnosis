# -*- coding: utf-8 -*-
"""
Deduplication and cleaning: random non-tested patients' discharge summaries + radiology reports
Objectives:
  - Input can be either a WIDE table (discharge_text, radiology_text)
    or a LONG table (text, note_source)
  - Merge multiple texts per (subject_id, hadm_id), deduplicate, and apply minimum length filtering
  - By default, keep only admissions that have BOTH discharge_text and radiology_text
  - Output: data_cleaned/discharge_radiology_random_cleaned.csv (wide table)
"""

import pandas as pd
from pathlib import Path

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ORIG = BASE_DIR.parent / "data_orig"
DATA_CLEANED = BASE_DIR.parent / "data_cleaned"
DATA_CLEANED.mkdir(parents=True, exist_ok=True)

# Raw file exported from SQL (can be long or wide format)
DATA_PATH = DATA_ORIG / "random_patients_with_notes.csv"

# Output file (wide format)
OUTPUT_PATH = DATA_CLEANED / "discharge_radiology_random_cleaned.csv"

CHUNK_SIZE = None   # Random ~2000 rows usually do not require chunking; set to 200_000 if very large
MIN_LEN = 30        # Minimum text length threshold (short templates are treated as invalid)
REQUIRE_BOTH = True # True: keep only admissions with both discharge & radiology; False: allow missing one

# ------------- Utility functions -------------
def merge_texts(series):
    """Deduplicate and merge multiple texts under the same (sid, hadm_id)."""
    texts = [str(t).strip() for t in series.dropna()]
    texts = [t for t in texts if len(t) >= MIN_LEN]
    unique = list(dict.fromkeys(texts))  # deduplicate while preserving order
    return "\n---\n".join(unique) if unique else ""

def _normalize_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    # Enforce required ID columns
    must = {"subject_id", "hadm_id"}
    if not must.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns: {must - set(df.columns)}; existing columns: {list(df.columns)}")
    return df

print("step 1: loading CSV ...")
if CHUNK_SIZE:
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        chunks.append(chunk)
    raw = pd.concat(chunks, ignore_index=True)
else:
    raw = pd.read_csv(DATA_PATH, low_memory=False)

df = _normalize_cols(raw)
del raw
print(f"Raw record count: {len(df):,} | Columns: {list(df.columns)}")

# ------------- Handle long / wide formats -------------
has_wide = ("discharge_text" in df.columns) or ("radiology_text" in df.columns)
has_long = ("text" in df.columns) and ("note_source" in df.columns)

if has_wide and has_long:
    # If both formats are present, prefer the wide format
    df = df[["subject_id", "hadm_id"] +
            [c for c in ["discharge_text", "radiology_text"] if c in df.columns]]

elif has_long and not has_wide:
    # Pivot long format to wide format
    # Keep only the two note types of interest
    df = df[df["note_source"].isin(["discharge", "radiology"])].copy()

    # Aggregate and merge texts by (sid, hadm_id, note_source)
    long_agg = (
        df.groupby(["subject_id", "hadm_id", "note_source"], as_index=False)
          .agg(text=("text", merge_texts))
    )

    # Pivot to wide format
    wide = long_agg.pivot(index=["subject_id", "hadm_id"],
                          columns="note_source",
                          values="text").reset_index()
    # Standardize column names
    wide = wide.rename(columns={
        "discharge": "discharge_text",
        "radiology": "radiology_text"
    })
    df = wide

elif has_wide:
    # Ensure missing columns exist
    if "discharge_text" not in df.columns:
        df["discharge_text"] = ""
    if "radiology_text" not in df.columns:
        df["radiology_text"] = ""
    df = df[["subject_id", "hadm_id", "discharge_text", "radiology_text"]]

else:
    raise ValueError(
        "Input contains neither wide-format columns (discharge_text/radiology_text) "
        "nor long-format columns (text/note_source)."
    )

# ------------- Cleaning and merging -------------
# Aggregate again by (subject_id, hadm_id) to handle multiple rows even in wide input
df_clean = (
    df.groupby(["subject_id", "hadm_id"], as_index=False)
      .agg({
          "discharge_text": merge_texts,
          "radiology_text": merge_texts
      })
)

# Remove rows where both text fields are empty
mask_any = (df_clean["discharge_text"].str.strip() != "") | (df_clean["radiology_text"].str.strip() != "")
df_clean = df_clean.loc[mask_any].copy()

# If both texts are required, apply additional filtering
if REQUIRE_BOTH:
    mask_both = (df_clean["discharge_text"].str.strip() != "") & (df_clean["radiology_text"].str.strip() != "")
    df_clean = df_clean.loc[mask_both].copy()

print(f"Records after cleaning: {len(df_clean):,}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}")
