# strict_clean_llm_results.py
import pandas as pd, re, os
from pathlib import Path

# Paths: input comes from llm_interface output
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_PATH  = DATA_DIR / "llm_thyroid_summary_filtered_original.csv"
OUTPUT_PATH = DATA_DIR / "llm_thyroid_summary_strict_filtered_qwen3.csv"
PARSED_PATH = DATA_DIR / "llm_thyroid_summary_parsed.csv"

NO_INFO_PATTERNS = [
    "none", "no relevant", "no abnormal", "not specified",
    "no evidence", "unremarkable", "negative findings",
    "no thyroid abnormality", "no thyroid nodule", "no significant finding"
]
FIELDS = ["Symptoms", "Physical Findings", "Imaging Findings", "Treatment or Medication"]

def is_effective(text: str) -> bool:
    content = (text or "").lower().strip()
    if not content:
        return False
    return not any(p in content for p in NO_INFO_PATTERNS)

def grab_field(output: str, field: str) -> str:
    m = re.search(rf"### {re.escape(field)}:\s*(.*?)(?=\n### |\Z)", output or "", re.I | re.S)
    return (m.group(1).strip() if m else "") or ""

def clean_and_parse():
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}")
        return
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH.name}")

    keep_rows = []
    parsed_rows = []
    for _, row in df.iterrows():
        out = str(row.get("llm_output", "") or "")
        field_vals = {f: grab_field(out, f) for f in FIELDS}
        # Check whether at least one field contains effective information
        if any(is_effective(v) for v in field_vals.values()):
            keep_rows.append(row)
            parsed_rows.append({
                "uid": row.get("uid", ""),
                "subject_id": row.get("subject_id", ""),
                "hadm_id": row.get("hadm_id", ""),
                **{f.replace(" ", "_").lower(): field_vals[f] for f in FIELDS}
            })

    df_keep = pd.DataFrame(keep_rows).drop_duplicates(subset=["uid"])
    df_parsed = pd.DataFrame(parsed_rows).drop_duplicates(subset=["uid"])

    df_keep.to_csv(OUTPUT_PATH, index=False)
    df_parsed.to_csv(PARSED_PATH, index=False)
    print(f"Saved strict filtered -> {OUTPUT_PATH}")
    print(f"Saved parsed columns  -> {PARSED_PATH}")

if __name__ == "__main__":
    clean_and_parse()
