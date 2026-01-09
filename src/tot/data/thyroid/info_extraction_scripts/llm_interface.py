# llm_interface.py
# -*- coding: utf-8 -*-
"""
Batch extractor (Ollama) with checkpoint for thyroid-related info.

- Model: qwen3:8b (change MODEL_NAME if needed)
- Expected sections (exact, no 'Diagnosis'):
  ### Symptoms:
  ### Physical Findings:
  ### Imaging Findings:
  ### Treatment or Medication:
- Writes data/llm_thyroid_summary_filtered_original.csv
  and skips rows where all 4 sections == "None"
- Checkpoint: resume by uid
"""

import os
import re
import csv
import time
import json
import argparse
import requests
import pandas as pd
from pathlib import Path

# ====== GPU detection ======
def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[INFO] GPU available: {name} ({mem:.1f} GB)")
        else:
            print("[INFO] GPU not detected — Ollama may be running on CPU mode.")
    except Exception:
        print("[INFO] torch not installed — skipping GPU detection.")

# ====== YOU MAY CHANGE THESE ======
MODEL_NAME       = "qwen3:8b"
OLLAMA_ENDPOINT  = "http://localhost:11434/api/generate"
# OLLAMA_ENDPOINT  = "http://frp-sea.com:11434/api/generate"

# Input file: aligned with your cleaned file
# ===== Corrected path configuration =====
from pathlib import Path

# Current script directory, e.g.: .../experiments/data/info_extraction_scripts/
BASE_DIR = Path(__file__).resolve().parent
# Go back to the project data root (two levels up: info_extraction_scripts → data)
DATA_ROOT = BASE_DIR.parent.parent / "data"
# ===== Used to identify patients to be processed =====

# Input file: your cleaned merged text
CSV_PATH = DATA_ROOT / "data_cleaned" / "discharge_radiology_thyroid_cleaned.csv"
# Output directory: results will be written under data/
OUTPUT_DIR = DATA_ROOT / "data_orig"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SKIP_LOG_PATH = OUTPUT_DIR / "llm_thyroid_summary_skipped.csv"



print(f"[INFO] INPUT : {CSV_PATH}")
print(f"[INFO] OUTPUT DIR: {OUTPUT_DIR}")
# =============================

# Prompt template (4 sections, no Diagnosis)
TEMPLATE_PATH = str(DATA_ROOT / "prompts" / "extract_thyroid_related.txt")


# Context and truncation
NUM_CTX = 8192
# —— Only affects console printing, not file writing ——
PRINT_THINK = True          # Whether to print <think> content
PRINT_OUTPUT = True         # Whether to print cleaned output
PRINT_MAX_CHARS = None      # Max chars to print in console to avoid flooding; None = no truncation

MAX_CHARS_PER_DOC = None

SECTIONS = [
    "Symptoms",
    "Physical Findings",
    "Imaging Findings",
    "Treatment or Medication",
]

DISCHARGE_CANDS = ["discharge_summary", "discharge_text", "note_text", "discharge"]
RADIOLOGY_CANDS = ["radiology_report", "radiology_text", "imaging_report", "radiology"]
ID_CANDS        = ["subject_id", "hadm_id", "note_id", "stay_id", "charttime"]


# ====== Helpers ======
def _load_uids_from_csv(path: str) -> set:
    """
    Load uid column (if exists) from any CSV and return as a set.
    Return an empty set if the file does not exist.
    """
    s = set()
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("uid"):
                    s.add(r["uid"])
    return s

def _ensure_skip_writer(path: str):
    """
    Prepare a writer for skipped records (columns: uid, subject_id, hadm_id, reason)
    """
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=["uid", "subject_id", "hadm_id", "reason"])
    if not exists:
        w.writeheader()
        f.flush()
    return f, w

def _read_template() -> str:
    p = Path(TEMPLATE_PATH)
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Fallback safe template
    return (
        "You are a clinical assistant. Extract ONLY thyroid-related information.\n"
        "Return EXACTLY these sections:\n"
        "### Symptoms:\n"
        "### Physical Findings:\n"
        "### Imaging Findings:\n"
        "### Treatment or Medication:\n\n"
        "[Discharge Summary]\n{{DISCHARGE_TEXT}}\n\n"
        "[Radiology Report]\n{{RADIOLOGY_TEXT}}\n"
        "If no info, write 'None' under each section."
    )

def build_prompt(discharge_text: str, radiology_text: str) -> str:
    template = _read_template()
    return (template
            .replace("{{DISCHARGE_TEXT}}", discharge_text or "")
            .replace("{{RADIOLOGY_TEXT}}", radiology_text or ""))

def query_llm(prompt: str, model: str = MODEL_NAME, endpoint: str = OLLAMA_ENDPOINT, sess: requests.Session = None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",  # Keep the model resident to avoid reloading for each request
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0.2,
            "top_p": 0.9,
            # "num_predict": -1,  # Limit generation length for speed and stability
        },
    }
    s = sess or requests
    r = s.post(endpoint, json=payload, timeout=600)  # Increase timeout from 120s to 600s
    r.raise_for_status()
    data = r.json()
    return data.get("response", "") or data.get("message", {}).get("content", "")


def extract_think(text: str) -> str:
    m = re.search(r"<think>\s*(.*?)\s*</think>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def remove_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL | re.IGNORECASE)

def is_all_none(output_text: str) -> bool:
    if not output_text.strip():
        return True
    for name in SECTIONS:
        # Allow variable spaces/colon after ###
        pat = rf"###\s*{re.escape(name)}\s*:\s*(.*?)(?=\n###\s*|$)"
        m = re.search(pat, output_text or "", flags=re.DOTALL | re.IGNORECASE)
        chunk = (m.group(1).strip() if m else "")
        if chunk and chunk.lower() != "none":
            return False
    return True

def _get_uid(row: dict, idx: int) -> str:
    parts = [str(row.get(k, "")).strip() for k in ID_CANDS if str(row.get(k, "")).strip()]
    return "||".join(parts) if parts else f"idx-{idx}"

def _load_processed(out_csv: str) -> set:
    s = set()
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("uid"):
                    s.add(r["uid"])
    return s

def _ensure_writer(path: str):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=["uid", "subject_id", "hadm_id", "llm_thinking", "llm_output"])
    if not exists:
        w.writeheader()
        f.flush()
    return f, w

def _pick_col(df: pd.DataFrame, cands: list, logical_name: str) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise ValueError(f"[Column not found] Need column for {logical_name}. "
                     f"Tried: {cands}. Existing: {list(df.columns)}")


# ====== Main ======
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def process_records(csv_path: str, limit: int = None, skip: int = 0, sleep_s: float = 0.1):
    csv_path = str(csv_path)
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return

    # ===== Step 1: Read CSV =====
    df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
    total = len(df)

    # ===== Step 2: Auto-detect column names =====
    discharge_col = _pick_col(df, DISCHARGE_CANDS, "discharge")
    radiology_col = _pick_col(df, RADIOLOGY_CANDS, "radiology")
    print(f"[DEBUG] Using discharge_col = '{discharge_col}', radiology_col = '{radiology_col}'")

    # ===== Step 3: Keyword filtering (stricter) =====
    THYROID_KEYWORDS = [
        "thyroid", "t3", "t4", "tsh",
        "hypothyroid", "hyperthyroid", "thyrotoxic", "thyrotoxicosis",
        "graves", "hashimoto",
        "levothyroxine", "synthroid", "methimazole", "propylthiouracil", "ptu",
        "goiter", "thyroidectomy", "thyroid nodule", "thyroid mass",
        "papillary", "follicular", "medullary", "anaplastic",
        "thyroid cancer", "thyroid carcinoma"
    ]

    def contains_keywords(row):
        combined = (str(row.get(discharge_col, '')) + ' ' +
                    str(row.get(radiology_col, ''))).lower()
        combined = re.sub(r"[^a-z0-9\s]", " ", combined)
        for kw in THYROID_KEYWORDS:
            pattern = rf"\b{kw}s?\b"
            if re.search(pattern, combined):
                return True
        return False

    before_filter = len(df)
    df = df[df.apply(contains_keywords, axis=1)].reset_index(drop=True)
    after_filter = len(df)
    print(f"[FILTER] Filtered records: {before_filter:,} → {after_filter:,} (kept {after_filter / before_filter:.2%})")

    # ===== Step 4: Output files and checkpoint =====
    out_path = str(OUTPUT_DIR / "llm_thyroid_summary_filtered_original.csv")
    # out_path = str(OUTPUT_DIR / "llm_random_summary_filtered_original.csv")

    processed = set()
    processed |= _load_processed(out_path)
    processed |= _load_uids_from_csv(str(SKIP_LOG_PATH))
    f, writer = _ensure_writer(out_path)
    skip_f, skip_writer = _ensure_skip_writer(str(SKIP_LOG_PATH))

    print(f"[INFO] Output file will be saved to: {out_path}")
    print(f"[INFO] Skip file will be saved to: {SKIP_LOG_PATH}")
    print(f"[RESUME] Already processed (CSV+SKIP): {len(processed):,}")
    print(f"[INFO] Remaining: {total - len(processed):,}")

    # ===== Step 5: Parallel configuration =====
    MAX_WORKERS = 1
    print(f"[INFO] Parallel mode: {MAX_WORKERS} workers (ThreadPoolExecutor)")
    sess = requests.Session()
    done = skipped = 0
    started = time.time()

    def process_one(idx, row):
        rowd = row.to_dict()
        uid = _get_uid(rowd, idx)
        result = {"uid": uid, "status": "", "reason": "", "llm_output": "", "llm_thinking": ""}

        if uid in processed:
            result["status"] = "skipped"
            result["reason"] = "already_processed"
            return result

        discharge_text = str(rowd.get(discharge_col, "") or "")
        radiology_text = str(rowd.get(radiology_col, "") or "")
        if not discharge_text.strip() and not radiology_text.strip():
            result["status"] = "skipped"
            result["reason"] = "empty_input"
            return result

        prompt = build_prompt(discharge_text, radiology_text)

        tries = 0
        while True:
            try:
                t0 = time.time()
                raw = query_llm(prompt, sess=sess)
                result["llm_time"] = round(time.time() - t0, 2)
                break
            except Exception as e:
                tries += 1
                if tries >= 3:
                    result["status"] = "error"
                    result["reason"] = str(e)
                    return result
                time.sleep(1.5 * tries)

        think_text = extract_think(raw)
        result_text = remove_think(raw)

        if PRINT_THINK:
            print(f"\n[THINK {idx + 1}/{total} | uid={uid}]\n{think_text[:800]}\n" + "-" * 60)
        if PRINT_OUTPUT:
            print(f"[OUTPUT {idx + 1}/{total} | uid={uid}]\n{result_text[:800]}\n" + "=" * 60)

        if (not result_text.strip()) or is_all_none(result_text):
            result["status"] = "skipped"
            result["reason"] = "all_none"
        else:
            result["status"] = "done"
            result["llm_output"] = result_text
            result["llm_thinking"] = think_text
        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, idx, row): (idx, row) for idx, row in df.iterrows()}
        for future in as_completed(futures):
            idx, row = futures[future]
            res = future.result()
            uid = res["uid"]

            if res["status"] == "done":
                writer.writerow({
                    "uid": uid,
                    "subject_id": row.get("subject_id", ""),
                    "hadm_id": row.get("hadm_id", ""),
                    "llm_thinking": res["llm_thinking"],
                    "llm_output": res["llm_output"],
                })
                f.flush()
                done += 1
            else:
                skip_writer.writerow({
                    "uid": uid,
                    "subject_id": row.get("subject_id", ""),
                    "hadm_id": row.get("hadm_id", ""),
                    "reason": res["reason"],
                })
                skip_f.flush()
                skipped += 1

            processed.add(uid)
            elapsed = round(time.time() - started, 1)
            print(f"[PROGRESS] {done} done | {skipped} skipped | elapsed={elapsed:.1f}s | last uid={uid} | LLM={res.get('llm_time', 0)}s")

            if limit and done >= limit:
                break

    f.close()
    skip_f.close()
    elapsed = round(time.time() - started, 1)
    print(f"\n[OK] saved to: {out_path} | done={done}, skipped={skipped}, elapsed={elapsed}s")


if __name__ == "__main__":
    check_gpu()
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to input CSV")
    parser.add_argument("--limit", type=int, default=None, help="limit processed rows")
    parser.add_argument("--skip", type=int, default=0, help="skip N rows first")
    parser.add_argument("--sleep", type=float, default=0.01, help="sleep seconds between requests")
    args = parser.parse_args()
    process_records(args.csv, limit=args.limit, skip=args.skip, sleep_s=args.sleep)
