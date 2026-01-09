# llm_interface_random.py
# -*- coding: utf-8 -*-
"""
Batch extractor (Ollama) for RANDOM non-thyroid patients.

- Model: qwen3:8b
- Expected sections:
  ### Symptoms:
  ### Physical Findings:
  ### Imaging Findings:
  ### Treatment or Medication:
- Input : discharge_radiology_random_cleaned.csv
- Output: llm_random_summary_filtered_original.csv
- No keyword filtering (process all records)
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
from concurrent.futures import ThreadPoolExecutor, as_completed


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


# ====== Basic configuration ======
MODEL_NAME = "qwen3:8b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent.parent / "data"

# Input file (cleaned random patients)
CSV_PATH = DATA_ROOT / "data_cleaned" / "discharge_radiology_random_cleaned.csv"

# Output directory
OUTPUT_DIR = DATA_ROOT / "data_orig"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUTPUT_DIR / "llm_random_summary_filtered_original.csv"
SKIP_LOG_PATH = OUTPUT_DIR / "llm_random_summary_skipped.csv"

print(f"[INFO] INPUT : {CSV_PATH}")
print(f"[INFO] OUTPUT DIR: {OUTPUT_DIR}")

# Prompt template
TEMPLATE_PATH = str(DATA_ROOT / "prompts" / "extract_thyroid_related.txt")

NUM_CTX = 8192
PRINT_THINK = True
PRINT_OUTPUT = True

SECTIONS = [
    "Symptoms",
    "Physical Findings",
    "Imaging Findings",
    "Treatment or Medication",
]

DISCHARGE_CANDS = ["discharge_summary", "discharge_text", "note_text", "discharge"]
RADIOLOGY_CANDS = ["radiology_report", "radiology_text", "imaging_report", "radiology"]
ID_CANDS = ["subject_id", "hadm_id", "note_id", "stay_id", "charttime"]


# ====== Helpers ======
def _load_uids_from_csv(path: str) -> set:
    s = set()
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("uid"):
                    s.add(r["uid"])
    return s


def _ensure_skip_writer(path: str):
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
    return (
        "You are a clinical assistant. Extract any clinical information.\n"
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
    return (
        template.replace("{{DISCHARGE_TEXT}}", discharge_text or "")
                .replace("{{RADIOLOGY_TEXT}}", radiology_text or "")
    )


def query_llm(prompt: str, model: str = MODEL_NAME, endpoint: str = OLLAMA_ENDPOINT, sess: requests.Session = None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }
    s = sess or requests
    r = s.post(endpoint, json=payload, timeout=600)
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
    raise ValueError(f"[Column not found] Need column for {logical_name}. Tried: {cands}. Existing: {list(df.columns)}")


# ====== Main ======
def process_records(csv_path: str, limit: int = None):
    csv_path = str(csv_path)
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
    total = len(df)

    discharge_col = _pick_col(df, DISCHARGE_CANDS, "discharge")
    radiology_col = _pick_col(df, RADIOLOGY_CANDS, "radiology")
    print(f"[DEBUG] Using discharge_col = '{discharge_col}', radiology_col = '{radiology_col}'")

    print(f"[FILTER] Random patients — no keyword filter applied ({total:,} records kept)")

    out_path = str(OUT_PATH)
    processed = _load_processed(out_path) | _load_uids_from_csv(str(SKIP_LOG_PATH))
    f, writer = _ensure_writer(out_path)
    skip_f, skip_writer = _ensure_skip_writer(str(SKIP_LOG_PATH))

    print(f"[INFO] Output file: {out_path}")
    print(f"[INFO] Skip file: {SKIP_LOG_PATH}")
    print(f"[RESUME] Already processed: {len(processed):,}")
    print(f"[INFO] Remaining: {total - len(processed):,}")

    sess = requests.Session()
    done = skipped = 0
    started = time.time()

    def process_one(idx, row):
        rowd = row.to_dict()
        uid = _get_uid(rowd, idx)
        if uid in processed:
            return {"uid": uid, "status": "skipped", "reason": "already_processed"}

        discharge_text = str(rowd.get(discharge_col, "") or "")
        radiology_text = str(rowd.get(radiology_col, "") or "")
        if not discharge_text.strip() and not radiology_text.strip():
            return {"uid": uid, "status": "skipped", "reason": "empty_input"}

        prompt = build_prompt(discharge_text, radiology_text)
        tries = 0
        while True:
            try:
                t0 = time.time()
                raw = query_llm(prompt, sess=sess)
                break
            except Exception as e:
                tries += 1
                if tries >= 3:
                    return {"uid": uid, "status": "error", "reason": str(e)}
                time.sleep(1.5 * tries)

        think_text = extract_think(raw)
        result_text = remove_think(raw)
        if (not result_text.strip()) or is_all_none(result_text):
            return {"uid": uid, "status": "skipped", "reason": "all_none"}

        return {
            "uid": uid,
            "status": "done",
            "llm_thinking": think_text,
            "llm_output": result_text,
        }

    with ThreadPoolExecutor(max_workers=1) as executor:
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
            print(f"[PROGRESS] {done} done | {skipped} skipped | elapsed={elapsed:.1f}s | last uid={uid}")

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
    args = parser.parse_args()
    process_records(args.csv, limit=args.limit)
