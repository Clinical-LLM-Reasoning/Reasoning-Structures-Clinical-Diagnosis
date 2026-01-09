## Environment Setup

This project uses `pyproject.toml` to define all dependencies.
We recommend using **uv** for fast and reproducible environment management.

### Setup with `uv` (Recommended)

Install `uv` following the official instructions:
https://github.com/astral-sh/uv

Then, from the project root, run:

```bash
uv sync
```
This command will:

- Create a virtual environment automatically

- Install all dependencies specified in `pyproject.toml`

- Ensure consistent environments across platforms

All commands in this README assume the environment has been set up using `uv`.

## Data Preparation (MIMIC-IV)

This repository does **not** distribute any clinical data.  
All data must be reconstructed locally from **MIMIC-IV** using SQL queries and the provided processing scripts.

The data preparation pipeline consists of three stages:
1. SQL-based extraction from MIMIC-IV
2. Clinical text cleaning and merging
3. (Optional) LLM-based information extraction

---

### MIMIC-IV Data Source

All clinical data used in this project are derived from **MIMIC-IV (version 3.1)**, a freely accessible,
de-identified electronic health record (EHR) dataset hosted on PhysioNet.

- **MIMIC-IV v3.1 dataset page**:  
  https://physionet.org/content/mimiciv/3.1/

You must download MIMIC-IV locally and load it into a database (e.g., PostgreSQL)
according to the official PhysioNet instructions before running any SQL queries.

---

### Citation Requirements

If you use this repository or reproduce the experiments, please cite the following resources as required by PhysioNet.

**Primary dataset citation (MIMIC-IV v3.1):**

Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024).  
*MIMIC-IV (version 3.1).* PhysioNet.  
RRID: SCR_007345.  
https://doi.org/10.13026/kpb9-mt58

**Original MIMIC-IV publication:**

Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023).  
*MIMIC-IV, a freely accessible electronic health record dataset.*  
Scientific Data, 10, 1.  
https://doi.org/10.1038/s41597-022-01899-x

**PhysioNet platform citation:**

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., et al. (2000).  
*PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.*  
Circulation, 101(23), e215–e220.  
RRID: SCR_007345

### MIMIC-IV-Note Data Source

Unstructured clinical text used in this project is derived from **MIMIC-IV-Note (version 2.2)**,
which contains deidentified free-text clinical notes linked to the MIMIC-IV database.

- **MIMIC-IV-Note v2.2 dataset page**:  
  https://www.physionet.org/content/mimic-iv-note/2.2/

This dataset includes discharge summaries and radiology reports used for clinical text
cleaning and optional LLM-based information extraction.

---

### Additional Citation for MIMIC-IV-Note

If you use the unstructured clinical notes or reproduce experiments involving clinical text,
please also cite the following resource:

Johnson, A., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).  
*MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2).*  
PhysioNet.  
RRID: SCR_007345.  
https://doi.org/10.13026/1n74-ne17

Please also include the standard PhysioNet citation:

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., et al. (2000).  
*PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.*  
Circulation, 101(23), e215–e220.  
RRID: SCR_007345


### 1. MIMIC-IV Databases Used

Data are extracted from the following MIMIC-IV schemas:

- `mimiciv_hosp`  
  Used for structured data:
  - Thyroid-related laboratory measurements (e.g., TSH, T3, T4, FT4, FTI, TPOAb)
  - Diagnosis codes (ICD-9 / ICD-10)
  - Patient and admission identifiers (`subject_id`, `hadm_id`)

- `mimiciv_note`  
  Used for unstructured clinical text:
  - Discharge summaries (`discharge`)
  - Radiology reports (`radiology`)

You must first download MIMIC-IV and load it into a database
(e.g., PostgreSQL) following the official PhysioNet instructions.

---

### 2. SQL-Based Data Extraction

After setting up the MIMIC-IV database, extract the required records using SQL.

At a minimum, export the following datasets as CSV files:

#### Structured data (`mimiciv_hosp`)
- Thyroid-related laboratory records  
  (filtered by ITEMIDs corresponding to TSH, T3, T4, FT4, FTI, TPOAb, etc.)
- Diagnosis records containing thyroid-related ICD-9 / ICD-10 codes
- Subject-level and admission-level identifiers

Example outputs:
- `thyroid_labs_full_1.csv`
- `thyroid_diagnosis_1.csv`

#### Unstructured text (`mimiciv_note`)
- Discharge summaries
- Radiology reports

Each exported record should contain:
- `subject_id`
- `hadm_id`
- full note text

You must export **two patient groups**:
1. Thyroid-related patients
2. Random non-thyroid control patients

SQL query files are not included in this repository due to data use restrictions.
All downstream scripts assume the standard MIMIC-IV table schemas.

---

### 3. Place Raw CSV Files

After exporting from the database, place all raw CSV files into:

```text
src/tot/data/thyroid/data_orig/
```

---

### 4. Clinical Text Cleaning and Merging
Run the following scripts:
```
python discharge_radiology_thyroid_cleaned.py
python discharge_radiology_random_cleaned.py
```

These scripts perform:

Deduplication of repeated notes

- Removal of extremely short or template-only text

- Merging discharge summaries and radiology reports

- Separation of thyroid-related and random control cohorts

Cleaned outputs are written to:
```
src/tot/data/thyroid/data_cleaned/
```
### 5. (Optional) LLM-Based Information Extraction

To convert unstructured clinical text into structured summaries, navigate to:
```
src/tot/data/thyroid/info_extraction_scripts/
```
Run:
```
python llm_interface.py          # Thyroid-related patients
python llm_interface_random.py   # Random control patients
```
Optional strict filtering of low-information outputs:
```
python strict_clean_llm_results.py
```
This step:

- Extracts thyroid-relevant information only

- Produces structured summaries (symptoms, imaging, medication, etc.)

- Stores results as CSV files under data_orig/

**Note**

This step requires a working LLM backend (local vLLM / Ollama / API).
It can be skipped if you want to reproduce lab-only experiments.

### 6. Output of the Data Preparation Stage

After completing the steps above, the following artifacts will be available:

- Raw structured laboratory data (CSV)

- Cleaned and merged clinical text

- (Optional) LLM-generated structured summaries

These outputs are consumed by the dataset construction scripts
in the next stage of the pipeline.

### 7. Dataset Construction

After completing the data preparation steps (Sections 1–6), the cleaned data can be assembled into
evaluation-ready datasets using the scripts in:
```
src/tot/data/thyroid/dataset_building/
```

Run the following scripts as needed:
```
python balanced_500_with_text_cleaned.py
python full_with_text_cleaned.py
python build_small_eval_dataset.py
```

These scripts construct different datasets for different experimental purposes:

- Balanced 500-patient dataset
A balanced dataset with equal numbers of thyroid-positive and thyroid-negative patients.
This dataset is used for controlled quantitative comparison across reasoning methods.

- Full cohort dataset
Includes all eligible patients without sampling.
This dataset is used to study scalability and overall performance trends.

- Small evaluation dataset
A small, difficulty-controlled subset designed for qualitative analysis and human expert evaluation.

Across all datasets:

- Labels are assigned at the patient level

- All available laboratory sessions are preserved

- Each patient is associated with at most one text summary

The generated datasets are written to the `data_cleaned/` directory and are used as inputs
for all subsequent reasoning experiments.

### 8. Running Reasoning Methods

All reasoning methods are implemented under:
```
src/tot/
```

Each method exposes a unified `solve()` interface and produces predictions in the same format,
enabling direct comparison.

Available reasoning methods:

- Pure LLM baseline — `pure_llm.py`

- Chain-of-Thought (CoT) — `cot.py`

- Tree-of-Thoughts (ToT) — `bfs.py`

- Buffer-of-Thoughts (BoT) — `bot.py`

- - Template-only decision tree — `dtree.py`

All methods support:

- Running with or without clinical text (`use_text = True / False`)

- Identical input format

- Identical output format (`Final: 0` or `Final: 1`)

Predictions are typically saved as JSONL files (e.g., `*_predictions.json`) for later evaluation.

### 9. Evaluation

Evaluation utilities are provided in:
```
src/tot/evaluation/
```
**Standard evaluation**

Run:
```
python evaluate_utils.py
```

This script computes:

- Accuracy

- Precision / Recall

- Macro F1 score

Evaluation results are printed to the console and saved as CSV and JSON files for record keeping.

**Template-only decision tree evaluation**

For the template-only baseline (`dtree.py`), run:
```
python evaluate_dtree.py
```

In this evaluation:

- Unclassified predictions (`-1`) are treated as errors

- Errors are further decomposed into false positives and false negatives

This provides a strict baseline for comparison with LLM-based reasoning methods.

### 10. LLM Backend Configuration

All LLM calls are routed through a unified interface:
```
src/tot/models.py
src/tot/llm_call_api.py
```

Supported backends include:

- Local vLLM servers

- Ollama

- OpenAI-compatible APIs (e.g., OpenAI, Qwen, VAPI)

The backend provider and model name can be configured directly in `models.py.`

Ensure the selected backend is running and accessible before launching any LLM-based scripts.


### API Key Configuration

If you use a remote LLM backend (e.g., OpenAI-compatible APIs),
you must configure **your own API key** in the following file:

- `src/tot/llm_call_api.py`

**Important**
The API key is **not provided** with this repository and must be supplied by the user.
It must **not** be committed to version control.

### 11. Reproducibility Notes

- Processed MIMIC-IV data are not redistributed due to data use restrictions

- Random sampling uses fixed seeds where applicable

- Template-based reasoning logic is fully deterministic

- LLM-based components may introduce minor nondeterminism

- All scripts are designed to be run independently and logged for reproducibility

### 12. Disclaimer

This repository is intended **for research and educational purposes only.**
It does **not** provide clinical decision support and must not be used in real-world medical practice.