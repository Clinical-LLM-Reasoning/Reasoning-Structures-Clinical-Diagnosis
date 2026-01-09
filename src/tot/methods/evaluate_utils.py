# evaluate_utils.py
from sklearn.metrics import classification_report
import os, json, pandas as pd
import numpy as np
from datetime import datetime

def evaluate(y_true, y_pred, model_name="default_model",
             buffer_path=None, method_name=None, metrics_path=None):
    """
    Print and save the classification report:

    - Save a CSV file (with timestamps) under `results/`

    - If `buffer_path` (i.e., the path to `*_predictions.json`) is provided, `*_metrics.json` will be automatically derived to store the metrics.

    Alternatively, you can explicitly pass in `metrics_path`.
    """
    # --- START: Filter out entries with a value of -1.---
    print(f"开始评估... 接收到 {len(y_pred)} 个原始预测。")
    
    # Using the NumPy method (recommended)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    mask = y_pred_np != -1
    
    y_true_filtered = y_true_np[mask]
    y_pred_filtered = y_pred_np[mask]
    
    # Check if there are any valid data points available for evaluation.
    if len(y_pred_filtered) == 0:
        print("[Warning] After filtering, no effective predictions are available for evaluation.")
        return None # Or return an empty report.

    num_filtered = len(y_pred) - len(y_pred_filtered)
    if num_filtered > 0:
        print(f"{num_filtered} unresolved predictions (value -1) have been ignored. The metric will be computed on {len(y_pred_filtered)} valid samples.")
    # --- END: Filtering ---

    # Forced type conversion to int prevents the mixing of "1" and 1.
    y_true_clean = y_true_filtered.astype(int)
    y_pred_clean = y_pred_filtered.astype(int)

    # Evaluation using filtered and cleaned data
    report = classification_report(y_true_clean, y_pred_clean, target_names=["0", "1"], output_dict=True)

    print("=== Overall Evaluation ===")
    print(classification_report(y_true_clean, y_pred_clean, target_names=["0", "1"]))

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{(model_name or 'model').replace('/', '_')}_{method_name or 'default'}_{timestamp}_classification_report.csv"
    csv_path = os.path.join("results", csv_name)
    pd.DataFrame(report).transpose().to_csv(csv_path, encoding="utf-8", index=True)
    print(f"[Saved] CSV report -> {csv_path}")

    # Compile a concise summary for easy program reading.
    summary = {
        "model": model_name,
        "method": method_name or "unknown",
        "accuracy": report["accuracy"],
        "macro": {
            "precision": report["macro avg"]["precision"],
            "recall":    report["macro avg"]["recall"],
            "f1":        report["macro avg"]["f1-score"],
        },
        "weighted": {
            "precision": report["weighted avg"]["precision"],
            "recall":    report["weighted avg"]["recall"],
            "f1":        report["weighted avg"]["f1-score"],
        },
        "per_class": {
            "0": report["0"],
            "1": report["1"],
        }
    }

    # Automatically determine metrics_path (with the same prefix as predictions).
    if metrics_path is None and buffer_path:
        if buffer_path.endswith("_predictions.json"):
            metrics_path = buffer_path.replace("_predictions.json", "_metrics.json")
        else:
            # Backup: Same directory, same name plus suffix
            base, ext = os.path.splitext(buffer_path)
            metrics_path = base + "_metrics.json"

    if metrics_path:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Saved] Metrics JSON -> {metrics_path}")

    return report, csv_path, metrics_path
