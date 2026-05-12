"""
Runner script for the Stretch Tuesday calibration assignment.
Loads the fine-tuned model, runs manual evaluation, and generates
the reliability diagram + ECE.
"""

import json
import os

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from manual_eval import manual_predict, compute_classification_report_from_arrays
from calibration import reliability_diagram, expected_calibration_error, plot_reliability


def main():
    model_dir = "model"
    data_path = os.environ.get("DATA_PATH", "data/app_reviews_train.csv")

    # ---------- Load model & tokenizer ----------
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    id2label = model.config.id2label

    # ---------- Prepare test split (same as lab.py: 80/20, seed=42) ----------
    print("Preparing test split...")
    from datasets import Dataset
    df = pd.read_csv(data_path)
    ds = Dataset.from_pandas(df, preserve_index=False)
    splits = ds.train_test_split(test_size=0.2, seed=42)
    test_texts = splits["test"]["text"]
    test_labels = np.array(splits["test"]["label"])

    # ---------- Manual predict ----------
    print(f"Running manual inference on {len(test_texts)} test examples...")
    preds, probs = manual_predict(model, tokenizer, test_texts, batch_size=16)

    # ---------- Classification report ----------
    report = compute_classification_report_from_arrays(test_labels, preds)
    print("\n=== Classification Report (manual) ===")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro-F1: {report['macro_f1']:.4f}")
    for cls_idx, metrics in sorted(report["per_class"].items()):
        label_name = id2label.get(str(cls_idx), id2label.get(cls_idx, str(cls_idx)))
        print(f"  {label_name}: P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

    # ---------- Reliability diagram ----------
    print("\n=== Calibration Analysis ===")
    centers, accs, counts = reliability_diagram(probs, test_labels, n_bins=10)

    print("\nBucket details:")
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        print(f"  [{edges[i]:.1f}–{edges[i+1]:.1f}] center={centers[i]:.2f}  "
              f"acc={accs[i]:.4f}  count={counts[i]}")

    # ---------- ECE ----------
    ece = expected_calibration_error(probs, test_labels, n_bins=10)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

    # ---------- Save diagram ----------
    os.makedirs("figures", exist_ok=True)
    plot_reliability(centers, accs, counts, "figures/reliability-diagram.png")
    print("\nSaved figures/reliability-diagram.png")

    # ---------- Save results for reference ----------
    results = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "ece": ece,
        "per_class": report["per_class"],
        "buckets": {
            f"{edges[i]:.1f}-{edges[i+1]:.1f}": {
                "center": float(centers[i]),
                "accuracy": float(accs[i]),
                "count": int(counts[i]),
            }
            for i in range(10)
        },
    }
    with open("stretch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved stretch_results.json")


if __name__ == "__main__":
    main()
