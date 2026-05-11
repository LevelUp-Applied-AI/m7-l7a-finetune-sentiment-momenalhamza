"""
Lab 7A autograder.

Runs against the learner's repo root after `python lab.py` has executed
end-to-end on the smoke fixture (see workflow YAML).
"""
import ast
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# Drill-style mechanical checks (functions + signatures)
# ---------------------------------------------------------------------------

def test_prepare_dataset_returns_dict_with_train_and_test():
    import lab
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    assert "train" in ds and "test" in ds


def test_prepare_dataset_split_sizes():
    import lab
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    total = len(ds["train"]) + len(ds["test"])
    assert abs(len(ds["test"]) - total * 0.2) <= 2


def test_tokenize_dataset_columns():
    import lab
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    tokenized = lab.tokenize_dataset(ds, tokenizer, max_length=64)
    for split in ("train", "test"):
        cols = tokenized[split].column_names
        assert "input_ids" in cols and "attention_mask" in cols


def test_tokenize_dataset_max_length_truncates():
    import lab
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = lab.prepare_dataset(os.path.join(REPO_ROOT, "fixtures", "tiny_app_reviews.csv"), test_size=0.2, seed=42)
    tokenized = lab.tokenize_dataset(ds, tokenizer, max_length=16)
    max_seen = max(len(x) for x in tokenized["train"]["input_ids"])
    assert max_seen <= 16


def test_make_training_args_attributes():
    import lab
    args = lab.make_training_args("model", lr=3e-5, epochs=1, batch_size=4, seed=7)
    assert args.learning_rate == 3e-5
    assert args.num_train_epochs == 1
    assert args.per_device_train_batch_size == 4
    assert args.seed == 7
    # Per the lab guide, set evaluation/save cadence to once per epoch and
    # logging cadence to every ~50 steps. The eval_strategy attribute is named
    # eval_strategy (not evaluation_strategy) in transformers>=4.41 — the
    # course pins that range in requirements.txt.
    assert str(args.eval_strategy) == "epoch", \
        f"eval_strategy must be 'epoch' (got {args.eval_strategy!r})"
    assert str(args.save_strategy) == "epoch", \
        f"save_strategy must be 'epoch' (got {args.save_strategy!r})"
    assert args.logging_steps == 50, \
        f"logging_steps must be 50 (got {args.logging_steps})"


def test_compute_metrics_shape():
    import lab
    logits = np.array([[0.1, 0.7, 0.2], [0.6, 0.2, 0.2]])
    labels = np.array([1, 0])
    result = lab.compute_metrics((logits, labels))
    assert "accuracy" in result and "macro_f1" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_compute_metrics_perfect_predictions():
    import lab
    logits = np.array([[0.1, 0.9, 0.0], [0.9, 0.05, 0.05], [0.0, 0.1, 0.9]])
    labels = np.array([1, 0, 2])
    result = lab.compute_metrics((logits, labels))
    assert abs(result["accuracy"] - 1.0) < 1e-9
    assert abs(result["macro_f1"] - 1.0) < 1e-9


def test_compute_metrics_known_confusion():
    """Hand-computed: 4 samples, 2 correct → accuracy 0.5; per-class F1 known."""
    import lab
    # true: [0, 0, 1, 1]; preds: [0, 1, 0, 1]
    # class 0: P=1/2, R=1/2, F1=0.5; class 1: P=1/2, R=1/2, F1=0.5; macro=0.5
    logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    result = lab.compute_metrics((logits, labels))
    assert abs(result["accuracy"] - 0.5) < 1e-9
    assert abs(result["macro_f1"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end pipeline checks (CI runs python lab.py first)
# ---------------------------------------------------------------------------

def test_metrics_json_exists_and_well_formed():
    path = os.path.join(REPO_ROOT, "metrics.json")
    assert os.path.isfile(path), "metrics.json must be produced by lab.py"
    with open(path) as f:
        m = json.load(f)
    assert "accuracy" in m and "macro_f1" in m and "per_class_f1" in m


def test_train_classifier_smoke_accuracy():
    """Liveness check on the smoke fixture training run.

    The smoke fixture is small (60 rows total, 12-row test split). A 0.6
    threshold needs 8/12 correct, so a single prediction flip is an 8-pt
    swing — flaky across initializations. We use 0.4 as a pure liveness
    floor: training loop wired correctly produces well above 0.4 reliably,
    and noise alone (3-class random) sits at ~0.33.
    """
    path = os.path.join(REPO_ROOT, "metrics.json")
    with open(path) as f:
        m = json.load(f)
    assert m["accuracy"] >= 0.4, (
        f"Smoke accuracy {m['accuracy']:.3f} below liveness threshold 0.4 — "
        "training loop may be broken (e.g., train_dataset not passed to Trainer)"
    )


def test_predictions_csv_has_required_columns():
    path = os.path.join(REPO_ROOT, "predictions.csv")
    assert os.path.isfile(path), "predictions.csv must be produced by lab.py"
    df = pd.read_csv(path)
    for col in ["text", "label", "predicted_label", "predicted_probability"]:
        assert col in df.columns, f"missing column: {col}"
    assert df["predicted_probability"].between(0.0, 1.0).all()


def test_train_classifier_produces_local_checkpoint():
    """After main() runs in CI, model/ dir must exist with weights + config."""
    model_dir = os.path.join(REPO_ROOT, "model")
    assert os.path.isdir(model_dir), "model/ should exist locally after training"
    files = os.listdir(model_dir)
    weights = any(f in {"pytorch_model.bin", "model.safetensors"} for f in files)
    assert weights, "model/ must contain pytorch_model.bin or model.safetensors"
    assert "config.json" in files


def test_main_produces_committed_artifacts():
    """metrics.json + predictions.csv must exist (committed); model/ exists locally but not tracked."""
    assert os.path.isfile(os.path.join(REPO_ROOT, "metrics.json"))
    assert os.path.isfile(os.path.join(REPO_ROOT, "predictions.csv"))


# ---------------------------------------------------------------------------
# Gitignore + tracking checks
# ---------------------------------------------------------------------------

def test_model_directory_gitignored():
    """model/ must be in .gitignore AND not tracked by git."""
    gitignore = os.path.join(REPO_ROOT, ".gitignore")
    assert os.path.isfile(gitignore)
    with open(gitignore) as f:
        contents = f.read()
    assert "model/" in contents or "model" in contents.split("\n"), \
        ".gitignore must include a `model/` entry"

    try:
        ls = subprocess.run(
            ["git", "ls-files", "model/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert not ls.stdout.strip(), \
            f"model/ must not be tracked. Tracked files: {ls.stdout!r}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("git not available in CI environment")