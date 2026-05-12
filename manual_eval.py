"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        # Tokenize the batch
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to model's device
        encodings = {k: v.to(device) for k, v in encodings.items()}

        # Forward pass — no gradient computation needed for inference
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits  # shape (batch, num_classes)

        # Softmax to get probabilities
        batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
        batch_preds = np.argmax(batch_probs, axis=-1)

        all_probs.append(batch_probs)
        all_preds.append(batch_preds)

    preds = np.concatenate(all_preds, axis=0)   # (N,)
    probs = np.concatenate(all_probs, axis=0)    # (N, num_classes)

    return preds, probs


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)

    # Accuracy
    accuracy = float(np.sum(y_pred == y_true) / N)

    # Identify all unique class labels
    classes = np.unique(np.concatenate([y_true, y_pred]))

    per_class = {}
    f1_scores = []

    for cls in classes:
        # True positives: predicted cls AND actually cls
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        # False positives: predicted cls BUT actually something else
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        # False negatives: actually cls BUT predicted something else
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))

        # Precision = TP / (TP + FP); guard against division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1 = harmonic mean of precision and recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[int(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
