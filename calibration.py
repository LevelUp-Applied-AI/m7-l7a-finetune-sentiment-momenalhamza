"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    # Bin edges: [0.0, 0.1, 0.2, ..., 1.0]
    edges = np.linspace(0, 1, n_bins + 1)

    # Bucket centers: midpoints of each bin
    bucket_centers = (edges[:-1] + edges[1:]) / 2.0

    # For each prediction, take the max probability and the predicted class
    max_probs = np.max(probs, axis=1)        # (N,)
    pred_classes = np.argmax(probs, axis=1)   # (N,)

    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i < n_bins - 1:
            # edges[i] <= p < edges[i+1]
            mask = (max_probs >= edges[i]) & (max_probs < edges[i + 1])
        else:
            # Last bin is inclusive on the right: edges[i] <= p <= edges[i+1]
            mask = (max_probs >= edges[i]) & (max_probs <= edges[i + 1])

        count = int(np.sum(mask))
        bucket_counts[i] = count

        if count > 0:
            bucket_accuracies[i] = float(np.mean(pred_classes[mask] == y_true[mask]))
        else:
            bucket_accuracies[i] = 0.0

    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    max_probs = np.max(probs, axis=1)
    pred_classes = np.argmax(probs, axis=1)
    N = len(y_true)

    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (max_probs >= edges[i]) & (max_probs < edges[i + 1])
        else:
            mask = (max_probs >= edges[i]) & (max_probs <= edges[i + 1])

        count = int(np.sum(mask))
        if count == 0:
            continue

        bucket_accuracy = float(np.mean(pred_classes[mask] == y_true[mask]))
        bucket_confidence = float(np.mean(max_probs[mask]))

        ece += (count / N) * abs(bucket_accuracy - bucket_confidence)

    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
