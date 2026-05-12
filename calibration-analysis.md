# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram (`figures/reliability-diagram.png`) shows a clear pattern of systematic **over-confidence** across nearly all probability buckets. The model's empirical accuracy consistently falls below the diagonal (perfect calibration line), meaning the model trusts its own predictions more than it should.

Specific bucket observations:

| Bucket        | Count | Empirical Accuracy | Expected (midpoint) | Gap    |
|---------------|-------|--------------------|----------------------|--------|
| 0.3–0.4       | 29    | 0.5172             | 0.35                 | +0.17 (under-confident) |
| 0.4–0.5       | 217   | 0.3963             | 0.45                 | −0.05 (over-confident) |
| 0.5–0.6       | 292   | 0.4486             | 0.55                 | −0.10 (over-confident) |
| 0.6–0.7       | 221   | 0.4932             | 0.65                 | −0.16 (over-confident) |
| 0.7–0.8       | 279   | 0.6165             | 0.75                 | −0.13 (over-confident) |
| 0.8–0.9       | 390   | 0.7179             | 0.85                 | −0.13 (over-confident) |
| 0.9–1.0       | 67    | 0.7164             | 0.95                 | −0.23 (over-confident) |

The 0.9–1.0 confidence bucket is the most miscalibrated: the model predicts with 90–100% confidence, yet only 71.6% of those predictions are correct. The low-confidence 0.3–0.4 bucket is the only one where the model is slightly under-confident (accuracy 0.52 vs. expected 0.35), though it contains very few samples (29).

## Expected Calibration Error

**ECE = 0.1225**

An ECE of 0.1225 indicates moderate miscalibration. In practical terms, the model's stated confidence is off by roughly 12 percentage points on average. For production use, this means:

- A prediction made with 85% confidence is actually correct only about 72% of the time.
- Raw softmax probabilities should **not** be surfaced directly to end-users as trust scores without post-hoc calibration.
- For applications where calibrated probabilities matter (e.g., risk-sensitive triage, automated routing based on confidence thresholds), this level of miscalibration could lead to systematically over-trusting the model and routing too many ambiguous cases to the "confident" path.

An ECE below 0.05 is generally considered well-calibrated. At 0.12, this model needs calibration correction before deployment.

## A specific calibration pattern

The dominant pattern is **over-confidence driven by the neutral class**. The per-class metrics reveal:

- **Negative**: Precision 0.58, Recall 0.71, F1 0.64
- **Neutral**: Precision 0.40, Recall 0.25, F1 0.31
- **Positive**: Precision 0.62, Recall 0.70, F1 0.66

The neutral class has drastically lower recall (0.25) — the model misclassifies 75% of neutral samples, typically pushing them into negative or positive with high confidence. This inflates the model's max-probability scores (because it "commits" to negative or positive) while getting the prediction wrong. The high-confidence buckets (0.7–1.0) contain many of these misclassified neutral examples, which is why their empirical accuracy is 10–23 percentage points below the stated confidence.

This pattern arose because: (1) neutral is likely the minority class in the training data, and (2) DistilBERT fine-tuning with cross-entropy loss doesn't penalize over-confidence — the model is rewarded for maximizing the logit of the correct class, which naturally produces sharply peaked softmax distributions even when uncertainty would be more appropriate.

## A proposed engineering action

**Temperature scaling** is the most direct fix for this systematic over-confidence. The approach:

1. Reserve a held-out calibration set (e.g., 10% of the training data, separate from train and test).
2. After fine-tuning, freeze all model weights and learn a single scalar temperature parameter *T* by minimizing negative log-likelihood on the calibration set: replace `softmax(logits)` with `softmax(logits / T)`.
3. For this model's over-confidence pattern, *T* > 1 would soften the probability distributions, bringing them closer to the diagonal.

Temperature scaling is appealing because it preserves the model's ranking (same predictions, same accuracy) while improving probability calibration. It adds zero inference latency and requires only a few lines of code on top of the existing pipeline. Given the ECE of 0.12, I would expect temperature scaling to reduce it to the 0.03–0.05 range, making the model's confidence scores usable for threshold-based routing in production (e.g., abstaining on predictions below 0.6 calibrated confidence and routing those to human review).
