# Adversarial Evaluation Analysis

This memo summarizes the results of the adversarial evaluation of the fine-tuned sentiment classifier.

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 4 | 6 | 66.7% |
| lexical_trigger | 2 | 6 | 33.3% |
| domain_shift | 3 | 6 | 50.0% |
| length_extreme | 3 | 5 | 60.0% |
| sarcasm | 0 | 5 | 0.0% |
| other | 2 | 5 | 40.0% |

**Overall Accuracy: 42.4% (14/33)**

## Confirmed hypotheses

The model's decision boundary is heavily influenced by specific "sentiment keywords" without sufficient contextual integration, confirming several hypotheses:

- **Sarcasm (IDs 24, 25, 26, 27, 28):** The model failed on 100% of sarcastic examples. For instance, ID 24 ("Oh great another update that breaks everything") was predicted as **positive (0.83)** because of the word "great", completely ignoring the negative consequence described.
- **Lexical Triggers (IDs 12, 13):** The model is easily misled by positive words in negative contexts. ID 12 ("It's a wonderful way to waste your time") was predicted as **positive (0.75)** despite the negative outcome, showing that "wonderful" acts as a high-weight feature that overrides the rest of the sentence.

## Refuted hypotheses

Interestingly, the model handled certain types of complexity better than hypothesized:

- **Simple Negation (IDs 1, 6, 7, 8):** I hypothesized the model would miss negation cues, but it correctly classified "did not improve" (ID 1) and "wouldn't call this app user-friendly" (ID 8). This suggests the fine-tuning successfully captured local negation structures.
- **Long-winded Frustration (ID 23):** Despite the extreme length (50+ words), the model correctly identified the negative sentiment in a very long, rambling sentence, refuting the idea that long sentences always dilute the signal.

## What the results reveal about the decision boundary

The adversarial testing reveals that the model's decision boundary is **asymmetric and keyword-dependent**:

1. **Positive Bias for Keywords:** The model is "trigger-happy" with positive labels. If a sentence contains words like "great", "love", or "beautiful", it almost inevitably tips toward positive, even if negated by sarcasm or complex structure.
2. **Neutral/Negative Confusion:** In domain-shift cases (IDs 16, 18), the model struggles to maintain neutrality, often defaulting to "negative" for news or sports text that mentions "charges" or "final minute".
3. **Double Negation Blindness:** While simple negation works, the model fails on double negations (ID 4: "not bad at all"), treating them as negative because of the word "bad", rather than performing the logical flip.
