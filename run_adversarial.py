"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    df = pd.read_csv(adv_csv_path)
    model.eval()

    predicted_labels = []
    predicted_probs = []
    correct_list = []

    id2label = model.config.id2label

    for _, row in df.iterrows():
        text = row['text']
        expected = row['expected_label']

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            pred_prob = probs[0][pred_id].item()

        # id2label keys in config.json are strings, but model.config.id2label might have ints
        # Let's ensure we handle both
        pred_label = id2label[pred_id] if pred_id in id2label else id2label[str(pred_id)]
        
        predicted_labels.append(pred_label)
        predicted_probs.append(pred_prob)
        correct_list.append(str(pred_label) == str(expected))

    df['predicted_label'] = predicted_labels
    df['predicted_probability'] = predicted_probs
    df['correct'] = correct_list

    return df


def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()
