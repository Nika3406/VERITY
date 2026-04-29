#!/usr/bin/env python3
"""
active_learning_selector.py

Implements active learning selection for propaganda labeling.
Instead of randomly sampling Reddit posts for LLM labeling,
this selects the MOST INFORMATIVE samples — those where the
current model is most uncertain.

This follows your professor's recommendation:
  "Select the top 1,000–1,500 most uncertain or borderline
   high-confidence samples (active-learning style: high entropy
   or confidence close to threshold)."

Usage:
    # After training a base model (even an early checkpoint):
    python active_learning_selector.py \
        --reddit_csv datasets/ \
        --model_dir ./deberta-propaganda-multilabel \
        --output_csv active_learning_candidates.csv \
        --n_samples 1500 \
        --strategy entropy

Strategies:
    entropy      - Select samples with highest prediction entropy (most uncertain)
    margin       - Select samples where top-2 label scores are closest
    least_conf   - Select samples with lowest max confidence
    mixed        - Combine all three strategies (recommended)
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# ===================== CONFIGURATION =====================
DEFAULT_MODEL_DIR   = "./deberta-propaganda-multilabel"
DEFAULT_REDDIT_DIR  = "./datasets"
DEFAULT_OUTPUT      = "active_learning_candidates.csv"
DEFAULT_N_SAMPLES   = 1500
DEFAULT_STRATEGY    = "mixed"
BATCH_SIZE          = 16
MAX_LENGTH          = 128
MIN_TEXT_LEN        = 20


# ===================== DATASET =====================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ===================== UNCERTAINTY METRICS =====================

def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Multi-label entropy: sum of binary entropies per label.

    High entropy = model is uncertain about many labels simultaneously.
    This is the primary uncertainty signal for active learning.

    Args:
        probs: shape (N, num_labels), values in [0, 1]

    Returns:
        entropy: shape (N,), higher = more uncertain
    """
    # Binary entropy for each label: -p*log(p) - (1-p)*log(1-p)
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    per_label_entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    # Sum across labels — uncertain on more labels = more valuable
    return per_label_entropy.sum(axis=1)


def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Margin sampling: difference between top-2 label confidences.

    Low margin = model can't distinguish which is the dominant technique.
    
    Args:
        probs: shape (N, num_labels)

    Returns:
        margin: shape (N,), LOWER = more uncertain (we negate before sorting)
    """
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    if sorted_probs.shape[1] >= 2:
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        margin = sorted_probs[:, 0]
    # Return negative so higher = more uncertain (consistent with entropy)
    return -margin


def least_confidence(probs: np.ndarray) -> np.ndarray:
    """
    Least confidence: distance from 0.5 threshold.

    Samples closest to 0.5 on their most confident label are most uncertain.

    Args:
        probs: shape (N, num_labels)

    Returns:
        uncertainty: shape (N,), higher = more uncertain
    """
    # Distance from 0.5 for each label
    dist_from_threshold = np.abs(probs - 0.5)
    # Min distance across labels (most uncertain label drives selection)
    min_dist = dist_from_threshold.min(axis=1)
    # Invert: 0 distance = perfectly uncertain
    return -min_dist


def borderline_high_confidence(probs: np.ndarray, threshold: float = 0.5,
                                 margin: float = 0.15) -> np.ndarray:
    """
    Identify samples that are "borderline high confidence" —
    near but above/below the decision threshold.

    These are the most valuable for LLM labeling because:
    - The model has made a decision
    - But we're not sure if that decision is correct

    Args:
        probs: shape (N, num_labels)
        threshold: decision threshold (typically 0.5)
        margin: how close to threshold counts as borderline

    Returns:
        score: shape (N,), higher = more borderline
    """
    dist = np.abs(probs - threshold)
    # Check if ANY label is within the margin
    in_margin = (dist < margin).any(axis=1).astype(float)
    # Score by how many labels are in the margin zone
    margin_count = (dist < margin).sum(axis=1).astype(float)
    return in_margin * margin_count


def combined_uncertainty(probs: np.ndarray, weights: dict = None) -> np.ndarray:
    """
    Combine multiple uncertainty signals.

    Args:
        probs: shape (N, num_labels)
        weights: dict of {metric_name: weight}

    Returns:
        combined_score: shape (N,), higher = more uncertain/valuable
    """
    if weights is None:
        weights = {
            "entropy": 0.4,
            "margin": 0.3,
            "borderline": 0.3,
        }

    scores = np.zeros(len(probs))

    if "entropy" in weights:
        e = prediction_entropy(probs)
        e_norm = (e - e.min()) / (e.max() - e.min() + 1e-9)
        scores += weights["entropy"] * e_norm

    if "margin" in weights:
        m = margin_uncertainty(probs)
        m_norm = (m - m.min()) / (m.max() - m.min() + 1e-9)
        scores += weights["margin"] * m_norm

    if "borderline" in weights:
        b = borderline_high_confidence(probs)
        b_norm = (b - b.min()) / (b.max() - b.min() + 1e-9)
        scores += weights["borderline"] * b_norm

    return scores


# ===================== INFERENCE =====================

def run_inference(texts: list, model, tokenizer, device: torch.device,
                  batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Run model inference on texts and return sigmoid probabilities.

    Args:
        texts: list of strings
        model: loaded DeBERTa model
        tokenizer: loaded tokenizer
        device: torch device

    Returns:
        probs: np.ndarray of shape (N, num_labels)
    """
    model.eval()
    all_probs = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)

    return np.vstack(all_probs)


# ===================== LOADING REDDIT DATA =====================

def load_reddit_texts(reddit_dir: str) -> pd.DataFrame:
    """Load all Reddit CSV files from the datasets directory."""
    csv_files = glob.glob(os.path.join(reddit_dir, "S-Dataset_r_*.csv"))
    csv_files += glob.glob(os.path.join(reddit_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {reddit_dir}. "
            "Expected files matching 'S-Dataset_r_*.csv'"
        )

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if "text" in df.columns:
                dfs.append(df[["text"]])
            elif "body" in df.columns:
                df = df.rename(columns={"body": "text"})
                dfs.append(df[["text"]])
        except Exception as e:
            print(f"[WARNING] Could not load {f}: {e}")

    if not dfs:
        raise ValueError("No valid text columns found in any CSV file.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["text"] = df_all["text"].astype(str).str.strip()
    df_all = df_all[df_all["text"].str.len() >= MIN_TEXT_LEN]
    df_all = df_all[~df_all["text"].isin(["[deleted]", "[removed]", "nan"])]
    df_all = df_all.drop_duplicates(subset=["text"])

    print(f"[INFO] Loaded {len(df_all):,} unique Reddit texts")
    return df_all.reset_index(drop=True)


# ===================== MAIN SELECTION LOGIC =====================

def select_active_learning_samples(
    reddit_dir: str,
    model_dir: str,
    output_csv: str,
    n_samples: int = DEFAULT_N_SAMPLES,
    strategy: str = DEFAULT_STRATEGY,
    exclude_already_labeled: str = None,
    diversity_boost: bool = True,
):
    """
    Full active learning selection pipeline.

    Args:
        reddit_dir: Directory with Reddit CSV files
        model_dir: Path to trained DeBERTa model
        output_csv: Where to save selected samples
        n_samples: How many samples to select
        strategy: "entropy", "margin", "least_conf", or "mixed"
        exclude_already_labeled: CSV of already-labeled texts to exclude
        diversity_boost: Ensure rare techniques are represented
    """

    print("=" * 70)
    print("ACTIVE LEARNING SELECTOR")
    print("=" * 70)
    print(f"Strategy:  {strategy}")
    print(f"N samples: {n_samples:,}")
    print(f"Model:     {model_dir}")
    print()

    # ---- Load model ----
    print("[INFO] Loading model...")
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Train a base model first with debertaL.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Model loaded on {device}")

    # Load label mapping
    label_file = os.path.join(model_dir, "label_mapping.json")
    with open(label_file) as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    num_labels = len(label_mapping)
    label_names = [label_mapping[i] for i in range(num_labels)]
    print(f"[INFO] {num_labels} propaganda techniques")

    # ---- Load Reddit data ----
    df_reddit = load_reddit_texts(reddit_dir)

    # ---- Exclude already-labeled samples ----
    if exclude_already_labeled and Path(exclude_already_labeled).exists():
        df_labeled = pd.read_csv(exclude_already_labeled)
        if "text" in df_labeled.columns:
            already_labeled = set(df_labeled["text"].tolist())
            before = len(df_reddit)
            df_reddit = df_reddit[~df_reddit["text"].isin(already_labeled)]
            print(f"[INFO] Excluded {before - len(df_reddit):,} already-labeled samples")

    print(f"[INFO] Candidate pool: {len(df_reddit):,} texts")

    # ---- Run inference ----
    print("\n[INFO] Running model inference on candidate pool...")
    texts = df_reddit["text"].tolist()
    probs = run_inference(texts, model, tokenizer, device)

    # ---- Compute uncertainty scores ----
    print(f"\n[INFO] Computing uncertainty scores (strategy='{strategy}')...")

    if strategy == "entropy":
        scores = prediction_entropy(probs)
    elif strategy == "margin":
        scores = margin_uncertainty(probs)
    elif strategy == "least_conf":
        scores = least_confidence(probs)
    elif strategy == "mixed":
        scores = combined_uncertainty(probs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df_reddit["uncertainty_score"] = scores

    # Add per-label probabilities for inspection
    for i, label in enumerate(label_names):
        df_reddit[f"prob_{label}"] = probs[:, i]

    # Add prediction entropy for reference
    df_reddit["entropy"] = prediction_entropy(probs)

    # ---- Diversity boost ----
    # Ensure rare techniques are represented in selection
    if diversity_boost and n_samples >= 100:
        print("\n[INFO] Applying diversity boost for rare techniques...")
        selected_indices = set()
        n_per_technique = max(10, n_samples // (num_labels * 2))

        for i, label in enumerate(label_names):
            # For each technique, pick top uncertain samples where this technique is active
            technique_mask = probs[:, i] > 0.3  # Some activation
            if technique_mask.sum() < 5:
                continue
            technique_indices = np.where(technique_mask)[0]
            technique_scores = scores[technique_indices]
            top_k = min(n_per_technique, len(technique_indices))
            top_idx = technique_indices[np.argsort(technique_scores)[-top_k:]]
            selected_indices.update(top_idx.tolist())

        # Fill remaining slots with globally most uncertain
        remaining = n_samples - len(selected_indices)
        if remaining > 0:
            sorted_indices = np.argsort(scores)[::-1]
            for idx in sorted_indices:
                if idx not in selected_indices:
                    selected_indices.add(idx)
                if len(selected_indices) >= n_samples:
                    break

        selected_indices = list(selected_indices)[:n_samples]
        df_selected = df_reddit.iloc[selected_indices].copy()

    else:
        # Pure uncertainty sampling
        df_selected = df_reddit.nlargest(n_samples, "uncertainty_score")

    # ---- Sort by uncertainty score ----
    df_selected = df_selected.sort_values("uncertainty_score", ascending=False)
    df_selected = df_selected.reset_index(drop=True)
    df_selected["selection_rank"] = df_selected.index + 1

    # ---- Save output ----
    df_selected.to_csv(output_csv, index=False)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)
    print(f"Selected: {len(df_selected):,} samples from {len(df_reddit):,} candidates")
    print(f"Output:   {output_csv}")

    print(f"\nUncertainty Score Distribution:")
    print(f"  Mean:   {df_selected['uncertainty_score'].mean():.4f}")
    print(f"  Median: {df_selected['uncertainty_score'].median():.4f}")
    print(f"  Min:    {df_selected['uncertainty_score'].min():.4f}")
    print(f"  Max:    {df_selected['uncertainty_score'].max():.4f}")

    print(f"\nTop-5 most uncertain samples:")
    for _, row in df_selected.head(5).iterrows():
        text_preview = row["text"][:100].replace("\n", " ")
        print(f"\n  Rank {int(row['selection_rank'])} | Score: {row['uncertainty_score']:.4f}")
        print(f"  Text: {text_preview}...")

    print(f"\nPer-technique coverage in selected set:")
    prob_cols = [c for c in df_selected.columns if c.startswith("prob_")]
    for col in prob_cols:
        label = col.replace("prob_", "")
        activated = (df_selected[col] > 0.3).sum()
        print(f"  {label:<45} {activated:>5} samples ({activated/len(df_selected)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"1. Send {output_csv} for LLM labeling:")
    print(f"   python llm_ensemble_labeler.py --input {output_csv}")
    print(f"2. Or manually review top-ranked samples first")
    print(f"3. After labeling, merge into training set and retrain")

    return df_selected


# ===================== ENTRY POINT =====================

def main():
    parser = argparse.ArgumentParser(
        description="Active learning sample selector for propaganda detection"
    )
    parser.add_argument(
        "--reddit_csv", default=DEFAULT_REDDIT_DIR,
        help="Directory containing Reddit CSV files"
    )
    parser.add_argument(
        "--model_dir", default=DEFAULT_MODEL_DIR,
        help="Path to trained DeBERTa model"
    )
    parser.add_argument(
        "--output_csv", default=DEFAULT_OUTPUT,
        help="Output CSV for selected samples"
    )
    parser.add_argument(
        "--n_samples", type=int, default=DEFAULT_N_SAMPLES,
        help="Number of samples to select (professor recommends 1000-1500)"
    )
    parser.add_argument(
        "--strategy",
        choices=["entropy", "margin", "least_conf", "mixed"],
        default=DEFAULT_STRATEGY,
        help="Uncertainty sampling strategy"
    )
    parser.add_argument(
        "--exclude", default=None,
        help="CSV of already-labeled texts to exclude"
    )
    parser.add_argument(
        "--no_diversity", action="store_true",
        help="Disable diversity boost (pure uncertainty sampling)"
    )

    args = parser.parse_args()

    select_active_learning_samples(
        reddit_dir=args.reddit_csv,
        model_dir=args.model_dir,
        output_csv=args.output_csv,
        n_samples=args.n_samples,
        strategy=args.strategy,
        exclude_already_labeled=args.exclude,
        diversity_boost=not args.no_diversity,
    )


if __name__ == "__main__":
    main()