#!/usr/bin/env python3
"""
factoid_loader.py

Downloads and prepares the FACTOID Reddit dataset for domain adaptation.

FACTOID: https://github.com/caisa-lab/FACTOID-dataset
3.3M Reddit posts from 4,150 users with misinformation/credibility labels.

Usage:
    Step 1 - Download the dataset IDs:
        git clone https://github.com/caisa-lab/FACTOID-dataset
        (or download factoid_dataset.gzip manually from the repo)

    Step 2 - Run this script to extract and prepare texts:
        python factoid_loader.py --input factoid_dataset.gzip --output factoid_texts.csv

    Step 3 - Use factoid_texts.csv for domain adaptation training in debertaL_v2.py
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ===================== CONFIGURATION =====================
DEFAULT_INPUT = "FACTOID-dataset/data/factoid_dataset.gzip"
DEFAULT_OUTPUT = "factoid_texts.csv"
MIN_TEXT_LENGTH = 20       # Skip very short posts
MAX_TEXT_LENGTH = 512      # Truncate at token-friendly limit (chars)
MAX_SAMPLES = 100_000      # Cap to keep domain-adapt training fast
RANDOM_SEED = 42

# Credibility score mapping from FACTOID
# very_low=0, low=1, mixed=2, mostly_factual=3, high=4, very_high=5
# We use misinformation posts (low credibility) for propaganda domain adapt
MISINFO_CREDIBILITY_MAX = 1   # Include very_low and low credibility users


def load_factoid(input_path: str) -> pd.DataFrame:
    """Load the FACTOID gzip dataset."""
    print(f"[INFO] Loading FACTOID dataset from {input_path}...")

    if not Path(input_path).exists():
        print(f"\n[ERROR] {input_path} not found.")
        print("\nTo download FACTOID:")
        print("  git clone https://github.com/caisa-lab/FACTOID-dataset")
        print("  # Dataset file is at: FACTOID-dataset/data/factoid_dataset.gzip")
        print("\nAlternatively, download manually from:")
        print("  https://github.com/caisa-lab/FACTOID-dataset")
        raise FileNotFoundError(input_path)

    df = pd.read_pickle(input_path, compression="gzip")
    print(f"[INFO] Loaded DataFrame with shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


def extract_texts(df: pd.DataFrame, strategy: str = "misinfo") -> pd.DataFrame:
    """
    Extract post texts from FACTOID.

    Args:
        df: Raw FACTOID DataFrame
        strategy:
            "misinfo"   - Only posts from low-credibility users (propaganda-likely)
            "all"       - All posts (broader domain adaptation)
            "balanced"  - Equal split misinfo / real news
    """
    print(f"\n[INFO] Extracting texts using strategy='{strategy}'...")

    # Identify the text column (varies by FACTOID version)
    text_col = None
    for candidate in ["body", "selftext", "text", "content", "post_text"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        print(f"[WARNING] No text column found. Available: {df.columns.tolist()}")
        print("[INFO] FACTOID stores post IDs; you may need to crawl text separately.")
        print("       See: https://github.com/caisa-lab/FACTOID-dataset#crawling")
        return pd.DataFrame()

    # Credibility column
    cred_col = None
    for candidate in ["credibility", "factuality", "label", "user_credibility"]:
        if candidate in df.columns:
            cred_col = candidate
            break

    # Filter by strategy
    if strategy == "misinfo" and cred_col is not None:
        mask = df[cred_col] <= MISINFO_CREDIBILITY_MAX
        texts_df = df[mask][[text_col, cred_col]].copy()
        texts_df.columns = ["text", "credibility"]
        print(f"[INFO] Low-credibility posts: {len(texts_df):,}")

    elif strategy == "balanced" and cred_col is not None:
        misinfo = df[df[cred_col] <= MISINFO_CREDIBILITY_MAX][[text_col, cred_col]]
        real = df[df[cred_col] >= 3][[text_col, cred_col]]
        n = min(len(misinfo), len(real), MAX_SAMPLES // 2)
        texts_df = pd.concat([
            misinfo.sample(n, random_state=RANDOM_SEED),
            real.sample(n, random_state=RANDOM_SEED)
        ])
        texts_df.columns = ["text", "credibility"]
        print(f"[INFO] Balanced sample: {len(texts_df):,} posts")

    else:
        texts_df = df[[text_col]].copy()
        texts_df.columns = ["text"]
        if cred_col:
            texts_df["credibility"] = df[cred_col].values
        print(f"[INFO] All posts: {len(texts_df):,}")

    return texts_df


def clean_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter texts for domain adaptation."""
    print(f"\n[INFO] Cleaning {len(df):,} texts...")

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()

    # Remove deleted/removed posts
    df = df[~df["text"].isin(["[deleted]", "[removed]", "nan", ""])]

    # Filter by length
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    df["text"] = df["text"].str[:MAX_TEXT_LENGTH * 4]  # rough char limit

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"[INFO] Removed {before - len(df):,} duplicates")

    # Sample if too large
    if len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=RANDOM_SEED)
        print(f"[INFO] Sampled down to {MAX_SAMPLES:,}")

    print(f"[INFO] Final clean text count: {len(df):,}")
    return df.reset_index(drop=True)


def create_domain_adapt_csv(df: pd.DataFrame, output_path: str):
    """Save domain adaptation dataset."""
    # Keep only text column for domain adaptation
    out = df[["text"]].copy()
    out.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved {len(out):,} texts to {output_path}")
    print(f"[INFO] This file is ready for domain adaptation in debertaL_v2.py")


def create_mock_factoid(output_path: str, n: int = 500):
    """
    Create a small mock FACTOID-style CSV if the real dataset isn't available.
    Useful for testing the pipeline locally.
    """
    print(f"\n[INFO] Creating mock FACTOID dataset ({n} samples) for testing...")

    mock_texts = [
        "The government is hiding the truth about this disaster from the public.",
        "They want to destroy everything our ancestors built for us.",
        "Wake up sheeple, the elites control everything you see and hear.",
        "Our children are being indoctrinated by radical leftist teachers.",
        "The media never reports on the real crimes happening in our cities.",
        "Big Pharma doesn't want you to know about these natural cures.",
        "Every single poll is rigged by the deep state to control elections.",
        "These immigrants are destroying the culture and economy of our country.",
        "The real unemployment rate is being hidden from the American people.",
        "Climate change is a hoax invented to control and tax ordinary citizens.",
        "I just bought groceries today. Prices seem stable at my local store.",
        "The local city council voted to approve the new budget last Tuesday.",
        "Scientists published a new study about protein folding in cells.",
        "The weather forecast says rain is expected later this week.",
        "New community garden opens downtown, accepting volunteer applications.",
    ] * (n // 15 + 1)

    np.random.seed(RANDOM_SEED)
    texts = np.random.choice(mock_texts, n, replace=True).tolist()

    # Add some variation
    suffixes = [
        " Share this before it gets deleted!",
        " The truth they don't want you to know.",
        " Wake up America!",
        " Do your research.",
        "",
        "",
        "",
    ]
    texts = [t + np.random.choice(suffixes) for t in texts]

    df = pd.DataFrame({"text": texts})
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Mock dataset saved to {output_path}")
    print("[WARNING] This is mock data for testing only. Use real FACTOID for training.")
    return df


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description="Prepare FACTOID dataset for domain adaptation")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to factoid_dataset.gzip")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--strategy", choices=["misinfo", "all", "balanced"], default="misinfo",
                        help="Text selection strategy")
    parser.add_argument("--mock", action="store_true",
                        help="Create a small mock dataset for testing (no real FACTOID needed)")
    parser.add_argument("--mock_n", type=int, default=500,
                        help="Number of mock samples to generate")
    args = parser.parse_args()

    print("=" * 70)
    print("FACTOID DATASET LOADER")
    print("=" * 70)

    if args.mock:
        create_mock_factoid(args.output, n=args.mock_n)
        return

    # Load real FACTOID
    try:
        df_raw = load_factoid(args.input)
    except FileNotFoundError:
        print("\n[TIP] Run with --mock to create a test dataset while you download FACTOID:")
        print(f"      python factoid_loader.py --mock --output {args.output}")
        return

    # Extract texts
    df_texts = extract_texts(df_raw, strategy=args.strategy)

    if df_texts.empty:
        print("\n[INFO] No texts extracted. The FACTOID dataset stores post IDs,")
        print("       not raw text. You need to crawl Reddit to fill in the text.")
        print("\nInstructions from the FACTOID repo:")
        print("  https://github.com/caisa-lab/FACTOID-dataset#crawling-reddit-posts")
        print("\nAlternatively, run with --mock for testing:")
        print(f"  python factoid_loader.py --mock --output {args.output}")
        return

    # Clean
    df_clean = clean_texts(df_texts)

    # Save
    create_domain_adapt_csv(df_clean, args.output)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"1. File ready: {args.output}")
    print("2. Run domain adaptation:")
    print("   python debertaL_v2.py --phase domain_adapt --factoid_csv factoid_texts.csv")
    print("3. Then run propaganda fine-tuning:")
    print("   python debertaL_v2.py --phase finetune")
    print("4. Then run active learning:")
    print("   python active_learning_selector.py")


if __name__ == "__main__":
    main()