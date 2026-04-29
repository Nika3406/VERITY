#!/usr/bin/env python3
"""
semeval_data_processor.py  (UNIFIED — replaces v1 and v2)

Reads semeval_examples.csv (output of semeval_extract_with_gold.py)
and produces semeval_processed.csv for training.

Key design decisions
---------------------
1.  Canonical 14-label set only (matches gold.csv / SemEval-2020 Task 11).
    No more 18-label messy mapping.

2.  Soft (percentage) labels instead of binary 0/1.
    Each label value = fraction of the text that is covered by that technique,
    capped at 1.0.  This gives the model calibration signal — a snippet that is
    entirely loaded language gets 1.0, one that is 80% loaded language gets 0.8.

    During training, BCEWithLogitsLoss handles float targets in [0, 1] natively.
    During evaluation, we still threshold at a per-label cutoff.

3.  Deduplication by text, keeping the union of all technique labels across
    duplicate snippets (same text may appear with different techniques).

4.  Statistics report so you can see the label distribution before training.

Usage
-----
    python semeval_data_processor.py

Input:  semeval_examples.csv   (doc_id, label_raw, label, start, end, text_snippet)
Output: semeval_processed.csv  (text, <14 label columns as floats 0.0–1.0>)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ===================== CANONICAL LABEL SET =====================
# Must match gold.csv exactly — do not change order or spelling.
SEMEVAL_14 = [
    "appeal_to_authority",
    "appeal_to_fear_prejudice",
    "bandwagon_reductio_ad_hitlerum",
    "black_and_white_fallacy",
    "causal_oversimplification",
    "doubt",
    "exaggeration_minimisation",
    "flag_waving",
    "loaded_language",
    "name_calling_labeling",
    "repetition",
    "slogans",
    "thought_terminating_cliches",
    "whataboutism_straw_men_red_herring",
]

# ===================== RAW → CANONICAL MAPPING =====================
# Every raw SemEval label that has ever appeared in the dataset.
# Maps to one of the 14 canonical labels above.
RAW_TO_CANON: dict[str, str] = {
    # --- exact SemEval-2020 raw strings ---
    "Appeal_to_Authority":                    "appeal_to_authority",
    "Appeal_to_fear-prejudice":               "appeal_to_fear_prejudice",
    "Bandwagon,Reductio_ad_hitlerum":         "bandwagon_reductio_ad_hitlerum",
    "Black-and-White_Fallacy":                "black_and_white_fallacy",
    "Causal_Oversimplification":              "causal_oversimplification",
    "Doubt":                                  "doubt",
    "Exaggeration,Minimisation":              "exaggeration_minimisation",
    "Flag-Waving":                            "flag_waving",
    "Loaded_Language":                        "loaded_language",
    "Name_Calling,Labeling":                  "name_calling_labeling",
    "Repetition":                             "repetition",
    "Slogans":                                "slogans",
    "Thought-terminating_Cliches":            "thought_terminating_cliches",
    "Whataboutism,Straw_Men,Red_Herring":     "whataboutism_straw_men_red_herring",
    # --- common normalisation variants (in case semeval_extract.py produced them) ---
    "appeal_to_authority":                    "appeal_to_authority",
    "appeal_to_fear_prejudice":               "appeal_to_fear_prejudice",
    "appeal_to_fear/prejudice":               "appeal_to_fear_prejudice",
    "appeal_to_fear":                         "appeal_to_fear_prejudice",
    "bandwagon_reductio_ad_hitlerum":         "bandwagon_reductio_ad_hitlerum",
    "bandwagon":                              "bandwagon_reductio_ad_hitlerum",
    "reductio_ad_hitlerum":                   "bandwagon_reductio_ad_hitlerum",
    "black_and_white_fallacy":                "black_and_white_fallacy",
    "black_white_fallacy":                    "black_and_white_fallacy",
    "black-and-white_fallacy":                "black_and_white_fallacy",
    "causal_oversimplification":              "causal_oversimplification",
    "oversimplification":                     "causal_oversimplification",
    "scapegoating":                           "causal_oversimplification",
    "doubt":                                  "doubt",
    "exaggeration_minimisation":              "exaggeration_minimisation",
    "exaggeration":                           "exaggeration_minimisation",
    "minimisation":                           "exaggeration_minimisation",
    "minimization":                           "exaggeration_minimisation",
    "flag_waving":                            "flag_waving",
    "flag-waving":                            "flag_waving",
    "loaded_language":                        "loaded_language",
    "name_calling_labeling":                  "name_calling_labeling",
    "name_calling":                           "name_calling_labeling",
    "labeling":                               "name_calling_labeling",
    "name_calling,labeling":                  "name_calling_labeling",
    "repetition":                             "repetition",
    "slogans":                                "slogans",
    "slogan":                                 "slogans",
    "thought_terminating_cliches":            "thought_terminating_cliches",
    "thought_terminating_cliche":             "thought_terminating_cliches",
    "thought-terminating_cliches":            "thought_terminating_cliches",
    "whataboutism_straw_men_red_herring":     "whataboutism_straw_men_red_herring",
    "whataboutism":                           "whataboutism_straw_men_red_herring",
    "straw_man":                              "whataboutism_straw_men_red_herring",
    "strawman":                               "whataboutism_straw_men_red_herring",
    "red_herring":                            "whataboutism_straw_men_red_herring",
    "red herring":                            "whataboutism_straw_men_red_herring",
    "whataboutism,straw_men,red_herring":     "whataboutism_straw_men_red_herring",
    # --- old semeval_data_processor.py outputs (v1 18-label space) ---
    "appeal_to_fear":                         "appeal_to_fear_prejudice",
    "black_white_fallacy":                    "black_and_white_fallacy",
    "thought_terminating_cliche":             "thought_terminating_cliches",
    "obfuscation":                            "loaded_language",   # nearest match
    "slogans":                                "slogans",
    "flag_waving":                            "flag_waving",
}


def _normalise_raw(label_raw: str) -> str | None:
    """Return canonical label or None if unmappable."""
    stripped = label_raw.strip()
    if stripped in RAW_TO_CANON:
        return RAW_TO_CANON[stripped]
    # fallback: lowercase + replace punctuation
    normalised = (
        stripped.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace("/", "_")
    )
    return RAW_TO_CANON.get(normalised, None)


def _coverage(snippet: str, span_len: int) -> float:
    """
    Soft label: fraction of the snippet covered by one propaganda span.

    Since semeval_examples.csv stores text_snippet = the extracted span itself
    (not the full sentence), span_len == len(snippet) by construction, giving 1.0.

    HOWEVER: if the same text appears multiple times with the same label,
    we cap at 1.0 regardless of how many overlapping annotations exist.

    For genuinely partial spans (future data with full-sentence context),
    this will return the correct ratio < 1.0.
    """
    if len(snippet) == 0:
        return 0.0
    return min(1.0, span_len / len(snippet))


# ===================== MAIN =====================

INPUT_FILE  = "semeval_examples.csv"
OUTPUT_FILE = "semeval_processed.csv"
MIN_TEXT_LEN = 10


def main() -> None:
    print("=" * 70)
    print("SEMEVAL DATA PROCESSOR  (unified, 14-label, soft labels)")
    print("=" * 70)

    # ---- Load ----
    if not Path(INPUT_FILE).exists():
        sys.exit(
            f"[ERROR] {INPUT_FILE} not found.\n"
            "Run semeval_extract_with_gold.py first:\n"
            "  python semeval_extract_with_gold.py "
            "--label_file semeval_datasets/train-task2-TC.labels "
            "--articles_dir semeval_datasets/train-articles"
        )

    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Loaded {len(df):,} rows from {INPUT_FILE}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    # Detect text column
    for cname in ["text_snippet", "text", "snippet"]:
        if cname in df.columns:
            text_col = cname
            break
    else:
        sys.exit(f"[ERROR] No text column found. Available: {df.columns.tolist()}")

    # Detect label column
    for cname in ["label", "label_raw"]:
        if cname in df.columns:
            label_col = cname
            break
    else:
        sys.exit(f"[ERROR] No label column found. Available: {df.columns.tolist()}")

    print(f"[INFO] Using text column='{text_col}', label column='{label_col}'")

    # ---- Normalise labels ----
    df["_canon"] = df[label_col].apply(_normalise_raw)

    unknown = df[df["_canon"].isna()][label_col].unique()
    if len(unknown):
        print(f"[WARNING] {len(unknown)} unmappable label(s) — will be skipped:")
        for u in sorted(unknown):
            print(f"  '{u}'")

    df = df[df["_canon"].notna()].copy()
    print(f"[INFO] {len(df):,} rows after dropping unmappable labels")

    # ---- Clean text ----
    df["_text"] = df[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["_text"].str.len() >= MIN_TEXT_LEN].copy()

    # ---- Compute soft label per (text, technique) ----
    # For each row: coverage = len(span) / len(snippet)
    # Since text_snippet IS the span, this is almost always 1.0.
    # We keep the formula here so it generalises to future full-sentence data.
    if "start" in df.columns and "end" in df.columns:
        df["_span_len"] = (df["end"] - df["start"]).clip(lower=0)
    else:
        # No offsets available — assume full coverage
        df["_span_len"] = df["_text"].str.len()

    df["_coverage"] = df.apply(
        lambda r: _coverage(r["_text"], r["_span_len"]), axis=1
    )

    # ---- Aggregate: one row per unique text ----
    # For each text, take the MAX coverage per label across all its spans.
    # (Multiple overlapping spans of the same technique → cap at 1.0)
    text_label_max: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for _, row in df.iterrows():
        text  = row["_text"]
        label = row["_canon"]
        cov   = row["_coverage"]
        text_label_max[text][label] = max(text_label_max[text][label], cov)

    # ---- Build output DataFrame ----
    rows = []
    for text, label_vals in text_label_max.items():
        row: dict = {"text": text}
        for lab in SEMEVAL_14:
            row[lab] = round(label_vals.get(lab, 0.0), 4)
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # ---- Drop rows where all labels are zero (shouldn't happen, safety net) ----
    label_sum = df_out[SEMEVAL_14].sum(axis=1)
    n_zero = (label_sum == 0).sum()
    if n_zero:
        print(f"[WARNING] Dropping {n_zero} rows with all-zero labels")
        df_out = df_out[label_sum > 0]

    df_out = df_out.reset_index(drop=True)

    # ---- Save ----
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[SUCCESS] Saved {len(df_out):,} samples → {OUTPUT_FILE}")

    # ---- Statistics ----
    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)
    print(f"\n{'Label':<45} {'N>0':>6} {'% rows':>8} {'Mean':>8} {'Max':>6}")
    print("-" * 75)

    for lab in SEMEVAL_14:
        col        = df_out[lab]
        n_nonzero  = (col > 0).sum()
        pct        = 100 * n_nonzero / len(df_out)
        mean_val   = col[col > 0].mean() if n_nonzero else 0.0
        max_val    = col.max()
        flag       = " ⚠  LOW" if n_nonzero < 30 else ""
        print(f"  {lab:<43} {n_nonzero:>6} {pct:>7.1f}% {mean_val:>8.3f} {max_val:>6.3f}{flag}")

    avg_active = (df_out[SEMEVAL_14] > 0).sum(axis=1).mean()
    print(f"\n  Average active labels per sample : {avg_active:.2f}")
    print(f"  Total samples                    : {len(df_out):,}")

    # ---- Soft vs hard comparison ----
    binary_sum   = (df_out[SEMEVAL_14] > 0).sum().sum()
    soft_sum     = df_out[SEMEVAL_14].sum().sum()
    print(f"\n  Binary label mass  (count of 1s) : {binary_sum:,}")
    print(f"  Soft   label mass  (sum of vals) : {soft_sum:.1f}")
    print(f"  Ratio (soft/binary)              : {soft_sum/binary_sum:.3f}")
    print(
        "\n  A ratio near 1.0 means most spans are full-coverage (expected).\n"
        "  When you add full-sentence context later, this ratio will drop\n"
        "  below 1.0 and the soft labels will carry more signal.\n"
    )

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"  1.  Train   : python debertaL_v2.py --phase finetune")
    print(f"  2.  Export  : python export_pred_sentence_fragments.py \\")
    print(f"                  --input_texts semeval_input_texts.csv \\")
    print(f"                  --model_dir   ./deberta-propaganda-multilabel \\")
    print(f"                  --write_pred")
    print(f"  3.  Sweep   : python threshold_sweep_semeval.py \\")
    print(f"                  --gold gold.csv --sentence_probs sentence_probs.csv")
    print(f"  4.  Eval    : python threshold_optimizer.py \\")
    print(f"                  --model_dir ./deberta-propaganda-multilabel \\")
    print(f"                  --gold gold.csv --sentence_probs sentence_probs.csv")
    print(f"  5.  Score   : python propaganda_fragment_eval.py \\")
    print(f"                  --gold gold.csv --pred pred.csv")


if __name__ == "__main__":
    main()