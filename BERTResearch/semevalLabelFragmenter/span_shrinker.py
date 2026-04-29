#!/usr/bin/env python3
"""
span_shrinker.py

Post-processing step: converts sentence-level predictions to tighter spans.

The core problem: our model predicts at sentence level, but SemEval scoring
is at character span level. A sentence may be 200 chars; the gold span may
be only 20 chars within it. Outputting the whole sentence gives near-zero
precision even when the prediction is conceptually correct.

This script re-scores every flagged sentence by sliding a window of 
decreasing width across its tokens, keeping the shortest window whose
score exceeds a minimum fraction of the full-sentence score.

Run from: BERTResearch/semevalLabelFragmenter/

    python span_shrinker.py \\
        --sentence_probs  ../sentence_probs.csv \\
        --input_texts     ../semeval_input_texts.csv \\
        --model_dir       ../deberta-propaganda-multilabel_rora_rora \\
        --out_pred        ../pred_tight.csv \\
        --threshold       0.5 \\
        --min_score_frac  0.85

Then evaluate:
    python propaganda_fragment_eval.py \\
        --gold ../gold.csv \\
        --pred ../pred_tight.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

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

# Minimum window sizes (in tokens) per technique.
# Shorter techniques (slogans, loaded_language) can be very brief.
# Longer ones (causal_oversimplification) need more context.
MIN_WINDOW_TOKENS = {
    "loaded_language":                   2,
    "name_calling_labeling":             2,
    "slogans":                           3,
    "thought_terminating_cliches":       3,
    "repetition":                        4,
    "flag_waving":                       4,
    "doubt":                             5,
    "appeal_to_authority":               5,
    "appeal_to_fear_prejudice":          5,
    "exaggeration_minimisation":         5,
    "black_and_white_fallacy":           6,
    "bandwagon_reductio_ad_hitlerum":    6,
    "causal_oversimplification":         7,
    "whataboutism_straw_men_red_herring": 7,
}

# How wide the initial shrink windows are (as fraction of sentence length)
WINDOW_FRACTIONS = [0.8, 0.65, 0.5, 0.35, 0.25]


def norm_label(x: str) -> str:
    return x.strip().lower().replace("-", "_").replace(" ", "_").replace(",", "")


def score_span(token_ids: torch.Tensor, attention_mask: torch.Tensor,
               model, label_idx: int, device: torch.device) -> float:
    """Run model on a token window and return sigmoid prob for label_idx."""
    with torch.no_grad():
        out = model(
            input_ids=token_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
        )
        return float(torch.sigmoid(out.logits[0, label_idx]).cpu())


def shrink_span(
    sentence: str,
    sent_start: int,
    sent_end: int,
    label: str,
    label_idx: int,
    full_score: float,
    tokenizer,
    model,
    device: torch.device,
    min_score_frac: float = 0.85,
    max_length: int = 128,
) -> tuple[int, int]:
    """
    Find the tightest token window within `sentence` whose score for `label`
    is >= min_score_frac * full_score.

    Returns (char_start, char_end) relative to the document.
    Falls back to the full sentence if no shorter window qualifies.
    """
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids    = enc["input_ids"][0]
    attn_mask    = enc["attention_mask"][0]
    offsets      = enc["offset_mapping"][0].numpy()  # (n_tokens, 2)
    n_tokens     = len(input_ids)

    min_win = max(MIN_WINDOW_TOKENS.get(label, 4), 2)
    target  = full_score * min_score_frac

    # Try progressively smaller windows, starting from largest
    best_start_char = sent_start
    best_end_char   = sent_end
    best_score      = full_score
    found           = False

    for frac in WINDOW_FRACTIONS:
        win_size = max(min_win, int(n_tokens * frac))
        if win_size >= n_tokens:
            continue  # no shrinkage at this size

        for start_tok in range(1, n_tokens - win_size):  # skip [CLS]
            end_tok = start_tok + win_size
            if end_tok >= n_tokens - 1:  # skip [SEP]
                break

            # Build token window with [CLS] and [SEP]
            cls   = input_ids[:1]
            sep   = input_ids[-1:]
            chunk = input_ids[start_tok:end_tok]
            ids   = torch.cat([cls, chunk, sep])
            mask  = torch.ones(len(ids), dtype=torch.long)

            sc = score_span(ids, mask, model, label_idx, device)

            if sc >= target and sc >= best_score * 0.97:
                # Map token offsets back to char positions
                tok_start_char = int(offsets[start_tok][0])
                tok_end_char   = int(offsets[end_tok - 1][1])
                doc_start      = sent_start + tok_start_char
                doc_end        = sent_start + tok_end_char

                if doc_end > doc_start:
                    best_start_char = doc_start
                    best_end_char   = doc_end
                    best_score      = sc
                    found           = True

        if found:
            break  # stop at first fraction that works

    return best_start_char, best_end_char


def main():
    ap = argparse.ArgumentParser(
        description="Shrink sentence-level predictions to tight token spans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run from BERTResearch/semevalLabelFragmenter/:

  python span_shrinker.py \\
      --sentence_probs  ../sentence_probs.csv \\
      --input_texts     ../semeval_input_texts.csv \\
      --model_dir       ../deberta-propaganda-multilabel_rora_rora \\
      --out_pred        ../pred_tight.csv

  python propaganda_fragment_eval.py \\
      --gold ../gold.csv \\
      --pred ../pred_tight.csv
        """
    )
    ap.add_argument("--sentence_probs", default="../sentence_probs.csv")
    ap.add_argument("--model_dir",      required=True)
    ap.add_argument("--out_pred",       default="../pred_tight.csv")
    ap.add_argument("--threshold",      type=float, default=0.3,
        help="Minimum sentence-level score to attempt shrinking (default 0.3)")
    ap.add_argument("--min_score_frac", type=float, default=0.85,
        help="Tight span must score >= this fraction of full-sentence score (default 0.85)")
    ap.add_argument("--max_length",     type=int,   default=128)
    ap.add_argument("--use_thresholds", action="store_true",
        help="Load per-label thresholds from model_dir/per_label_thresholds.json. "
             "WARNING: calibrated thresholds target 90%% precision and may filter "
             "too aggressively for evaluation. Use --threshold 0.3-0.5 instead.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading model from {model_dir}...")
    class RoRALinear(torch.nn.Module):
        def __init__(self, linear, rank, alpha, dropout=0.1):
            super().__init__()
            self.in_features  = linear.in_features
            self.out_features = linear.out_features
            self.rank  = rank
            self.scale = alpha / math.sqrt(rank)
            self.weight = linear.weight
            self.bias   = linear.bias
            self.lora_A = torch.nn.Parameter(torch.zeros(rank, linear.in_features))
            self.lora_B = torch.nn.Parameter(torch.zeros(linear.out_features, rank))
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, x):
            base = torch.nn.functional.linear(x, self.weight, self.bias)
            return base + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scale

    def inject_rora(model, rank=32, alpha=64, dropout=0.1,
                    target_layers=list(range(12,24)),
                    target_modules=["query_proj","key_proj","value_proj","dense"]):
        for layer_idx in target_layers:
            layer = model.deberta.encoder.layer[layer_idx]
            for mod_name in target_modules:
                for submod_name, submod in [
                    ("attention.self", layer.attention.self),
                    ("attention.output", layer.attention.output),
                    ("intermediate", layer.intermediate),
                    ("output", layer),
                ]:
                    if hasattr(submod, mod_name):
                        original = getattr(submod, mod_name)
                        if isinstance(original, torch.nn.Linear):
                            setattr(submod, mod_name,
                                    RoRALinear(original, rank, alpha, dropout).to(original.weight.device))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    # Load RoRA adapters — critical, otherwise sub-sentence windows score identically
    # to the full sentence and the shrinker never actually shrinks
    rora_path = model_dir / "rora_adapters.pt"
    if rora_path.exists():
        inject_rora(model)
        state = torch.load(rora_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        adapter_keys = [k for k in state if "lora" in k]
        print(f"[INFO] Loaded {len(adapter_keys)} RoRA adapter tensors from {rora_path}")
    else:
        print(f"[WARNING] rora_adapters.pt not found at {rora_path} — shrinker will not work correctly")

    model.eval()

    # Load label mapping
    with open(model_dir / "label_mapping.json") as f:
        lm = {int(k): v for k, v in json.load(f).items()}
    label_to_idx = {norm_label(v): k for k, v in lm.items()}

    # Per-label thresholds
    if args.use_thresholds:
        thr_path = model_dir / "per_label_thresholds.json"
        if thr_path.exists():
            with open(thr_path) as f:
                thresholds = {k: float(v) for k, v in json.load(f).items()}
            print(f"[INFO] Using per-label thresholds from {thr_path}")
        else:
            print(f"[WARNING] {thr_path} not found, using --threshold {args.threshold}")
            thresholds = {l: args.threshold for l in SEMEVAL_14}
    else:
        thresholds = {l: args.threshold for l in SEMEVAL_14}

    # Load sentence probs — must have a "text" column (re-run export_pred_sentence_fragments.py
    # if your existing sentence_probs.csv is missing it)
    df_probs = pd.read_csv(args.sentence_probs)

    if "text" not in df_probs.columns:
        print("[ERROR] sentence_probs.csv is missing a 'text' column.")
        print("  Re-run export_pred_sentence_fragments.py with the updated version,")
        print("  which stores the sentence text alongside the scores.")
        print("  The updated export script is in your outputs folder.")
        return

    print(f"[INFO] Sentences to process: {len(df_probs):,}")
    print(f"[INFO] Shrink threshold (min_score_frac): {args.min_score_frac}")

    pred_rows = []
    skipped   = 0

    for _, row in tqdm(df_probs.iterrows(), total=len(df_probs), desc="Shrinking spans"):
        doc_id     = str(row["doc_id"])
        sent_start = int(row["start"])
        sent_end   = int(row["end"])
        sentence   = str(row["text"])

        if not sentence.strip():
            skipped += 1
            continue

        for label in SEMEVAL_14:
            if label not in df_probs.columns:
                continue
            score = float(row[label])
            thr   = thresholds.get(label, args.threshold)

            if score < thr:
                continue

            # Find label index in model output
            label_idx = label_to_idx.get(norm_label(label))
            if label_idx is None:
                # Label not in model — still output sentence span
                pred_rows.append({
                    "doc_id": doc_id,
                    "label":  label,
                    "start":  sent_start,
                    "end":    sent_end,
                })
                continue

            # Shrink the span
            tight_start, tight_end = shrink_span(
                sentence=sentence,
                sent_start=sent_start,
                sent_end=sent_end,
                label=label,
                label_idx=label_idx,
                full_score=score,
                tokenizer=tokenizer,
                model=model,
                device=device,
                min_score_frac=args.min_score_frac,
                max_length=args.max_length,
            )

            pred_rows.append({
                "doc_id": doc_id,
                "label":  label,
                "start":  tight_start,
                "end":    tight_end,
            })

    out_path = Path(args.out_pred)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(pred_rows)
    df_out.to_csv(out_path, index=False)

    print(f"\n[DONE] {len(pred_rows):,} tight spans → {out_path}")
    if skipped:
        print(f"[INFO] Skipped {skipped:,} empty sentences")

    # Summary stats
    if pred_rows:
        span_lengths = df_out["end"] - df_out["start"]
        print(f"\nSpan length stats:")
        print(f"  Mean  : {span_lengths.mean():.1f} chars")
        print(f"  Median: {span_lengths.median():.1f} chars")
        print(f"  Min   : {span_lengths.min()} chars")
        print(f"  Max   : {span_lengths.max()} chars")

        print(f"\nPredictions per label:")
        for label in SEMEVAL_14:
            n = (df_out["label"] == label).sum()
            if n > 0:
                print(f"  {label:<45} {n:>5}")

    print(f"\nNext step:")
    print(f"  python propaganda_fragment_eval.py \\")
    print(f"      --gold ../gold.csv --pred {out_path}")


if __name__ == "__main__":
    main()