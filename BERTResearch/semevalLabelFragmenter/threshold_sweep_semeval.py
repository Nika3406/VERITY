#!/usr/bin/env python3
"""
threshold_sweep_semeval.py

Sweeps thresholds PER SemEval technique and selects the best threshold
using one of three strategies depending on the curve shape:

  1. PEAK   — curve has a clear local maximum above the floor. Use that peak.
  2. ELBOW  — curve is monotonically decreasing. Find the elbow (sharpest
               change in slope) — the natural balance between precision/recall.
  3. NOISY  — curve is erratic with no reliable trend (small gold count).
               Use the first local maximum above the floor as a conservative estimate.

Outputs:
  - *_best_thresholds.csv   (with strategy column)
  - *_per_label_curves.csv
  - *_needs_work.csv
  - *_best_f1_by_label.png
  - *_final_thresholds.png
  - *_f1_vs_thr_<label>.png  (one per label, threshold marked)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


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

# Curve analysis parameters
PEAK_MIN_THR   = 0.20   # Peak must be above this threshold to count as real
PEAK_MIN_GAIN  = 0.005  # Min absolute F1 gain from floor to peak
NOISY_GOLD_MAX = 100    # Gold count below which curve is treated as noisy
FLAT_RANGE     = 0.01   # If max-min F1 is below this, curve is flat


@dataclass(frozen=True)
class Frag:
    start: int
    end: int
    label: str
    def length(self) -> int:
        return max(0, self.end - self.start)


def overlap_len(a: Frag, b: Frag) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def C(s: Frag, t: Frag, h: int) -> float:
    if h <= 0 or s.label != t.label:
        return 0.0
    return overlap_len(s, t) / float(h)


def precision(S: List[Frag], T: List[Frag]) -> float:
    if not S:
        return 0.0
    total = 0.0
    for s in S:
        h = s.length()
        if h == 0:
            continue
        for t in T:
            total += C(s, t, h)
    return total / float(len(S))


def recall(S: List[Frag], T: List[Frag]) -> float:
    if not T:
        return 0.0
    total = 0.0
    for t in T:
        h = t.length()
        if h == 0:
            continue
        for s in S:
            total += C(s, t, h)
    return total / float(len(T))


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else (2.0 * p * r) / (p + r)


def load_gold(gold_csv: str) -> Dict[str, List[Frag]]:
    df = pd.read_csv(gold_csv)
    by_doc = defaultdict(list)
    for _, r in df.iterrows():
        by_doc[str(r["doc_id"])].append(
            Frag(int(r["start"]), int(r["end"]), str(r["label"]))
        )
    bad = sorted(set(df["label"]) - set(SEMEVAL_14))
    if bad:
        raise ValueError(f"gold.csv contains labels outside SemEval-14: {bad}")
    return by_doc


def make_pred_from_probs(probs_df: pd.DataFrame, label: str, thr: float) -> Dict[str, List[Frag]]:
    by_doc = defaultdict(list)
    for _, r in probs_df[probs_df[label] >= thr].iterrows():
        by_doc[str(r["doc_id"])].append(Frag(int(r["start"]), int(r["end"]), label))
    return by_doc


def pooled_score(gold_by_doc, pred_by_doc, label):
    pooled_T, pooled_S = [], []
    for d in set(gold_by_doc) | set(pred_by_doc):
        pooled_T.extend([t for t in gold_by_doc.get(d, []) if t.label == label])
        pooled_S.extend([s for s in pred_by_doc.get(d, []) if s.label == label])
    p = precision(pooled_S, pooled_T)
    r = recall(pooled_S, pooled_T)
    return p, r, f1(p, r), len(pooled_T), len(pooled_S)


def find_elbow(thresholds: np.ndarray, f1_vals: np.ndarray) -> float:
    """
    Find the elbow of a monotonically decreasing curve.
    Uses the point of maximum second derivative — where the curve
    bends most sharply, i.e. the transition from gradual to steep decline.
    This is the natural balance between precision and recall.
    """
    if len(f1_vals) < 3:
        return float(thresholds[0])

    # Smooth to reduce noise before derivatives
    kernel   = np.ones(3) / 3
    smoothed = np.convolve(f1_vals, kernel, mode="same")

    d1 = np.diff(smoothed)
    d2 = np.diff(d1)

    if len(d2) == 0:
        return float(thresholds[0])

    # Elbow = where slope changes most steeply downward
    elbow_idx = int(np.argmin(d2)) + 1
    elbow_idx = max(0, min(elbow_idx, len(thresholds) - 1))
    return float(thresholds[elbow_idx])


def select_threshold(lab_df: pd.DataFrame, gold_count: int,
                     tmin: float, tstep: float) -> Tuple[float, str]:
    """
    Intelligently select the best threshold based on curve shape.
    Returns (threshold, strategy_name).
    """
    thresholds = lab_df["thr"].values
    f1_vals    = lab_df["f1"].values
    f_range    = f1_vals.max() - f1_vals.min()
    floor_f1   = lab_df[lab_df["thr"] <= tmin + tstep]["f1"].max()

    # FLAT — curve barely moves
    if f_range < FLAT_RANGE:
        return float(thresholds[len(thresholds) // 2]), "flat_midpoint"

    # NOISY — small gold count, erratic curve
    if gold_count <= NOISY_GOLD_MAX:
        for i in range(1, len(f1_vals) - 1):
            if (f1_vals[i] > f1_vals[i-1] and
                    f1_vals[i] > f1_vals[i+1] and
                    thresholds[i] > PEAK_MIN_THR):
                return float(thresholds[i]), "noisy_first_peak"
        return float(thresholds[len(thresholds) // 2]), "noisy_midpoint"

    # PEAK — curve has a clear maximum above the floor region
    peak_idx = int(np.argmax(f1_vals))
    peak_thr = float(thresholds[peak_idx])
    peak_f1  = float(f1_vals[peak_idx])

    if peak_thr > PEAK_MIN_THR and (peak_f1 - floor_f1) > PEAK_MIN_GAIN:
        return peak_thr, "peak"

    # MONOTONE DECREASING — find the elbow
    mid = len(f1_vals) // 2
    if f1_vals[:mid].mean() > f1_vals[mid:].mean():
        elbow_thr = find_elbow(thresholds, f1_vals)
        return elbow_thr, "elbow"

    # FALLBACK — use raw sweep peak
    return peak_thr, "raw_peak"


STRATEGY_COLORS = {
    "peak":             "#e74c3c",
    "elbow":            "#2ecc71",
    "noisy_first_peak": "#e67e22",
    "noisy_midpoint":   "#e67e22",
    "flat_midpoint":    "#95a5a6",
    "raw_peak":         "#3498db",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold",            required=True)
    ap.add_argument("--sentence_probs",  required=True)
    ap.add_argument("--out_prefix",      default="semeval_sweep")
    ap.add_argument("--tmin",            type=float, default=0.05)
    ap.add_argument("--tmax",            type=float, default=0.95)
    ap.add_argument("--tstep",           type=float, default=0.05)
    ap.add_argument("--min_gold",        type=int,   default=5)
    ap.add_argument("--needs_work_f1",   type=float, default=0.20)
    ap.add_argument("--needs_work_thr_hi", type=float, default=0.85)
    args = ap.parse_args()

    gold_by_doc = load_gold(args.gold)
    probs_df    = pd.read_csv(args.sentence_probs)

    missing = [l for l in SEMEVAL_14 if l not in probs_df.columns]
    if missing:
        raise ValueError(f"sentence_probs.csv missing SemEval-14 columns: {missing}")

    thresholds = np.arange(args.tmin, args.tmax + 1e-9, args.tstep)

    gold_counts = defaultdict(int)
    for Ts in gold_by_doc.values():
        for t in Ts:
            gold_counts[t.label] += 1

    curves    = []
    best_rows = []

    for lab in SEMEVAL_14:
        gold_count = gold_counts.get(lab, 0)
        if gold_count < args.min_gold:
            continue

        for thr in thresholds:
            pred_by_doc = make_pred_from_probs(probs_df, lab, float(thr))
            p, r, fsc, g, s = pooled_score(gold_by_doc, pred_by_doc, lab)
            curves.append({
                "label": lab, "thr": float(thr),
                "precision": p, "recall": r, "f1": fsc,
                "gold": g, "pred": s
            })

        lab_df = pd.DataFrame([c for c in curves if c["label"] == lab])
        final_thr, strategy = select_threshold(lab_df, gold_count, args.tmin, args.tstep)

        pred_final = make_pred_from_probs(probs_df, lab, final_thr)
        fp, fr, ff, fg, fs = pooled_score(gold_by_doc, pred_final, lab)

        print(f"  {lab:<45}  thr={final_thr:.2f}  F1={ff:.4f}  [{strategy}]")

        best_rows.append({
            "label":          lab,
            "gold":           int(gold_count),
            "best_thr":       final_thr,
            "strategy":       strategy,
            "best_precision": fp,
            "best_recall":    fr,
            "best_f1":        ff,
            "pred_at_best":   int(fs),
        })

        # Per-label plot
        color = STRATEGY_COLORS.get(strategy, "#3498db")
        fig, ax = plt.subplots()
        ax.plot(lab_df["thr"], lab_df["f1"], marker="o", label="F1", zorder=3)
        ax.axvline(x=final_thr, color=color, linestyle="-", linewidth=2,
                   label=f"Final threshold ({final_thr:.2f})  [{strategy}]", zorder=4)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 (overlap-aware, pooled)")
        ax.set_title(f"F1 vs Threshold: {lab}")
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_f1_vs_thr_{lab}.png", dpi=150)
        plt.close(fig)

    curves_df = pd.DataFrame(curves)
    best_df   = pd.DataFrame(best_rows).sort_values("best_f1", ascending=False)

    threshold_dict = {
        row["label"]: float(row["best_thr"])
        for _, row in best_df.iterrows()
    }

    with open("per_label_thresholds.json", "w") as f:
        json.dump(threshold_dict, f, indent=2)

    print("[DONE] wrote per_label_thresholds.json")

    curves_df.to_csv(f"{args.out_prefix}_per_label_curves.csv", index=False)
    best_df.to_csv(  f"{args.out_prefix}_best_thresholds.csv",  index=False)

    # Summary F1 bar chart
    bar_colors = [STRATEGY_COLORS.get(s, "#3498db") for s in best_df["strategy"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(best_df["label"], best_df["best_f1"], color=bar_colors)
    ax.set_xticks(range(len(best_df)))
    ax.set_xticklabels(best_df["label"], rotation=75, ha="right")
    ax.set_ylabel("F1 at Final Threshold")
    ax.set_title(
        "Best overlap-aware F1 by SemEval technique\n"
        "Red = peak  |  Green = elbow  |  Orange = noisy  |  Grey = flat"
    )
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_best_f1_by_label.png", dpi=150)
    plt.close(fig)

    # Summary threshold bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(best_df["label"], best_df["best_thr"], color=bar_colors)
    ax.set_xticks(range(len(best_df)))
    ax.set_xticklabels(best_df["label"], rotation=75, ha="right")
    ax.set_ylabel("Final Threshold")
    ax.set_title(
        "Final thresholds per technique\n"
        "Red = peak  |  Green = elbow  |  Orange = noisy  |  Grey = flat"
    )
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_final_thresholds.png", dpi=150)
    plt.close(fig)

    # Needs work
    needs = best_df[
        (best_df["best_f1"] < args.needs_work_f1) |
        (best_df["best_thr"] >= args.needs_work_thr_hi)
    ].copy()
    needs.to_csv(f"{args.out_prefix}_needs_work.csv", index=False)

    # Final printout
    print("\n" + "=" * 72)
    print("FINAL THRESHOLDS — copy these into predict_v2.py THRESHOLD_OVERRIDES")
    print("=" * 72)
    print(f"{'Label':<45} {'Threshold':>10} {'Strategy':<22} {'F1':>8}")
    print("-" * 72)
    for _, row in best_df.iterrows():
        print(f"  {row['label']:<43} {row['best_thr']:>10.2f}   "
              f"{row['strategy']:<22} {row['best_f1']:>8.4f}")

    print(f"\n[DONE] wrote:")
    print(f"  {args.out_prefix}_per_label_curves.csv")
    print(f"  {args.out_prefix}_best_thresholds.csv")
    print(f"  {args.out_prefix}_needs_work.csv")
    print(f"  {args.out_prefix}_best_f1_by_label.png")
    print(f"  {args.out_prefix}_final_thresholds.png")
    print(f"  per-label curve PNGs with threshold markers")


if __name__ == "__main__":
    main()
