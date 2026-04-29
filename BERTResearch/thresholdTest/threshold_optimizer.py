#!/usr/bin/env python3
"""
threshold_optimizer.py  (REPLACEMENT)

Evaluates your trained DeBERTa model using the proper SemEval-style
overlap-aware scoring — NOT cosine similarity.

The old threshold_optimizer.py measured how well all-MiniLM cosine
similarity detected propaganda.  That was evaluating the wrong model.
This version evaluates your actual DeBERTa predictions against gold.csv.

What it does
------------
1. Loads sentence_probs.csv  (from export_pred_sentence_fragments.py)
   and gold.csv               (from semeval_extract_with_gold.py)
2. Sweeps thresholds per label using overlap-aware P/R/F1
   (same metric as threshold_sweep_semeval.py but with richer output)
3. Computes overall model health diagnostics:
   - Which techniques are well-calibrated vs broken
   - Label-level confusion (what is getting confused with what)
   - Confidence distribution for TP vs FP predictions
4. Produces:
   - threshold_optimizer_best.csv      — per-label best threshold + scores
   - threshold_optimizer_report.txt    — human-readable diagnostic
   - threshold_optimizer_heatmap.png   — F1 heatmap: technique × threshold
   - threshold_optimizer_calibration.png — TP/FP confidence distributions

Usage
-----
    python threshold_optimizer.py \\
        --gold gold.csv \\
        --sentence_probs sentence_probs.csv \\
        --out_prefix threshold_optimizer

    # optionally write final per-label thresholds for predict_v2.py:
        --save_thresholds ./deberta-propaganda-multilabel/per_label_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# optional visualisation — skip gracefully if matplotlib not available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found — skipping plots")

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

SHORT_NAMES = {
    "appeal_to_authority":                "Auth.",
    "appeal_to_fear_prejudice":           "Fear",
    "bandwagon_reductio_ad_hitlerum":     "Band./Reductio",
    "black_and_white_fallacy":            "B&W",
    "causal_oversimplification":          "CausalSimp.",
    "doubt":                              "Doubt",
    "exaggeration_minimisation":          "Exagg.",
    "flag_waving":                        "Flag",
    "loaded_language":                    "Loaded",
    "name_calling_labeling":              "NameCall.",
    "repetition":                         "Repet.",
    "slogans":                            "Slogans",
    "thought_terminating_cliches":        "Thought-term.",
    "whataboutism_straw_men_red_herring": "Whatabout.",
}


# ===================== FRAGMENT SCORING (inline, no import) =====================

@dataclass(frozen=True)
class Frag:
    start: int
    end: int
    label: str

    def length(self) -> int:
        return max(0, self.end - self.start)


def _overlap(a: Frag, b: Frag) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _C(s: Frag, t: Frag) -> float:
    h = s.length()
    if h == 0 or s.label != t.label:
        return 0.0
    return _overlap(s, t) / h


def _precision(S: List[Frag], T: List[Frag]) -> float:
    if not S:
        return 0.0
    return sum(_C(s, t) for s in S for t in T) / len(S)


def _recall(S: List[Frag], T: List[Frag]) -> float:
    if not T:
        return 0.0
    return sum(_C(s, t) for s in S for t in T) / len(T)


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _score_label(gold_by_doc, pred_by_doc, label: str) -> Tuple[float, float, float, int, int]:
    """Return (precision, recall, f1, n_gold, n_pred) pooled across docs."""
    T = [f for fs in gold_by_doc.values() for f in fs if f.label == label]
    S = [f for fs in pred_by_doc.values() for f in fs if f.label == label]
    return _precision(S, T), _recall(S, T), _f1(_precision(S, T), _recall(S, T)), len(T), len(S)


# ===================== LOADING =====================

def load_gold(gold_csv: str) -> Dict[str, List[Frag]]:
    df = pd.read_csv(gold_csv)
    by_doc: Dict[str, List[Frag]] = defaultdict(list)
    for _, r in df.iterrows():
        by_doc[str(r["doc_id"])].append(
            Frag(int(r["start"]), int(r["end"]), str(r["label"]))
        )
    return by_doc


def make_pred(probs_df: pd.DataFrame, label: str, thr: float) -> Dict[str, List[Frag]]:
    by_doc: Dict[str, List[Frag]] = defaultdict(list)
    for _, r in probs_df[probs_df[label] >= thr].iterrows():
        by_doc[str(r["doc_id"])].append(
            Frag(int(r["start"]), int(r["end"]), label)
        )
    return by_doc


# ===================== THRESHOLD SWEEP =====================

def sweep(gold_by_doc, probs_df, label: str,
          thresholds: np.ndarray) -> pd.DataFrame:
    """Full P/R/F1 curve for one label across all thresholds."""
    rows = []
    for thr in thresholds:
        pred_by_doc = make_pred(probs_df, label, thr)
        T_all = [f for fs in gold_by_doc.values() for f in fs if f.label == label]
        S_all = [f for fs in pred_by_doc.values() for f in fs if f.label == label]
        p = _precision(S_all, T_all)
        r = _recall(S_all, T_all)
        rows.append({
            "label":     label,
            "threshold": round(float(thr), 3),
            "precision": p,
            "recall":    r,
            "f1":        _f1(p, r),
            "n_gold":    len(T_all),
            "n_pred":    len(S_all),
        })
    return pd.DataFrame(rows)


# ===================== DIAGNOSTICS =====================

def diagnose_label(label: str, curve: pd.DataFrame) -> dict:
    """
    Classify a label's behaviour:
      - 'good'        : best_f1 >= 0.40
      - 'low_recall'  : best precision ok but recall never gets above 0.25
      - 'low_prec'    : recall ok but precision < 0.30 at best_f1 threshold
      - 'broken'      : best_f1 < 0.10 — model has no signal for this technique
      - 'threshold_extreme' : best threshold < 0.10 or > 0.90
    """
    best_row = curve.loc[curve["f1"].idxmax()]
    bf1   = best_row["f1"]
    bp    = best_row["precision"]
    br    = best_row["recall"]
    bthr  = best_row["threshold"]
    ng    = int(best_row["n_gold"])

    if ng < 10:
        status = "too_few_gold"
    elif bf1 >= 0.40:
        status = "good"
    elif bf1 < 0.10:
        status = "broken"
    elif br < 0.25:
        status = "low_recall"
    elif bp < 0.30:
        status = "low_precision"
    elif bthr <= 0.10 or bthr >= 0.90:
        status = "threshold_extreme"
    else:
        status = "mediocre"

    return {
        "label":      label,
        "status":     status,
        "best_f1":    round(bf1, 4),
        "best_p":     round(bp, 4),
        "best_r":     round(br, 4),
        "best_thr":   round(bthr, 3),
        "n_gold":     ng,
        "action":     _recommend(status, label),
    }


def _recommend(status: str, label: str) -> str:
    recs = {
        "good":            "No action needed.",
        "broken":          "Model has no signal. Need more training data for this technique.",
        "low_recall":      "Threshold too high or too few positive training examples. "
                           "Lower threshold or add more examples.",
        "low_precision":   "Too many false positives. "
                           "Raise threshold or add hard negatives for this technique.",
        "mediocre":        "Marginal performance. More targeted training data will help.",
        "threshold_extreme": "Optimal threshold is at an extreme. "
                             "Model may be miscalibrated. Check training label distribution.",
        "too_few_gold":    "Not enough gold examples to evaluate reliably (< 10).",
    }
    return recs.get(status, "Unknown status.")


# ===================== CONFIDENCE DISTRIBUTION =====================

def confidence_distribution(probs_df: pd.DataFrame,
                             gold_by_doc: Dict[str, List[Frag]],
                             label: str,
                             best_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a given label, split predicted spans into TP and FP
    and return their confidence scores.
    """
    gold_spans = {
        (doc_id, f.start, f.end)
        for doc_id, frags in gold_by_doc.items()
        for f in frags if f.label == label
    }

    tp_confs, fp_confs = [], []
    for _, r in probs_df[probs_df[label] >= best_thr].iterrows():
        conf     = float(r[label])
        doc_id   = str(r["doc_id"])
        start    = int(r["start"])
        end      = int(r["end"])

        # Check for any overlap with a gold span of same label
        is_tp = any(
            max(start, gs) < min(end, ge)
            for (gd, gs, ge) in gold_spans if gd == doc_id
        )
        (tp_confs if is_tp else fp_confs).append(conf)

    return np.array(tp_confs), np.array(fp_confs)


# ===================== PLOTTING =====================

def plot_heatmap(f1_matrix: np.ndarray, labels: list,
                 thresholds: np.ndarray, out_path: str) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(f1_matrix, aspect="auto", vmin=0, vmax=1,
                   cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([SHORT_NAMES.get(l, l) for l in labels], fontsize=9)
    ax.set_xlabel("Threshold")
    ax.set_title("Overlap-aware F1 by Technique × Threshold\n"
                 "(green=good, red=bad)")
    plt.colorbar(im, ax=ax, label="F1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] {out_path}")


def plot_calibration(probs_df, gold_by_doc, labels, best_thrs, out_path):
    if not HAS_MPL:
        return
    n = len(labels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()

    for i, lab in enumerate(labels):
        thr = best_thrs.get(lab, 0.5)
        tp_c, fp_c = confidence_distribution(probs_df, gold_by_doc, lab, thr)
        ax = axes[i]
        bins = np.linspace(0, 1, 20)
        if len(tp_c):
            ax.hist(tp_c, bins=bins, alpha=0.6, color="green", label="TP")
        if len(fp_c):
            ax.hist(fp_c, bins=bins, alpha=0.6, color="red",   label="FP")
        ax.axvline(thr, color="black", linestyle="--", linewidth=1)
        ax.set_title(SHORT_NAMES.get(lab, lab), fontsize=8)
        ax.set_xlim(0, 1)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confidence Distributions: TP (green) vs FP (red)\n"
                 "dashed line = best threshold", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] {out_path}")


# ===================== MAIN =====================

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate DeBERTa propaganda model — threshold optimisation and diagnostics"
    )
    ap.add_argument("--gold",            required=True,
                    help="gold.csv from semeval_extract_with_gold.py")
    ap.add_argument("--sentence_probs",  required=True,
                    help="sentence_probs.csv from export_pred_sentence_fragments.py")
    ap.add_argument("--out_prefix",      default="threshold_optimizer")
    ap.add_argument("--tmin",            type=float, default=0.05)
    ap.add_argument("--tmax",            type=float, default=0.95)
    ap.add_argument("--tstep",           type=float, default=0.05)
    ap.add_argument("--min_gold",        type=int,   default=5,
                    help="Skip label in sweep if fewer than this many gold fragments")
    ap.add_argument("--save_thresholds", default=None,
                    help="If set, write best thresholds to this JSON path "
                         "(used by predict_v2.py)")
    args = ap.parse_args()

    os.makedirs(Path(args.out_prefix).parent, exist_ok=True) if Path(args.out_prefix).parent != Path(".") else None

    print("=" * 70)
    print("THRESHOLD OPTIMIZER  —  DeBERTa model evaluation")
    print("=" * 70)

    gold_by_doc = load_gold(args.gold)
    probs_df    = pd.read_csv(args.sentence_probs)

    # Validate columns
    missing = [l for l in SEMEVAL_14 if l not in probs_df.columns]
    if missing:
        raise ValueError(
            f"sentence_probs.csv is missing SemEval-14 columns: {missing}\n"
            "Re-run export_pred_sentence_fragments.py."
        )

    thresholds = np.arange(args.tmin, args.tmax + 1e-9, args.tstep)

    # ---- Gold support ----
    gold_counts: dict[str, int] = defaultdict(int)
    for fs in gold_by_doc.values():
        for f in fs:
            gold_counts[f.label] += 1

    # ---- Per-label sweep ----
    all_curves: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    best_thrs: dict[str, float] = {}
    f1_matrix = np.zeros((len(SEMEVAL_14), len(thresholds)))

    print(f"\nSweeping {len(thresholds)} thresholds × {len(SEMEVAL_14)} techniques...")

    for li, lab in enumerate(SEMEVAL_14):
        ng = gold_counts.get(lab, 0)
        if ng < args.min_gold:
            print(f"  {lab:<43}  SKIP (only {ng} gold fragments)")
            best_thrs[lab] = 0.5
            continue

        curve = sweep(gold_by_doc, probs_df, lab, thresholds)
        all_curves.append(curve)

        f1_matrix[li, :] = curve["f1"].values

        diag = diagnose_label(lab, curve)
        diagnostics.append(diag)
        best_thrs[lab] = diag["best_thr"]

        status_str = f"[{diag['status'].upper():>18}]"
        print(
            f"  {lab:<43}  F1={diag['best_f1']:.3f}  "
            f"P={diag['best_p']:.3f}  R={diag['best_r']:.3f}  "
            f"thr={diag['best_thr']:.2f}  {status_str}"
        )

    # ---- Save curves ----
    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    curves_df.to_csv(f"{args.out_prefix}_curves.csv", index=False)

    # ---- Save best thresholds ----
    diag_df = pd.DataFrame(diagnostics).sort_values("best_f1", ascending=False)
    diag_df.to_csv(f"{args.out_prefix}_best.csv", index=False)

    if args.save_thresholds:
        with open(args.save_thresholds, "w") as fh:
            json.dump(best_thrs, fh, indent=2)
        print(f"\n[SAVED] Per-label thresholds → {args.save_thresholds}")

    # ---- Overall model health summary ----
    status_counts: dict[str, int] = defaultdict(int)
    for d in diagnostics:
        status_counts[d["status"]] += 1

    macro_f1 = np.mean([d["best_f1"] for d in diagnostics]) if diagnostics else 0.0
    weighted_f1 = (
        np.average(
            [d["best_f1"] for d in diagnostics],
            weights=[max(1, d["n_gold"]) for d in diagnostics],
        )
        if diagnostics else 0.0
    )

    # ---- Text report ----
    report_lines = [
        "=" * 70,
        "DEBERTA PROPAGANDA MODEL — DIAGNOSTIC REPORT",
        "=" * 70,
        "",
        f"Gold file       : {args.gold}",
        f"Probs file      : {args.sentence_probs}",
        f"Threshold range : {args.tmin:.2f} – {args.tmax:.2f}  step={args.tstep}",
        "",
        "OVERALL SCORES",
        "-" * 40,
        f"  Macro F1 (over evaluated labels)    : {macro_f1:.4f}",
        f"  Weighted F1 (by gold count)         : {weighted_f1:.4f}",
        "",
        "STATUS BREAKDOWN",
        "-" * 40,
    ]
    for status, count in sorted(status_counts.items()):
        report_lines.append(f"  {status:<22}: {count}")

    report_lines += [
        "",
        "PER-LABEL RESULTS",
        "-" * 40,
        f"{'Label':<45} {'F1':>6} {'P':>6} {'R':>6} {'thr':>5} {'gold':>6}  Status / Action",
        "-" * 115,
    ]
    for d in sorted(diagnostics, key=lambda x: x["best_f1"], reverse=True):
        report_lines.append(
            f"  {d['label']:<43} {d['best_f1']:>6.3f} {d['best_p']:>6.3f} "
            f"{d['best_r']:>6.3f} {d['best_thr']:>5.2f} {d['n_gold']:>6}  "
            f"[{d['status']}]  {d['action']}"
        )

    report_lines += [
        "",
        "LABELS NEEDING ATTENTION",
        "-" * 40,
    ]
    needs_work = [d for d in diagnostics if d["status"] not in ("good", "too_few_gold")]
    if needs_work:
        for d in sorted(needs_work, key=lambda x: x["best_f1"]):
            report_lines.append(f"\n  {d['label']}  (F1={d['best_f1']:.3f})")
            report_lines.append(f"    Status : {d['status']}")
            report_lines.append(f"    Action : {d['action']}")
    else:
        report_lines.append("  None — all evaluated labels are in good shape!")

    report_lines += ["", "=" * 70]
    report_text = "\n".join(report_lines)

    report_path = f"{args.out_prefix}_report.txt"
    Path(report_path).write_text(report_text, encoding="utf-8")
    print(f"\n[SAVED] Report → {report_path}")

    # Print key sections to stdout
    print("\n" + "=" * 70)
    print("MODEL HEALTH SUMMARY")
    print("=" * 70)
    print(f"  Macro F1             : {macro_f1:.4f}")
    print(f"  Weighted F1          : {weighted_f1:.4f}")
    print(f"  Good techniques      : {status_counts.get('good', 0)}")
    print(f"  Broken techniques    : {status_counts.get('broken', 0)}")
    print(f"  Mediocre techniques  : {status_counts.get('mediocre', 0)}")
    if needs_work:
        print(f"\n  Techniques needing attention:")
        for d in sorted(needs_work, key=lambda x: x["best_f1"])[:5]:
            print(f"    {d['label']:<43}  F1={d['best_f1']:.3f}  [{d['status']}]")

    # ---- Plots ----
    plot_heatmap(
        f1_matrix, SEMEVAL_14, thresholds,
        f"{args.out_prefix}_heatmap.png"
    )
    plot_calibration(
        probs_df, gold_by_doc, SEMEVAL_14, best_thrs,
        f"{args.out_prefix}_calibration.png"
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"  {args.out_prefix}_best.csv         — per-label best threshold + F1")
    print(f"  {args.out_prefix}_curves.csv       — full P/R/F1 curve per label")
    print(f"  {args.out_prefix}_report.txt       — human-readable diagnostics")
    if HAS_MPL:
        print(f"  {args.out_prefix}_heatmap.png      — F1 heatmap")
        print(f"  {args.out_prefix}_calibration.png  — TP/FP confidence distributions")
    if args.save_thresholds:
        print(f"  {args.save_thresholds}  — thresholds for predict_v2.py")


if __name__ == "__main__":
    main()