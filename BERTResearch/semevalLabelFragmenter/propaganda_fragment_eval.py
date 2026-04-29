#!/usr/bin/env python3
"""
propaganda_fragment_eval.py

Fragment-level evaluation for propaganda technique detection with PARTIAL OVERLAP credit.

Implements the overlap-aware precision/recall/F1 described in:
Da San Martino et al. (2019) "Fine-Grained Analysis of Propaganda in News Articles"

Expected CSV formats (UTF-8, header row):
  gold.csv: doc_id,label,start,end
  pred.csv: doc_id,label,start,end

Offsets are integer character offsets in the original document, half-open intervals [start,end).

Notes:
- Default follows paper's double-sum definition for P/R (can double-count if many overlaps).
- Pass --one_to_one for greedy 1-1 matching by max credit to reduce double counting.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Frag:
    start: int
    end: int
    label: str

    def length(self) -> int:
        return max(0, self.end - self.start)


def _overlap_len(a: Frag, b: Frag) -> int:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    return max(0, hi - lo)


def C(s: Frag, t: Frag, h: int) -> float:
    if h <= 0:
        return 0.0
    if s.label != t.label:
        return 0.0
    return _overlap_len(s, t) / float(h)


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


def _load_frags_csv(path: str) -> Dict[str, List[Frag]]:
    by_doc: Dict[str, List[Frag]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"doc_id", "label", "start", "end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            doc_id = str(row["doc_id"]).strip()
            label = str(row["label"]).strip()
            if not doc_id or not label:
                continue
            try:
                start = int(row["start"])
                end = int(row["end"])
            except Exception as e:
                raise ValueError(f"Bad start/end in row: {row}") from e

            if end < start:
                start, end = end, start
            by_doc[doc_id].append(Frag(start=start, end=end, label=label))

    for d in by_doc:
        by_doc[d].sort(key=lambda x: (x.start, x.end, x.label))
    return by_doc


def _one_to_one_score(S: List[Frag], T: List[Frag], mode: str) -> float:
    """
    Greedy 1-to-1 matching by max overlap credit to reduce double counting.
    """
    if mode not in {"precision", "recall"}:
        raise ValueError("mode must be precision or recall")

    if mode == "precision":
        if not S:
            return 0.0
        edges: List[Tuple[float, int, int]] = []
        for i, s in enumerate(S):
            h = s.length()
            if h == 0:
                continue
            for j, t in enumerate(T):
                if s.label != t.label:
                    continue
                cred = C(s, t, h)
                if cred > 0:
                    edges.append((cred, i, j))
        edges.sort(reverse=True, key=lambda x: x[0])
        used_s, used_t = set(), set()
        total = 0.0
        for cred, i, j in edges:
            if i in used_s or j in used_t:
                continue
            used_s.add(i)
            used_t.add(j)
            total += cred
        return total / float(len(S))

    else:
        if not T:
            return 0.0
        edges: List[Tuple[float, int, int]] = []
        for j, t in enumerate(T):
            h = t.length()
            if h == 0:
                continue
            for i, s in enumerate(S):
                if s.label != t.label:
                    continue
                cred = C(s, t, h)
                if cred > 0:
                    edges.append((cred, i, j))
        edges.sort(reverse=True, key=lambda x: x[0])
        used_s, used_t = set(), set()
        total = 0.0
        for cred, i, j in edges:
            if i in used_s or j in used_t:
                continue
            used_s.add(i)
            used_t.add(j)
            total += cred
        return total / float(len(T))


def evaluate(gold_by_doc: Dict[str, List[Frag]],
             pred_by_doc: Dict[str, List[Frag]],
             one_to_one: bool = False) -> Dict[str, float]:
    all_docs = sorted(set(gold_by_doc.keys()) | set(pred_by_doc.keys()))

    # Macro across docs (average per-doc P and R)
    ps, rs = [], []
    for d in all_docs:
        T = gold_by_doc.get(d, [])
        S = pred_by_doc.get(d, [])
        if one_to_one:
            p = _one_to_one_score(S, T, "precision")
            r = _one_to_one_score(S, T, "recall")
        else:
            p = precision(S, T)
            r = recall(S, T)
        ps.append(p)
        rs.append(r)

    macro_p = sum(ps) / len(ps) if ps else 0.0
    macro_r = sum(rs) / len(rs) if rs else 0.0
    macro_f = f1(macro_p, macro_r)

    # Pooled (micro-ish)
    pooled_T = [t for d in all_docs for t in gold_by_doc.get(d, [])]
    pooled_S = [s for d in all_docs for s in pred_by_doc.get(d, [])]
    if one_to_one:
        pooled_p = _one_to_one_score(pooled_S, pooled_T, "precision")
        pooled_r = _one_to_one_score(pooled_S, pooled_T, "recall")
    else:
        pooled_p = precision(pooled_S, pooled_T)
        pooled_r = recall(pooled_S, pooled_T)
    pooled_f = f1(pooled_p, pooled_r)

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "pooled_precision": pooled_p,
        "pooled_recall": pooled_r,
        "pooled_f1": pooled_f,
        "num_docs": float(len(all_docs)),
        "num_gold_frags": float(len(pooled_T)),
        "num_pred_frags": float(len(pooled_S)),
    }


def evaluate_per_label(gold_by_doc: Dict[str, List[Frag]],
                       pred_by_doc: Dict[str, List[Frag]],
                       one_to_one: bool = False) -> Dict[str, Dict[str, float]]:
    labels = set()
    for Ts in gold_by_doc.values():
        labels.update(t.label for t in Ts)
    for Ss in pred_by_doc.values():
        labels.update(s.label for s in Ss)

    out: Dict[str, Dict[str, float]] = {}
    for lab in sorted(labels):
        pooled_T = [t for Ts in gold_by_doc.values() for t in Ts if t.label == lab]
        pooled_S = [s for Ss in pred_by_doc.values() for s in Ss if s.label == lab]
        if one_to_one:
            p = _one_to_one_score(pooled_S, pooled_T, "precision")
            r = _one_to_one_score(pooled_S, pooled_T, "recall")
        else:
            p = precision(pooled_S, pooled_T)
            r = recall(pooled_S, pooled_T)
        out[lab] = {
            "precision": p,
            "recall": r,
            "f1": f1(p, r),
            "gold": float(len(pooled_T)),
            "pred": float(len(pooled_S)),
        }
    return out


def _print_per_label(per_label: Dict[str, Dict[str, float]], min_gold: int = 0) -> None:
    rows = []
    for lab, m in per_label.items():
        if int(m["gold"]) < min_gold:
            continue
        rows.append((lab, m["precision"], m["recall"], m["f1"], int(m["gold"]), int(m["pred"])))
    rows.sort(key=lambda x: x[3], reverse=True)

    print("\nPER-LABEL (pooled)".ljust(80, "-"))
    print(f"{'label':32} {'P':>8} {'R':>8} {'F1':>8} {'gold':>7} {'pred':>7}")
    for lab, p, r, f, g, s in rows:
        print(f"{lab:32} {p:8.4f} {r:8.4f} {f:8.4f} {g:7d} {s:7d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Gold CSV: doc_id,label,start,end")
    ap.add_argument("--pred", required=True, help="Pred CSV: doc_id,label,start,end")
    ap.add_argument("--one_to_one", action="store_true",
                    help="Greedy 1-1 matching to reduce double-counting overlaps.")
    ap.add_argument("--min_gold", type=int, default=0,
                    help="Only show per-label rows with at least this many gold instances.")
    args = ap.parse_args()

    gold_by_doc = _load_frags_csv(args.gold)
    pred_by_doc = _load_frags_csv(args.pred)

    overall = evaluate(gold_by_doc, pred_by_doc, one_to_one=args.one_to_one)
    per_label = evaluate_per_label(gold_by_doc, pred_by_doc, one_to_one=args.one_to_one)

    print("OVERALL".ljust(80, "-"))
    for k in ["macro_precision", "macro_recall", "macro_f1",
                "pooled_precision", "pooled_recall", "pooled_f1"]:
        print(f"{k:20s}: {overall[k]:.4f}")
    print(f"{'num_docs':20s}: {int(overall['num_docs'])}")
    print(f"{'num_gold_frags':20s}: {int(overall['num_gold_frags'])}")
    print(f"{'num_pred_frags':20s}: {int(overall['num_pred_frags'])}")

    _print_per_label(per_label, min_gold=args.min_gold)


if __name__ == "__main__":
    main()
