#!/usr/bin/env python3
"""
semeval_extract_with_gold.py

Run from: BERTResearch/semevalLabelFragmenter/

Reads raw SemEval-2020 Task 11 files and writes three output CSVs
to BERTResearch/ (one level up).

Outputs (all in BERTResearch/):
  ../semeval_examples.csv      — doc_id, label_raw, label, start, end, text_snippet
  ../gold.csv                  — doc_id, label, start, end  (SemEval-14 canonical labels)
  ../semeval_input_texts.csv   — doc_id, text  (ALL articles, for inference)

Example:
  cd BERTResearch/semevalLabelFragmenter
  python semeval_extract_with_gold.py \\
      --label_file  ../semeval_datasets/train-task2-TC.labels \\
      --articles_dir ../semeval_datasets/train-articles

The SemEval dataset can be downloaded from:
  https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2020_task_11
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Canonical SemEval-14 label set (normalised)
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

# Raw SemEval label string → canonical normalised form
RAW_TO_CANON = {
    "Appeal_to_Authority":                "appeal_to_authority",
    "Appeal_to_fear-prejudice":           "appeal_to_fear_prejudice",
    "Bandwagon,Reductio_ad_hitlerum":     "bandwagon_reductio_ad_hitlerum",
    "Black-and-White_Fallacy":            "black_and_white_fallacy",
    "Causal_Oversimplification":          "causal_oversimplification",
    "Doubt":                              "doubt",
    "Exaggeration,Minimisation":          "exaggeration_minimisation",
    "Flag-Waving":                        "flag_waving",
    "Loaded_Language":                    "loaded_language",
    "Name_Calling,Labeling":              "name_calling_labeling",
    "Repetition":                         "repetition",
    "Slogans":                            "slogans",
    "Thought-terminating_Cliches":        "thought_terminating_cliches",
    "Whataboutism,Straw_Men,Red_Herring": "whataboutism_straw_men_red_herring",
}


def to_canon(label_raw: str) -> str:
    label_raw = label_raw.strip()
    if label_raw in RAW_TO_CANON:
        return RAW_TO_CANON[label_raw]
    # Fallback normalisation (rarely needed for clean SemEval data)
    return (
        label_raw.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(",", "")
        .strip()
    )


def main():
    ap = argparse.ArgumentParser(
        description="Extract SemEval-2020 Task 11 spans and write gold/example CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run from BERTResearch/semevalLabelFragmenter/:
  python semeval_extract_with_gold.py \\
      --label_file  ../semeval_datasets/train-task2-TC.labels \\
      --articles_dir ../semeval_datasets/train-articles
        """
    )
    ap.add_argument(
        "--label_file", required=True,
        help="Path to train-task2-TC.labels (tab-separated: article_id, label, start, end)",
    )
    ap.add_argument(
        "--articles_dir", required=True,
        help="Directory containing article<ID>.txt files",
    )
    # Output paths default to BERTResearch/ (one level up from this script's location)
    ap.add_argument("--out_examples",     default="../semeval_examples.csv")
    ap.add_argument("--out_gold",         default="../gold.csv")
    ap.add_argument("--out_input_texts",  default="../semeval_input_texts.csv")
    args = ap.parse_args()

    label_path   = Path(args.label_file)
    articles_dir = Path(args.articles_dir)

    if not label_path.exists():
        raise FileNotFoundError(f"[ERROR] Label file not found: {label_path}")
    if not articles_dir.exists():
        raise FileNotFoundError(f"[ERROR] Articles directory not found: {articles_dir}")

    # ---- Load span labels ----
    labels = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            doc_id, lab_raw, start, end = parts
            labels.append({
                "doc_id":    doc_id.strip(),
                "label_raw": lab_raw.strip(),
                "label":     to_canon(lab_raw),
                "start":     int(start),
                "end":       int(end),
            })
    print(f"[INFO] Loaded {len(labels):,} spans from {label_path}")

    # ---- Extract text snippets ----
    text_cache: dict = {}
    rows = []
    missing = 0

    for it in labels:
        doc_id = it["doc_id"]
        p = articles_dir / f"article{doc_id}.txt"
        if not p.exists():
            missing += 1
            continue
        if doc_id not in text_cache:
            text_cache[doc_id] = p.read_text(encoding="utf-8")

        text = text_cache[doc_id]
        s, e = it["start"], it["end"]
        if e < s:
            s, e = e, s
        snippet = text[s:e].replace("\n", " ").strip()

        rows.append({
            "doc_id":       doc_id,
            "label_raw":    it["label_raw"],
            "label":        it["label"],
            "start":        s,
            "end":          e,
            "text_snippet": snippet,
        })

    if missing:
        print(f"[WARNING] {missing} spans skipped — article file not found")

    # ---- Write semeval_examples.csv ----
    df_ex = pd.DataFrame(rows)
    out_ex = Path(args.out_examples)
    out_ex.parent.mkdir(parents=True, exist_ok=True)
    df_ex.to_csv(out_ex, index=False, encoding="utf-8")
    print(f"[DONE] {out_ex}  ({len(df_ex):,} rows)")

    # ---- Write gold.csv ----
    df_gold = df_ex[["doc_id", "label", "start", "end"]].copy()
    bad = sorted(set(df_gold["label"]) - set(SEMEVAL_14))
    if bad:
        raise ValueError(
            f"[ERROR] gold.csv contains labels not in SemEval-14: {bad}\n"
            "Check RAW_TO_CANON mapping in this script."
        )
    out_gold = Path(args.out_gold)
    out_gold.parent.mkdir(parents=True, exist_ok=True)
    df_gold.to_csv(out_gold, index=False, encoding="utf-8")
    print(f"[DONE] {out_gold}  ({len(df_gold):,} rows)")

    # ---- Write semeval_input_texts.csv (ALL articles for inference) ----
    texts = []
    for p in sorted(articles_dir.glob("article*.txt")):
        doc_id = p.stem.replace("article", "")
        texts.append({"doc_id": doc_id, "text": p.read_text(encoding="utf-8")})
    df_text = pd.DataFrame(texts)
    out_txt = Path(args.out_input_texts)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    df_text.to_csv(out_txt, index=False, encoding="utf-8")
    print(f"[DONE] {out_txt}  ({len(df_text):,} articles)")

    print(f"\n[NEXT] Run from BERTResearch/:")
    print(f"  python semeval_data_processor.py")


if __name__ == "__main__":
    main()