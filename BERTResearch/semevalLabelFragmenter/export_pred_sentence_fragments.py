#!/usr/bin/env python3
"""
export_pred_sentence_fragments.py (SemEval-14 label space)

Inputs:
  --input_texts: CSV with columns doc_id,text (ALL articles)
  --model_dir: HF model directory with tokenizer + model + label_mapping.json

Outputs:
  - sentence_probs.csv: one row per sentence with char offsets + SemEval-14 probs
      columns: doc_id,sent_id,start,end,<14 labels...>

Optional:
  - pred.csv if --write_pred is set, using a chosen threshold (mainly for quick checks)

Key feature:
  If the model label space != SemEval-14, we map/merge model probabilities into SemEval-14.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_SENT_RE = re.compile(r"[^.!?]+[.!?]?", re.MULTILINE)

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


# ===================== RoRA ADAPTER LOADING =====================

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
                target_layers=list(range(12, 24)),
                target_modules=["query_proj", "key_proj", "value_proj", "dense"]):
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


def load_model_with_rora(model_dir: Path, device: torch.device):
    """Load model and inject RoRA adapter weights if available."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    rora_path = model_dir / "rora_adapters.pt"
    if rora_path.exists():
        inject_rora(model)
        state = torch.load(rora_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        adapter_keys = [k for k in state if "lora" in k]
        print(f"[INFO] Loaded {len(adapter_keys)} RoRA adapter tensors from {rora_path}")
    else:
        print(f"[WARNING] rora_adapters.pt not found at {rora_path} — sentence probs will be inaccurate")

    model.eval()
    return tokenizer, model


# ===================== HELPERS =====================

def sentence_spans(text: str, min_words: int = 5) -> List[Tuple[str, int, int]]:
    spans = []
    for m in _SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        sent = text[s:e].strip()
        if not sent:
            continue
        if len(sent.split()) < min_words:
            continue
        spans.append((sent, s, e))
    return spans


def load_label_mapping(model_dir: Path) -> Dict[int, str]:
    p = model_dir / "label_mapping.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Your training must save label_mapping.json.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in raw.items()}


def norm_label(x: str) -> str:
    return x.strip().lower().replace("-", "_").replace(" ", "_").replace(",", "")


def union_prob(ps: List[float]) -> float:
    """Probability of union assuming independence-ish: 1 - Π(1-p)."""
    prod = 1.0
    for p in ps:
        prod *= (1.0 - float(p))
    return 1.0 - prod


def build_model_to_semeval_map(model_labels: List[str]) -> Dict[str, List[str]]:
    """
    Returns: semeval_label -> list of model_label(s) to union/merge into it.
    If model already uses SemEval-14, mapping becomes identity.
    """
    model_norm = {norm_label(l) for l in model_labels}

    if set(SEMEVAL_14).issubset(model_norm):
        return {lab: [lab] for lab in SEMEVAL_14}

    aliases = {
        "appeal_to_fear_prejudice": ["appeal_to_fear_prejudice", "appeal_to_fear", "appeal_to_fear/prejudice"],
        "name_calling_labeling": ["name_calling_labeling", "name_calling", "labeling"],
        "exaggeration_minimisation": ["exaggeration_minimisation", "exaggeration", "minimization", "minimisation"],
        "bandwagon": ["bandwagon"],
        "reductio_ad_hitlerum": ["reductio_ad_hitlerum"],
        "whataboutism": ["whataboutism"],
        "straw_man": ["straw_man", "strawman"],
        "red_herring": ["red_herring"],
        "black_and_white_fallacy": ["black_and_white_fallacy", "black_and_white", "blackwhite_fallacy", "dictatorship"],
        "appeal_to_authority": ["appeal_to_authority", "authority"],
        "causal_oversimplification": ["causal_oversimplification", "oversimplification", "scapegoating"],
        "thought_terminating_cliches": ["thought_terminating_cliches", "thought_terminating_cliche", "thought_terminating"],
        "flag_waving": ["flag_waving"],
        "loaded_language": ["loaded_language"],
        "repetition": ["repetition"],
        "slogans": ["slogans", "slogan"],
        "doubt": ["doubt"],
    }

    def present(alias_list: List[str]) -> List[str]:
        out = []
        for a in alias_list:
            a_norm = norm_label(a)
            if a_norm in model_norm:
                out.append(a_norm)
        return out

    mapping: Dict[str, List[str]] = {}
    mapping["appeal_to_authority"] = present(aliases["appeal_to_authority"])
    mapping["appeal_to_fear_prejudice"] = present(aliases["appeal_to_fear_prejudice"])
    mapping["black_and_white_fallacy"] = present(aliases["black_and_white_fallacy"])
    mapping["causal_oversimplification"] = present(aliases["causal_oversimplification"])
    mapping["doubt"] = present(aliases["doubt"])
    mapping["exaggeration_minimisation"] = present(aliases["exaggeration_minimisation"])
    mapping["flag_waving"] = present(aliases["flag_waving"])
    mapping["loaded_language"] = present(aliases["loaded_language"])
    mapping["name_calling_labeling"] = present(aliases["name_calling_labeling"])
    mapping["repetition"] = present(aliases["repetition"])
    mapping["slogans"] = present(aliases["slogans"])
    mapping["thought_terminating_cliches"] = present(aliases["thought_terminating_cliches"])

    bw = present(aliases["bandwagon"])
    rah = present(aliases["reductio_ad_hitlerum"])
    mapping["bandwagon_reductio_ad_hitlerum"] = list(dict.fromkeys(bw + rah))

    w = present(aliases["whataboutism"])
    s = present(aliases["straw_man"])
    r = present(aliases["red_herring"])
    mapping["whataboutism_straw_men_red_herring"] = list(dict.fromkeys(w + s + r))

    return mapping


# ===================== MAIN =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_texts", required=True, help="CSV with columns doc_id,text")
    ap.add_argument("--model_dir", required=True, help="Trained model directory")
    ap.add_argument("--out_sentence_probs", default="sentence_probs.csv",
                    help="Output sentence probs CSV in SemEval-14 label space")
    ap.add_argument("--write_pred", action="store_true",
                    help="If set, also write pred.csv using --threshold (SemEval-14 labels)")
    ap.add_argument("--out_pred", default="pred.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--min_words", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    idx_to_label = load_label_mapping(model_dir)

    model_labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    model_labels_norm = [norm_label(x) for x in model_labels]
    label_to_index = {norm_label(lab): i for i, lab in enumerate(model_labels)}

    semeval_map = build_model_to_semeval_map(model_labels_norm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with RoRA adapters
    tokenizer, model = load_model_with_rora(model_dir, device)

    df = pd.read_csv(args.input_texts)
    if "doc_id" not in df.columns or "text" not in df.columns:
        raise ValueError("input_texts must contain columns: doc_id,text")

    prob_rows = []
    pred_rows = []

    for _, r in df.iterrows():
        doc_id = str(r["doc_id"])
        text = str(r["text"])
        sents = sentence_spans(text, min_words=args.min_words)
        if not sents:
            continue

        for i in range(0, len(sents), args.batch_size):
            chunk = sents[i:i + args.batch_size]
            texts = [c[0] for c in chunk]
            offsets = [(c[1], c[2]) for c in chunk]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()

            for b in range(probs.shape[0]):
                start, end = offsets[b]
                sent_id = i + b

                out = {
                    "doc_id": doc_id,
                    "sent_id": int(sent_id),
                    "start": int(start),
                    "end": int(end),
                    "text": texts[b],
                }

                for sem_lab in SEMEVAL_14:
                    src_labels = semeval_map.get(sem_lab, [])
                    if not src_labels:
                        out[sem_lab] = 0.0
                        continue
                    src_ps = []
                    for sl in src_labels:
                        j = label_to_index.get(norm_label(sl))
                        if j is None:
                            continue
                        src_ps.append(float(probs[b, j]))
                    out[sem_lab] = union_prob(src_ps) if src_ps else 0.0

                prob_rows.append(out)

                if args.write_pred:
                    for sem_lab in SEMEVAL_14:
                        if float(out[sem_lab]) >= args.threshold:
                            pred_rows.append({
                                "doc_id": doc_id,
                                "label": sem_lab,
                                "start": int(start),
                                "end": int(end),
                            })

    pd.DataFrame(prob_rows).to_csv(args.out_sentence_probs, index=False, encoding="utf-8")
    print(f"[DONE] Wrote {len(prob_rows)} rows -> {args.out_sentence_probs}")

    if args.write_pred:
        pd.DataFrame(pred_rows).to_csv(args.out_pred, index=False, encoding="utf-8")
        print(f"[DONE] Wrote {len(pred_rows)} thresholded frags -> {args.out_pred}")


if __name__ == "__main__":
    main()
