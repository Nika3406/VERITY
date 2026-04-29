#!/usr/bin/env python3
"""
llm_ensemble_labeler.py

Professor Janghoon's step 3: Send active-learning candidates to 3 local LLMs
via Ollama, keep only samples where at least 2/3 agree on every subcategory.

Requires Ollama running locally:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1:8b
    ollama pull mistral:7b
    ollama pull gemma2:9b
    ollama serve          # start server (runs on localhost:11434)

Usage (from BERTResearch/):

    # Check Ollama is running and models are available:
    python llm_ensemble_labeler.py --check

    # Dry run — print prompt for first 3 samples, no model calls:
    python llm_ensemble_labeler.py --input active_learning_candidates.csv --dry_run

    # Full labeling run:
    python llm_ensemble_labeler.py \\
        --input  active_learning_candidates.csv \\
        --output al_labeled_consensus.csv

    # Resume a crashed run:
    python llm_ensemble_labeler.py \\
        --input  active_learning_candidates.csv \\
        --output al_labeled_consensus.csv \\
        --resume

    # Use different models or Ollama host:
    python llm_ensemble_labeler.py \\
        --input active_learning_candidates.csv \\
        --models llama3.1:8b mistral:7b phi3:medium \\
        --ollama_host http://localhost:11434

Output: al_labeled_consensus.csv
    text, <14 label columns>, agreement_rate, n_llms_agreed, consensus_score
    Only rows where n_llms_agreed >= 2 are written.
"""

import os
import re
import json
import time
import argparse
import textwrap
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ===================== LABEL DEFINITIONS =====================

SEMEVAL_14_LABELS = [
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

LABEL_DESCRIPTIONS = {
    "appeal_to_authority":
        "Citing an authority figure to support a claim, often without relevant evidence.",
    "appeal_to_fear_prejudice":
        "Using fear, threats, or prejudice to influence the audience emotionally.",
    "bandwagon_reductio_ad_hitlerum":
        "Appealing to popularity ('everyone does it') or comparing opponents to Hitler/Nazis.",
    "black_and_white_fallacy":
        "Presenting only two options when more exist; false dichotomy.",
    "causal_oversimplification":
        "Attributing a complex event to a single cause, ignoring other factors.",
    "doubt":
        "Questioning the credibility of someone or something without providing evidence.",
    "exaggeration_minimisation":
        "Exaggerating or downplaying the significance of something.",
    "flag_waving":
        "Appealing to national or group pride/identity to justify an action.",
    "loaded_language":
        "Using words with strong emotional connotations to influence the audience.",
    "name_calling_labeling":
        "Attaching a negative label to a person or group to discredit them.",
    "repetition":
        "Repeating the same message or slogan multiple times to reinforce it.",
    "slogans":
        "A brief, striking phrase used to encapsulate a political position.",
    "thought_terminating_cliches":
        "Using a cliché to shut down critical thinking rather than engage with the argument.",
    "whataboutism_straw_men_red_herring":
        "Deflecting criticism by pointing to others' faults, misrepresenting arguments, "
        "or introducing irrelevant topics.",
}

DEFAULT_MODELS = ["llama3.1:8b", "mistral:7b", "gemma2:9b"]
DEFAULT_HOST   = "http://localhost:11434"


# ===================== PROMPT =====================

def build_prompt(text: str) -> str:
    label_block = "\n".join(
        f"  {i+1:>2}. {label}\n"
        f"       {LABEL_DESCRIPTIONS[label]}"
        for i, label in enumerate(SEMEVAL_14_LABELS)
    )

    return textwrap.dedent(f"""
You are an expert annotator for propaganda detection research.

TASK: Classify the following text by identifying which propaganda techniques it uses.

PROPAGANDA TECHNIQUES (use these exact names):
{label_block}

TEXT:
\"\"\"{text}\"\"\"

INSTRUCTIONS:
- Identify ALL techniques present. There may be zero or many.
- Reply with ONLY valid JSON. No explanation, no preamble, no markdown.
- Format exactly: {{"labels": ["technique_name", ...]}}
- Use the exact technique names listed above.
- If no techniques are present: {{"labels": []}}

JSON:
""").strip()


# ===================== OLLAMA CLIENT =====================

def check_ollama(host: str, models: list[str]) -> dict:
    """
    Check if Ollama is running and which requested models are available.
    Returns dict: {model_name: True/False}
    """
    try:
        import urllib.request
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        available = {m["name"].split(":")[0] for m in data.get("models", [])}
        # Also include full name tags
        available_full = {m["name"] for m in data.get("models", [])}
    except Exception as e:
        print(f"[ERROR] Cannot connect to Ollama at {host}: {e}")
        print("  Start Ollama with: ollama serve")
        return {m: False for m in models}

    result = {}
    for model in models:
        base = model.split(":")[0]
        result[model] = (model in available_full or base in available)
    return result


def call_ollama(text: str, model: str, host: str = DEFAULT_HOST,
                timeout: int = 120, retries: int = 2) -> list[str] | None:
    """
    Call Ollama generate API with the classification prompt.
    Returns list of valid label names, or None on failure.
    """
    import urllib.request
    import urllib.error

    prompt  = build_prompt(text)
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 300,
            "stop": ["\n\n", "TEXT:", "INSTRUCTIONS:"],
        },
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            raw = result.get("response", "").strip()
            return _parse_response(raw)

        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"[ERROR] {model} failed: {e}")
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"[ERROR] {model} unexpected error: {e}")
                return None


def _parse_response(raw: str) -> list[str]:
    """Parse LLM output into a validated list of label names."""
    # Strip markdown fences
    raw = raw.strip()
    if "```" in raw:
        raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()

    # Extract first JSON object
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if not m:
        # Try the whole string as JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return []

    if not isinstance(data, dict) or "labels" not in data:
        return []

    raw_labels = data["labels"]
    if not isinstance(raw_labels, list):
        return []

    valid = []
    for l in raw_labels:
        if not isinstance(l, str):
            continue
        l_norm = l.strip().lower().replace(" ", "_").replace("-", "_")
        if l_norm in SEMEVAL_14_LABELS:
            valid.append(l_norm)
        else:
            # Fuzzy match — accept if the canonical name contains the response
            for canon in SEMEVAL_14_LABELS:
                if l_norm in canon or canon.startswith(l_norm[:8]):
                    valid.append(canon)
                    break

    return list(set(valid))


# ===================== CONSENSUS =====================

def compute_consensus(responses: list, min_agreement: int = 2) -> tuple:
    """
    Majority vote across LLM responses.

    Returns:
        label_dict   {label: 0/1}
        agreement_rate  fraction of LLMs that matched the consensus set
        n_agreed     number of LLMs that agreed with the consensus
    """
    valid = [r for r in responses if r is not None]
    if not valid:
        return {l: 0 for l in SEMEVAL_14_LABELS}, 0.0, 0

    n_valid = len(valid)
    votes   = {l: 0 for l in SEMEVAL_14_LABELS}
    for response in valid:
        for label in response:
            if label in votes:
                votes[label] += 1

    label_dict    = {l: 1 if votes[l] >= min_agreement else 0
                     for l in SEMEVAL_14_LABELS}
    consensus_set = set(l for l, v in label_dict.items() if v == 1)

    agreed = sum(1 for r in valid if set(r) == consensus_set)
    return label_dict, agreed / n_valid, agreed


# ===================== MAIN PIPELINE =====================

def run_labeling(
    input_csv: str,
    output_csv: str,
    models: list[str] = None,
    host: str = DEFAULT_HOST,
    min_agreement: int = 2,
    max_samples: int = None,
    dry_run: bool = False,
    resume: bool = True,
    save_every: int = 10,
):
    if models is None:
        models = DEFAULT_MODELS

    print("=" * 70)
    print("LLM ENSEMBLE LABELER  (Ollama / local models)")
    print("=" * 70)
    print(f"Models        : {models}")
    print(f"Min agreement : {min_agreement}/{len(models)}")
    print(f"Ollama host   : {host}")
    print()

    # --- Verify Ollama + models ---
    if not dry_run:
        status = check_ollama(host, models)
        for model, ok in status.items():
            icon = "OK" if ok else "MISSING"
            print(f"  [{icon:^7}] {model}")
        missing = [m for m, ok in status.items() if not ok]
        if missing:
            print(f"\n[WARNING] Missing models. Pull them with:")
            for m in missing:
                print(f"  ollama pull {m}")
            print()
            if len(missing) >= len(models):
                print("[ERROR] No models available. Exiting.")
                return

    # --- Load input ---
    if not Path(input_csv).exists():
        raise FileNotFoundError(
            f"Input not found: {input_csv}\n"
            "Run active_learning_selector.py first."
        )
    df_in = pd.read_csv(input_csv)
    if "text" not in df_in.columns:
        raise ValueError("Input CSV must have a 'text' column.")
    if max_samples:
        df_in = df_in.head(max_samples)
    texts = df_in["text"].tolist()
    print(f"Input samples: {len(texts):,}")

    if dry_run:
        print("\n[DRY RUN] — showing prompts for first 3 samples\n")
        for i, text in enumerate(texts[:3]):
            print(f"{'='*60}\nSample {i+1}:\n{build_prompt(text[:400])}\n")
        return

    # --- Resume ---
    results     = []
    already_done = set()
    if resume and Path(output_csv).exists():
        df_ex = pd.read_csv(output_csv)
        already_done = set(df_ex["text"].tolist())
        results = df_ex.to_dict("records")
        print(f"[RESUME] {len(already_done):,} already labeled, continuing...")

    # --- Label ---
    accepted = rejected = 0
    pbar = tqdm(texts, desc="Labeling", unit="sample")

    for i, text in enumerate(pbar):
        if text in already_done:
            continue

        # Call all 3 models in parallel — avoids sequential model swapping
        from concurrent.futures import ThreadPoolExecutor, as_completed
        responses = [None] * len(models)
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(call_ollama, text, model, host): idx
                for idx, model in enumerate(models)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    responses[idx] = None

        label_dict, agreement_rate, n_agreed = compute_consensus(
            responses, min_agreement=min_agreement
        )

        if n_agreed >= min_agreement:
            results.append({
                "text":             text,
                **label_dict,
                "agreement_rate":   round(agreement_rate, 3),
                "n_llms_agreed":    n_agreed,
                "consensus_score":  round(sum(label_dict.values()) / 14, 3),
            })
            accepted += 1
        else:
            rejected += 1

        pbar.set_postfix(accepted=accepted, rejected=rejected)

        if (i + 1) % save_every == 0:
            _save(results, output_csv)

    _save(results, output_csv)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Accepted : {accepted:,}  ({100*accepted/max(len(texts),1):.1f}%)")
    print(f"  Rejected : {rejected:,}  (LLMs disagreed)")
    print(f"  Saved to : {output_csv}")

    if results:
        df_out = pd.DataFrame(results)
        print(f"\nLabel distribution in accepted samples ({len(df_out):,} total):")
        for label in SEMEVAL_14_LABELS:
            if label in df_out.columns:
                n   = int(df_out[label].sum())
                pct = 100 * n / len(df_out)
                bar = "█" * int(pct / 5)
                print(f"  {label:<45} {n:>5}  {pct:5.1f}%  {bar}")

    print(f"\nNext step:")
    print(f"  python debertaL_v2.py --phase al_retrain \\")
    print(f"      --al_labeled_csv {output_csv} \\")
    print(f"      --output_dir deberta-propaganda-multilabel_rora")


def _save(results: list, path: str):
    if not results:
        return
    df = pd.DataFrame(results)
    for col in SEMEVAL_14_LABELS:
        if col not in df.columns:
            df[col] = 0
    cols = (["text"] + SEMEVAL_14_LABELS +
            ["agreement_rate", "n_llms_agreed", "consensus_score"])
    df[[c for c in cols if c in df.columns]].to_csv(path, index=False)


# ===================== ENTRY POINT =====================

def main():
    parser = argparse.ArgumentParser(
        description="Local LLM ensemble labeler using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup (one time):
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3.1:8b
  ollama pull mistral:7b
  ollama pull gemma2:9b
  ollama serve

Run from BERTResearch/:
  python llm_ensemble_labeler.py --check
  python llm_ensemble_labeler.py --dry_run --input active_learning_candidates.csv
  python llm_ensemble_labeler.py --input active_learning_candidates.csv
        """
    )
    parser.add_argument("--input",   default="active_learning_candidates.csv")
    parser.add_argument("--output",  default="al_labeled_consensus.csv")
    parser.add_argument("--models",  nargs="+", default=DEFAULT_MODELS,
        help=f"Ollama models to use (default: {DEFAULT_MODELS})")
    parser.add_argument("--ollama_host", default=DEFAULT_HOST,
        help=f"Ollama API host (default: {DEFAULT_HOST})")
    parser.add_argument("--min_agreement", type=int, default=2,
        help="Min models that must agree (default: 2)")
    parser.add_argument("--max_samples",   type=int, default=None,
        help="Process only first N samples")
    parser.add_argument("--dry_run",  action="store_true",
        help="Print prompts only — no model calls")
    parser.add_argument("--no_resume", action="store_true",
        help="Start fresh, ignore existing output file")
    parser.add_argument("--save_every", type=int, default=10,
        help="Save progress every N samples (default: 10)")
    parser.add_argument("--check",  action="store_true",
        help="Check Ollama connection and model availability, then exit")

    args = parser.parse_args()

    if args.check:
        print(f"Checking Ollama at {args.ollama_host}...")
        status = check_ollama(args.ollama_host, args.models)
        all_ok = True
        for model, ok in status.items():
            icon = "✓" if ok else "✗"
            print(f"  {icon} {model}")
            if not ok:
                all_ok = False
                print(f"    → ollama pull {model}")
        if all_ok:
            print("\nAll models available. Ready to label.")
        return

    run_labeling(
        input_csv    = args.input,
        output_csv   = args.output,
        models       = args.models,
        host         = args.ollama_host,
        min_agreement= args.min_agreement,
        max_samples  = args.max_samples,
        dry_run      = args.dry_run,
        resume       = not args.no_resume,
        save_every   = args.save_every,
    )


if __name__ == "__main__":
    main()