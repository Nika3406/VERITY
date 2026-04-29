#!/usr/bin/env python3
"""
debertaL_v2.py

DeBERTa propaganda detection trainer — Professor Janghoon's pipeline.

Phases:
  finetune    — Full fine-tune on SemEval + Reddit (first run)
  rora        — Freeze backbone, inject RoRA adapters, fine-tune on SemEval
  al_retrain  — Add AL+LLM-labeled Reddit data, retrain with RoRA active
  calibrate   — Re-calibrate per-label thresholds only (no training)

RoRA (Rank-adaptive Reliability Optimization, 2025):
  Self-contained implementation — no peft package required.
  Key difference from standard LoRA: scaling = α/√r  (not α/r).
  Applied to the last 12 transformer layers of DeBERTa-v3-large.

Run from: BERTResearch/

  Step 1 — initial full fine-tune:
    python debertaL_v2.py --phase finetune

  Step 2 — freeze backbone + RoRA:
    python debertaL_v2.py --phase rora --epochs 4

  Step 3 — after LLM labeling, add AL data and retrain:
    python debertaL_v2.py --phase al_retrain \\
        --al_labeled_csv al_labeled_consensus.csv \\
        --output_dir deberta-propaganda-multilabel_rora

  Step 4 — recalibrate thresholds:
    python debertaL_v2.py --phase calibrate \\
        --output_dir deberta-propaganda-multilabel_rora
"""

import os
import glob
import json
import math
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    precision_recall_fscore_support,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from tqdm import tqdm

# ===================== CONFIGURATION =====================
SEMEVAL_FILE   = "semeval_processed.csv"
DATASET_DIR    = "datasets"
MODEL_NAME     = "microsoft/deberta-v3-large"
OUTPUT_DIR     = "deberta-propaganda-multilabel"
THRESHOLD_FILE = os.path.join(OUTPUT_DIR, "per_label_thresholds.json")
MAX_LENGTH     = 128
RANDOM_SEED    = 42

# RoRA settings — professor: rank 16-64, alpha per RoRA paper
LORA_RANK           = 32          # r
LORA_ALPHA          = 64          # alpha; RoRA scale = alpha/sqrt(r)
LORA_DROPOUT        = 0.1
# Target last 12 layers of DeBERTa-v3-large (24 total, 0-indexed)
LORA_TARGET_LAYERS  = list(range(12, 24))
# Linear module names inside each transformer layer to adapt
LORA_TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "dense"]

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


# ===================== RoRA — SELF-CONTAINED IMPLEMENTATION =====================

class RoRALinear(torch.nn.Module):
    """
    LoRA wrapper with RoRA scaling (alpha / sqrt(rank)).

    Standard LoRA: output += (B @ A @ x) * (alpha / rank)
    RoRA    2025 : output += (B @ A @ x) * (alpha / sqrt(rank))

    The original frozen weight is preserved; only A and B are trained.
    B is initialised to zero so the adapter starts at identity.
    """

    def __init__(self, linear: torch.nn.Linear, rank: int,
                 alpha: float, dropout: float = 0.1):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.rank         = rank
        self.scale        = alpha / math.sqrt(rank)   # RoRA key change

        # Keep the original frozen weight
        self.weight = linear.weight
        self.bias   = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Trainable low-rank matrices
        self.lora_A  = torch.nn.Linear(self.in_features,  rank, bias=False)
        self.lora_B  = torch.nn.Linear(rank, self.out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

        torch.nn.init.normal_(self.lora_A.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base    = torch.nn.functional.linear(x, self.weight, self.bias)
        adapter = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base + adapter


def apply_rora(model, target_layers, target_modules, rank, alpha, dropout):
    """
    1. Freeze all model parameters.
    2. Keep classifier + pooler head trainable.
    3. Inject RoRALinear adapters into target linear modules of target layers.
    Returns the number of adapters injected.
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze head
    for name, p in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            p.requires_grad = True

    # Find encoder layers
    try:
        encoder_layers = model.deberta.encoder.layer
    except AttributeError:
        print("[WARNING] Cannot locate model.deberta.encoder.layer — "
              "check model architecture. RoRA not applied.")
        return 0

    injected = 0
    for layer_idx in target_layers:
        if layer_idx >= len(encoder_layers):
            continue
        layer = encoder_layers[layer_idx]

        for mod_path, module in list(layer.named_modules()):
            if mod_path.split(".")[-1] not in target_modules:
                continue
            if not isinstance(module, torch.nn.Linear):
                continue

            parts  = mod_path.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)

            setattr(parent, parts[-1],
                    RoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
            injected += 1

    # Make adapter params trainable
    for m in model.modules():
        if isinstance(m, RoRALinear):
            m.lora_A.weight.requires_grad = True
            m.lora_B.weight.requires_grad = True

    return injected


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


# ===================== DATA LOADING =====================

def load_semeval(filepath: str):
    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"\n[ERROR] {filepath} not found.\n"
            "Run: python semeval_data_processor.py"
        )
    df = pd.read_csv(filepath)
    missing = [l for l in SEMEVAL_14_LABELS if l not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing label columns: {missing}")
    label_cols = [c for c in df.columns if c != "text"]
    print(f"[INFO] SemEval: {len(df):,} samples, {len(label_cols)} labels")
    return df, label_cols


def load_reddit_unlabeled(dataset_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(dataset_dir, "S-Dataset_r_*.csv"))
    if not csv_files:
        print(f"[WARNING] No Reddit CSVs in {dataset_dir}/")
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if "text" in df.columns:
                dfs.append(df[["text"]])
        except Exception as e:
            print(f"[WARNING] {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20].reset_index(drop=True)
    print(f"[INFO] Reddit: {len(df):,} from {len(csv_files)} subreddits")
    return df


def load_al_labeled(filepath: str, label_columns: list) -> pd.DataFrame:
    if not filepath or not Path(filepath).exists():
        if filepath:
            print(f"[WARNING] AL file not found: {filepath}")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    for col in [c for c in label_columns if c not in df.columns]:
        df[col] = 0.0
        print(f"[WARNING] AL missing column {col}, filled with 0")
    print(f"[INFO] AL labeled: {len(df):,} samples")
    return df[["text"] + label_columns]


# ===================== SEMANTIC LABELING =====================

def semantic_label_reddit(
    df_reddit: pd.DataFrame,
    df_semeval: pd.DataFrame,
    label_columns: list,
    sim_threshold: float = 0.60,
    borderline_upper: float = 0.65,
    min_examples: int = 5,
) -> pd.DataFrame:
    """
    Label Reddit texts via cosine similarity to SemEval technique anchors.
    Accept >= 0.60, reject borderline [0.60, 0.65) per professor's spec.
    """
    print(f"\n[INFO] Semantic labeling  threshold={sim_threshold}  "
          f"borderline reject <{borderline_upper}")
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("[ERROR] pip install sentence-transformers --break-system-packages")
        return pd.DataFrame()

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    tech_embs = {}
    for label in tqdm(label_columns, desc="Encoding anchors"):
        anchors = df_semeval[df_semeval[label] > 0.5]["text"].tolist()
        if len(anchors) < min_examples:
            print(f"[WARNING] {label}: only {len(anchors)} anchors — skipping")
            tech_embs[label] = None
        else:
            tech_embs[label] = emb_model.encode(
                anchors, convert_to_tensor=True, show_progress_bar=False
            )

    print(f"[INFO] Encoding {len(df_reddit):,} Reddit texts...")
    reddit_embs = emb_model.encode(
        df_reddit["text"].tolist(), batch_size=128,
        convert_to_tensor=True, show_progress_bar=True
    )

    matrix = np.zeros((len(df_reddit), len(label_columns)))
    for i, label in enumerate(label_columns):
        if tech_embs[label] is None:
            continue
        sims      = util.cos_sim(reddit_embs, tech_embs[label]).max(dim=1).values.cpu().numpy()
        activated = sims >= sim_threshold
        borderline = activated & (sims < borderline_upper)
        matrix[:, i] = activated.astype(float)
        matrix[borderline, i] = 0.0

    df_out = df_reddit[["text"]].copy()
    for i, label in enumerate(label_columns):
        df_out[label] = matrix[:, i].astype(int)

    mask   = df_out[label_columns].sum(axis=1) > 0
    df_out = df_out[mask].reset_index(drop=True)
    pct    = 100 * len(df_out) / max(len(df_reddit), 1)
    print(f"[INFO] Labeled: {len(df_out):,} ({pct:.1f}% of Reddit data)")
    return df_out


# ===================== DATASET PREPARATION =====================

def prepare_hf_dataset(df, label_columns, tokenizer):
    ds = Dataset.from_dict({
        "text":   df["text"].tolist(),
        "labels": df[label_columns].values.tolist(),
    })
    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True, padding=True,
                        max_length=MAX_LENGTH)
        enc["labels"] = [[float(x) for x in l] for l in batch["labels"]]
        return enc
    return ds.map(tok, batched=True, remove_columns=["text"])


def compute_class_weights(df, label_columns):
    N = len(df)
    weights = []
    for label in label_columns:
        n_pos = (df[label] > 0.5).sum()
        weights.append(min((N - n_pos) / n_pos if n_pos > 0 else 1.0, 20.0))
    w = torch.tensor(weights, dtype=torch.float)
    top5 = sorted(zip(label_columns, weights), key=lambda x: x[1], reverse=True)[:5]
    print("\n[INFO] Class weights (top 5):")
    for k, v in top5:
        print(f"  {k:<45} {v:.2f}x")
    return w


# ===================== TRAINER =====================

class WeightedMultilabelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None else None
        )
        loss = loss_fct(outputs.logits, labels.float())
        return (loss, outputs) if return_outputs else loss


class MultilabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["labels"].float()
        return batch


def make_compute_metrics(label_columns):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs  = torch.sigmoid(torch.tensor(logits)).numpy()
        preds  = (probs > 0.5).astype(int)
        labels = (labels > 0.5).astype(int)
        r = classification_report(labels, preds, target_names=label_columns,
                                   output_dict=True, zero_division=0)
        return {
            "hamming_loss": hamming_loss(labels, preds),
            "macro_f1":     r["macro avg"]["f1-score"],
            "micro_f1":     r["micro avg"]["f1-score"],
            "weighted_f1":  r["weighted avg"]["f1-score"],
        }
    return compute_metrics


def make_training_args(output_dir, num_epochs, lr=2e-5):
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        seed=RANDOM_SEED,
    )


# ===================== THRESHOLD CALIBRATION =====================

def calibrate_thresholds(model, tokenizer, val_df, label_columns,
                          device, target_precision=0.90):
    print(f"\n[INFO] Calibrating thresholds (target P ≥ {target_precision:.0%})...")
    model.eval()
    all_probs = []
    for i in tqdm(range(0, len(val_df), 32), desc="Inference"):
        inp = tokenizer(val_df["text"].iloc[i:i+32].tolist(),
                        truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            all_probs.append(torch.sigmoid(model(**inp).logits).cpu().numpy())
    all_probs   = np.vstack(all_probs)
    true_labels = (val_df[label_columns].values > 0.5).astype(int)
    thresholds  = {}
    for i, label in enumerate(label_columns):
        lp, lt = all_probs[:, i], true_labels[:, i]
        best_t, best_f1 = 0.5, -1.0
        if lt.sum() == 0:
            thresholds[label] = 0.5
            continue
        for t in np.arange(0.10, 0.95, 0.05):
            preds = (lp >= t).astype(int)
            if preds.sum() == 0:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                lt, preds, average="binary", zero_division=0)
            if p >= target_precision and f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[label] = best_t
    print("\n[INFO] Per-label thresholds:")
    for label, t in sorted(thresholds.items(), key=lambda x: x[1]):
        print(f"  {label:<45} {t:.2f}")
    return thresholds



# ===================== ADAPTER SAVE / LOAD =====================

def save_adapter_weights(model, output_dir: str):
    """Save only the RoRA adapter (lora_A, lora_B) weights separately."""
    adapter_state = {
        k: v.cpu() for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    path = os.path.join(output_dir, "rora_adapters.pt")
    torch.save(adapter_state, path)
    print(f"[INFO] Adapter weights saved: {len(adapter_state)} tensors → {path}")


def load_adapter_weights(model, output_dir: str):
    """Load RoRA adapter weights into an already-injected model."""
    path = os.path.join(output_dir, "rora_adapters.pt")
    if not Path(path).exists():
        print(f"[WARNING] No adapter file at {path} — adapters start fresh")
        return
    adapter_state = torch.load(path, map_location="cpu")
    missing, _ = model.load_state_dict(adapter_state, strict=False)
    loaded = len(adapter_state) - len(missing)
    print(f"[INFO] Adapter weights loaded: {loaded}/{len(adapter_state)} tensors")

# ===================== TRAINING PHASES =====================

def _save_model(trainer, tokenizer, label_columns, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({i: l for i, l in enumerate(label_columns)}, f, indent=2)
    print(f"\n[SUCCESS] Model saved to {output_dir}/")


def phase_finetune(semeval_file, dataset_dir, output_dir,
                   al_labeled_csv=None, num_epochs=4):
    """Full fine-tune — all parameters updated. Use for the first training run."""
    print("\n" + "="*60)
    print("PHASE: FULL FINE-TUNE (all parameters)")
    print("="*60)

    df_semeval, _ = load_semeval(semeval_file)
    label_columns = [l for l in SEMEVAL_14_LABELS if l in df_semeval.columns]

    df_reddit = load_reddit_unlabeled(dataset_dir)
    df_reddit_labeled = (
        semantic_label_reddit(df_reddit, df_semeval, label_columns)
        if not df_reddit.empty else pd.DataFrame()
    )
    df_al = load_al_labeled(al_labeled_csv, label_columns) if al_labeled_csv else pd.DataFrame()

    dfs = [df_semeval[["text"] + label_columns]]
    if not df_reddit_labeled.empty:
        dfs.append(df_reddit_labeled[["text"] + label_columns])
    if not df_al.empty:
        dfs.append(df_al[["text"] + label_columns])

    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"])
    df = df[df[label_columns].sum(axis=1) > 0].reset_index(drop=True)
    print(f"\n[INFO] Total: {len(df):,}  (SemEval={len(df_semeval):,}  "
          f"Reddit={len(df_reddit_labeled):,}  AL={len(df_al):,})")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=RANDOM_SEED)
    print(f"[INFO] Train: {len(train_df):,}  Val: {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_columns),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    class_weights = compute_class_weights(train_df, label_columns)
    trainer = WeightedMultilabelTrainer(
        model=model,
        args=make_training_args(output_dir, num_epochs),
        train_dataset=prepare_hf_dataset(train_df, label_columns, tokenizer),
        eval_dataset=prepare_hf_dataset(val_df, label_columns, tokenizer),
        compute_metrics=make_compute_metrics(label_columns),
        data_collator=MultilabelDataCollator(tokenizer),
        class_weights=class_weights,
    )
    print("\n[INFO] Training...")
    trainer.train()

    metrics = trainer.evaluate()
    print("\nFINAL METRICS:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    _save_model(trainer, tokenizer, label_columns, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    thresholds = calibrate_thresholds(model, tokenizer, val_df, label_columns, device)
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[SUCCESS] Thresholds → {THRESHOLD_FILE}")


def phase_rora(semeval_file, dataset_dir, output_dir,
               al_labeled_csv=None, num_epochs=4,
               lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA):
    """
    RoRA phase (professor step 2 / step 5):
      - Load existing trained model
      - Freeze backbone
      - Inject RoRA adapters (α/√r scaling) into last 12 layers
      - Train only adapters + classifier head
    """
    rora_scale = lora_alpha / math.sqrt(lora_rank)
    print("\n" + "="*60)
    print("PHASE: RoRA (freeze backbone + low-rank adapters)")
    print(f"  r={lora_rank}  α={lora_alpha}  scale α/√r={rora_scale:.3f}")
    print(f"  Target transformer layers: {LORA_TARGET_LAYERS[0]}–{LORA_TARGET_LAYERS[-1]}")
    print("="*60)

    base = output_dir if Path(output_dir).exists() else MODEL_NAME
    print(f"\n[INFO] Loading from: {base}")

    lm_path = os.path.join(base, "label_mapping.json")
    if Path(lm_path).exists():
        with open(lm_path) as f:
            lm = {int(k): v for k, v in json.load(f).items()}
        label_columns = [lm[i] for i in range(len(lm))]
    else:
        label_columns = SEMEVAL_14_LABELS

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(label_columns),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    n = apply_rora(model, LORA_TARGET_LAYERS, LORA_TARGET_MODULES,
                   lora_rank, lora_alpha, LORA_DROPOUT)
    trainable, total = count_trainable(model)
    print(f"\n[INFO] Adapters injected: {n}")
    print(f"[INFO] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    if n == 0:
        print("[WARNING] No adapters injected — training classifier head only.")

    # Load previously saved adapter weights if this is a continuation run
    # (e.g. al_retrain loading from a prior rora output directory)
    load_adapter_weights(model, base)

    # Load data: SemEval always, plus AL if provided
    df_semeval, _ = load_semeval(semeval_file)
    df_al = load_al_labeled(al_labeled_csv, label_columns) if al_labeled_csv else pd.DataFrame()

    dfs = [df_semeval[["text"] + label_columns]]
    if not df_al.empty:
        dfs.append(df_al[["text"] + label_columns])
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"])
    df = df[df[label_columns].sum(axis=1) > 0].reset_index(drop=True)
    print(f"\n[INFO] RoRA training set: {len(df):,} samples")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=RANDOM_SEED)

    class_weights = compute_class_weights(train_df, label_columns)
    rora_out = output_dir + "_rora"
    trainer = WeightedMultilabelTrainer(
        model=model,
        args=make_training_args(rora_out, num_epochs, lr=1e-4),
        train_dataset=prepare_hf_dataset(train_df, label_columns, tokenizer),
        eval_dataset=prepare_hf_dataset(val_df, label_columns, tokenizer),
        compute_metrics=make_compute_metrics(label_columns),
        data_collator=MultilabelDataCollator(tokenizer),
        class_weights=class_weights,
    )
    print("\n[INFO] RoRA training...")
    trainer.train()

    metrics = trainer.evaluate()
    print("\nFINAL RoRA METRICS:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    _save_model(trainer, tokenizer, label_columns, rora_out)
    save_adapter_weights(model, rora_out)
    print(f"[INFO] For next steps use --output_dir {rora_out}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    thresholds = calibrate_thresholds(model, tokenizer, val_df, label_columns, device)
    thr_file = os.path.join(rora_out, "per_label_thresholds.json")
    with open(thr_file, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[SUCCESS] Thresholds → {thr_file}")


# ===================== ENTRY POINT =====================

def main():
    parser = argparse.ArgumentParser(
        description="DeBERTa propaganda detector — Professor Janghoon's pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run from BERTResearch/:

  python debertaL_v2.py --phase finetune
  python debertaL_v2.py --phase rora --epochs 4
  python debertaL_v2.py --phase al_retrain --al_labeled_csv al_labeled_consensus.csv \\
      --output_dir deberta-propaganda-multilabel_rora
  python debertaL_v2.py --phase calibrate \\
      --output_dir deberta-propaganda-multilabel_rora
        """
    )
    parser.add_argument("--phase",
        choices=["finetune", "rora", "al_retrain", "calibrate"],
        default="finetune")
    parser.add_argument("--semeval_csv",    default=SEMEVAL_FILE)
    parser.add_argument("--dataset_dir",    default=DATASET_DIR)
    parser.add_argument("--output_dir",     default=OUTPUT_DIR)
    parser.add_argument("--al_labeled_csv", default=None,
        help="CSV of AL-selected + LLM-consensus-labeled Reddit samples")
    parser.add_argument("--epochs",         type=int,   default=4)
    parser.add_argument("--lora_rank",      type=int,   default=LORA_RANK,
        help=f"RoRA rank r (default: {LORA_RANK})")
    parser.add_argument("--lora_alpha",     type=float, default=LORA_ALPHA,
        help=f"RoRA alpha (default: {LORA_ALPHA}; scale = alpha/sqrt(rank))")
    args = parser.parse_args()

    print("=" * 60)
    print(f"DeBERTa Propaganda Detector  —  Phase: {args.phase.upper()}")
    print("=" * 60)

    if args.phase == "finetune":
        phase_finetune(args.semeval_csv, args.dataset_dir, args.output_dir,
                       args.al_labeled_csv, args.epochs)

    elif args.phase == "rora":
        phase_rora(args.semeval_csv, args.dataset_dir, args.output_dir,
                   args.al_labeled_csv, args.epochs, args.lora_rank, args.lora_alpha)

    elif args.phase == "al_retrain":
        if not args.al_labeled_csv:
            parser.error("--al_labeled_csv required for al_retrain")
        phase_rora(args.semeval_csv, args.dataset_dir, args.output_dir,
                   args.al_labeled_csv, min(args.epochs, 2),
                   args.lora_rank, args.lora_alpha)

    elif args.phase == "calibrate":
        if not Path(args.output_dir).exists():
            parser.error(f"Model not found: {args.output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with open(os.path.join(args.output_dir, "label_mapping.json")) as f:
            lm = {int(k): v for k, v in json.load(f).items()}
        label_columns = [lm[i] for i in range(len(lm))]
        df_semeval, _ = load_semeval(args.semeval_csv)
        _, val_df = train_test_split(df_semeval, test_size=0.15, random_state=RANDOM_SEED)
        thresholds = calibrate_thresholds(model, tokenizer, val_df, label_columns, device)
        thr_file = os.path.join(args.output_dir, "per_label_thresholds.json")
        with open(thr_file, "w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"[SUCCESS] Thresholds → {thr_file}")


if __name__ == "__main__":
    main()