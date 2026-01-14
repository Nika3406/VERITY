import glob
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ===================== CONFIGURATION =====================
DATASET_DIR = "./datasets"
SEMEVAL_FILE = "semeval_processed.csv"
MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = "./deberta-propaganda-multilabel"

MAX_LENGTH = 128  # CRITICAL: Reduced to 128 to fit in memory

print(f"[INFO] Training multi-label propaganda detection model")

# ===================== LOAD PROCESSED SEMEVAL DATA =====================
print(f"\n[INFO] Loading processed SemEval data from {SEMEVAL_FILE}")

if not os.path.exists(SEMEVAL_FILE):
    raise FileNotFoundError(
        f"{SEMEVAL_FILE} not found. Run semeval_data_processor.py first."
    )

df_semeval = pd.read_csv(SEMEVAL_FILE)

label_columns = [col for col in df_semeval.columns if col != "text"]
NUM_LABELS = len(label_columns)

print(f"[INFO] Loaded {len(df_semeval)} labeled examples")
print(f"[INFO] Detecting {NUM_LABELS} propaganda techniques:")
for label in label_columns:
    print(f"  - {label}: {int(df_semeval[label].sum())} examples")

# ===================== LOAD SCRAPED REDDIT DATA =====================
print(f"\n[INFO] Loading scraped Reddit data from {DATASET_DIR}")

csv_files = glob.glob(os.path.join(DATASET_DIR, "S-Dataset_r_*.csv"))

if csv_files:
    df_list = []
    for file in csv_files:
        temp = pd.read_csv(file)
        if "text" in temp.columns:
            df_list.append(temp[["text"]])
    df_reddit = pd.concat(df_list, ignore_index=True).dropna()
    df_reddit["text"] = df_reddit["text"].astype(str)
    print(f"[INFO] Loaded {len(df_reddit)} Reddit samples")
else:
    print("[WARNING] No Reddit data found. Using SemEval only.")
    df_reddit = pd.DataFrame()

# ===================== SEMANTIC LABELING =====================
if not df_reddit.empty:
    print("\n[INFO] Performing semantic labeling of Reddit data...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    technique_embeddings = {}
    for label in tqdm(label_columns, desc="Encoding SemEval by technique"):
        texts = df_semeval[df_semeval[label] == 1]["text"].tolist()
        technique_embeddings[label] = (
            embedding_model.encode(texts, convert_to_tensor=True)
            if texts else None
        )

    reddit_embeddings = embedding_model.encode(
        df_reddit["text"].tolist(),
        batch_size=128,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    reddit_labels = np.zeros((len(df_reddit), NUM_LABELS), dtype=int)
    SIM_THRESHOLD = 0.65

    for i, label in enumerate(label_columns):
        emb = technique_embeddings[label]
        if emb is None:
            continue
        sims = util.cos_sim(reddit_embeddings, emb).max(dim=1).values
        reddit_labels[:, i] = (sims > SIM_THRESHOLD).cpu().numpy().astype(int)

    for i, label in enumerate(label_columns):
        df_reddit[label] = reddit_labels[:, i]

    df_combined = pd.concat([df_semeval, df_reddit], ignore_index=True)
else:
    df_combined = df_semeval.copy()

# Remove samples with no labels
df_combined = df_combined[df_combined[label_columns].sum(axis=1) > 0]

print(f"\n[INFO] Total training samples: {len(df_combined)}")

# ===================== TRAIN / VAL SPLIT =====================
train_df, val_df = train_test_split(df_combined, test_size=0.15, random_state=42)

# ===================== LOAD MODEL =====================
print(f"\n[INFO] Loading {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

# ===================== DATASETS =====================
def prepare_dataset(df):
    return Dataset.from_dict({
        "text": df["text"].tolist(),
        "labels": df[label_columns].values.tolist(),
    })

train_ds = prepare_dataset(train_df)
val_ds = prepare_dataset(val_df)

def tokenize_fn(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = [[float(x) for x in lbl] for lbl in batch["labels"]]
    return enc

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# ===================== DATA COLLATOR =====================
class MultilabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["labels"].float()
        return batch

data_collator = MultilabelDataCollator(tokenizer)

# ===================== METRICS =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    labels = labels.astype(int)

    report = classification_report(
        labels,
        preds,
        target_names=label_columns,
        output_dict=True,
        zero_division=0,
    )

    return {
        "hamming_loss": hamming_loss(labels, preds),
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["micro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

# ===================== TRAINING =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=1,     # Back to 1 due to OOM
    per_device_eval_batch_size=2,      # Reduced to 2
    gradient_accumulation_steps=8,     # Keep effective batch at 8
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,       # Reduce memory pressure
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("\n[INFO] Starting training...")
trainer.train()

# ===================== FINAL EVAL =====================
print("\n[INFO] Final evaluation...")
metrics = trainer.evaluate()

print("\n" + "=" * 60)
print("FINAL METRICS")
print("=" * 60)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ===================== SAVE MODEL =====================
print(f"\n[INFO] Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

import json
with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
    json.dump({i: l for i, l in enumerate(label_columns)}, f, indent=2)

print("[DONE] Model saved successfully")