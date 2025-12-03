"""
Extracts propaganda examples from SemEval 2020 PTC dataset
and aligns them with their labeled propaganda techniques.
"""

import os
import pandas as pd

# ==== CONFIGURATION ====
LABEL_FILE = "semeval_datasets/train-task2-TC.labels"      # Your .labels file
ARTICLES_DIR = "semeval_datasets/train-articles"           # Folder with article*.txt files
OUTPUT_FILE = "semeval_examples.csv"                       # Output file

# ==== LOAD LABEL FILE ====
labels = []
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            article_id, label, start, end = parts
            labels.append({
                "article_id": article_id.strip(),
                "label": label.strip(),
                "start": int(start),
                "end": int(end)
            })

print(f"[INFO] Loaded {len(labels)} labeled propaganda spans")

# ==== EXTRACT TEXT SPANS ====
examples = []
missing_files = 0

for item in labels:
    article_path = os.path.join(ARTICLES_DIR, f"article{item['article_id']}.txt")
    if not os.path.exists(article_path):
        missing_files += 1
        continue

    # Read the article as one continuous string
    with open(article_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract the propaganda span using offsets
    start, end = item["start"], item["end"]
    snippet = text[start:end].replace('\n', ' ').strip()

    examples.append({
        "article_id": item["article_id"],
        "label": item["label"],
        "text_snippet": snippet
    })

print(f"[INFO] Extracted {len(examples)} examples from {len(labels)} labels")
if missing_files:
    print(f"[WARN] {missing_files} article files missing")

# ==== SAVE OUTPUT ====
df = pd.DataFrame(examples)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"[DONE] Saved extracted propaganda examples to {OUTPUT_FILE}")
