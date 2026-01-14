import pandas as pd
import numpy as np
from collections import defaultdict

"""
Process the SemEval propaganda dataset for multi-label classification training.
"""

# ===================== LOAD SEMEVAL DATA =====================
print("[INFO] Loading SemEval data...")
df = pd.read_csv("semeval_examples.csv")

print(f"[INFO] Loaded {len(df)} labeled examples")
print(f"[INFO] Columns: {df.columns.tolist()}")

# ===================== NORMALIZE LABELS =====================
print("\n[INFO] Normalizing propaganda technique labels...")

# Get all unique labels from the dataset
all_labels = set()
for label in df['label'].dropna():
    # Split comma-separated labels
    techniques = [t.strip() for t in str(label).split(',')]
    all_labels.update(techniques)

# Create normalized label mapping (remove spaces, lowercase, handle variations)
label_mapping = {}
STANDARD_LABELS = [
    "loaded_language",
    "name_calling",
    "repetition",
    "exaggeration",
    "appeal_to_fear",
    "causal_oversimplification",
    "doubt",
    "appeal_to_authority",
    "flag_waving",
    "black_white_fallacy",
    "thought_terminating_cliche",
    "whataboutism",
    "reductio_ad_hitlerum",
    "red_herring",
    "straw_man",
    "bandwagon",
    "obfuscation",
    "slogans"
]

# Map SemEval labels to our standard labels
for label in all_labels:
    normalized = label.lower().replace('-', '_').replace(' ', '_')
    
    # Handle specific mappings
    if 'name_calling' in normalized or 'labeling' in normalized:
        label_mapping[label] = 'name_calling'
    elif 'exaggeration' in normalized or 'minimisation' in normalized:
        label_mapping[label] = 'exaggeration'
    elif 'appeal_to_fear' in normalized or 'prejudice' in normalized:
        label_mapping[label] = 'appeal_to_fear'
    elif 'black' in normalized and 'white' in normalized and 'fallacy' in normalized:
        label_mapping[label] = 'black_white_fallacy'
    elif 'thought' in normalized and 'terminating' in normalized:
        label_mapping[label] = 'thought_terminating_cliche'
    elif 'whataboutism' in normalized or 'straw' in normalized or 'red_herring' in normalized:
        label_mapping[label] = 'whataboutism'
    elif 'bandwagon' in normalized or 'reductio' in normalized or 'hitlerum' in normalized:
        label_mapping[label] = 'bandwagon'
    elif 'slogans' in normalized:
        label_mapping[label] = 'slogans'
    elif 'obfuscation' in normalized:
        label_mapping[label] = 'obfuscation'
    else:
        # Try to match to standard labels
        for std_label in STANDARD_LABELS:
            if std_label.replace('_', '') in normalized.replace('_', ''):
                label_mapping[label] = std_label
                break
        else:
            # Keep original normalized form if no match found
            label_mapping[label] = normalized

print(f"[INFO] Found {len(all_labels)} unique label types")
print(f"[INFO] Mapped to {len(set(label_mapping.values()))} standard labels")

# ===================== CREATE MULTI-HOT ENCODED DATASET =====================
print("\n[INFO] Creating multi-hot encoded dataset...")

# Get final set of labels
final_labels = sorted(set(label_mapping.values()))
print(f"[INFO] Final label set ({len(final_labels)} labels):")
for label in final_labels:
    print(f"  - {label}")

# Group by text_snippet and aggregate labels
text_to_labels = defaultdict(set)
for _, row in df.iterrows():
    text = str(row['text_snippet']).strip()
    if not text or len(text) < 10:
        continue
    
    label_str = str(row['label'])
    if pd.isna(label_str) or label_str == 'nan':
        continue
    
    # Split comma-separated labels
    techniques = [t.strip() for t in label_str.split(',')]
    
    # Map to standard labels
    for technique in techniques:
        if technique in label_mapping:
            standard_label = label_mapping[technique]
            text_to_labels[text].add(standard_label)

# Create final dataframe
data_rows = []
for text, labels in text_to_labels.items():
    row = {'text': text}
    # Create multi-hot encoding
    for label in final_labels:
        row[label] = 1 if label in labels else 0
    data_rows.append(row)

df_processed = pd.DataFrame(data_rows)

print(f"\n[INFO] Processed {len(df_processed)} unique text samples")

# ===================== STATISTICS =====================
print("\n[INFO] Label distribution:")
for label in final_labels:
    count = df_processed[label].sum()
    pct = 100 * count / len(df_processed)
    print(f"  {label:.<40} {int(count):>6} ({pct:>5.1f}%)")

avg_labels = df_processed[final_labels].sum(axis=1).mean()
print(f"\n[INFO] Average labels per sample: {avg_labels:.2f}")

# ===================== SAVE PROCESSED DATA =====================
output_file = "semeval_processed.csv"
df_processed.to_csv(output_file, index=False)
print(f"\n[SUCCESS] Saved processed data to {output_file}")

print("\n[NEXT STEPS]")
print("1. This processed file is ready for training")
print("2. Run the updated training script with this data")
print("3. The Reddit data will be semantically labeled based on these examples")
