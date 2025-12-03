import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import glob
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==== CONFIG ====
SEMEVAL_FILE = "semeval_examples.csv"
SPEECH_FOLDER = "speeches"
THRESHOLDS = np.arange(0.55, 0.60, 0.01)  # 0.30 to 0.90 with step 0.05
MAX_SAMPLES_PER_DATASET = 1000
PROPAGANDA_TEST_SIZE = 1000  # Equal number for balanced evaluation
NONPROPAGANDA_TEST_SIZE = 1000

# Output files
OUTPUT_THRESHOLD_RESULTS = "ThresholdData/threshold_evaluation_results.csv"
OUTPUT_POLITICAL_ANALYSIS = "ThresholdData/political_speeches_analysis.csv"
OUTPUT_POLITICAL_FLAGGED = "ThresholdData/political_speeches_flagged_sentences.csv"
OUTPUT_NONPOLITICAL_ANALYSIS = "ThresholdData/nonpolitical_text_analysis.csv"
OUTPUT_NONPOLITICAL_FLAGGED = "ThresholdData/nonpolitical_text_flagged_sentences.csv"

# ==== TEXT CLEANING ====
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    return text

# ==== SENTENCE SPLITTING ====
def split_into_sentences(text):
    """Split text into sentences using basic punctuation."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

# ==== LOAD NON-POLITICAL SENTENCES ====
def load_nonpolitical_sentences(max_per_dataset=MAX_SAMPLES_PER_DATASET):
    """Load non-political sentences from various HuggingFace datasets."""
    print("\n[INFO] Loading non-political sentences from HuggingFace datasets...")
    all_sentences = []
    
    dataset_configs = [
        ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text", None),
        ("tatsu-lab/alpaca", None, "train", "output", None),
        ("derek-thomas/ScienceQA", None, "train", "answer", None),
        ("nvidia/OpenMathInstruct-1", None, "train", "generated_solution", "omit_think"),
        ("dair-ai/emotion", None, "train", "text", None),
        ("knkarthick/samsum", None, "train", "summary", None),
        ("roneneldan/TinyStories", None, "train", "text", "second_sentence"),
        ("HiTZ/multilingual_medical_corpus", "en", "train", "text", None),
        ("Abirate/english_quotes", None, "train", "quote", None),
        ("SocialGrep/one-million-reddit-confessions", None, "train", "confession", None),
        ("AGBonnet/augmented-clinical-notes", None, "train", "full_note", None),
    ]
    
    for config in dataset_configs:
        if len(config) == 5:
            dataset_name, subset, split, field, special = config
        else:
            dataset_name, split, field, special = config
            subset = None
        
        try:
            print(f"[INFO] Loading {dataset_name}...")
            
            if subset:
                ds = load_dataset(dataset_name, subset, split=split)
            else:
                ds = load_dataset(dataset_name, split=split)
            
            if len(ds) > max_per_dataset:
                ds = ds.shuffle(seed=42).select(range(max_per_dataset))
            
            texts = []
            for item in tqdm(ds, desc=f"Processing {dataset_name}", leave=False):
                try:
                    text = item.get(field, "")
                    if not text or not isinstance(text, str):
                        continue
                    
                    if special == "omit_think":
                        text = re.sub(r'<THINK>.*?</THINK>', '', text, flags=re.DOTALL)
                    elif special == "second_sentence":
                        sents = split_into_sentences(text)
                        text = sents[1] if len(sents) > 1 else ""
                    
                    cleaned = clean_text(text)
                    if 20 <= len(cleaned) <= 500:
                        texts.append(cleaned)
                
                except Exception:
                    continue
            
            all_sentences.extend(texts)
            print(f"[INFO] Extracted {len(texts)} sentences from {dataset_name}")
        
        except Exception as e:
            print(f"[WARNING] Failed to load {dataset_name}: {e}")
            continue
    
    all_sentences = list(set(all_sentences))
    print(f"\n[INFO] Total non-political sentences collected: {len(all_sentences)}")
    return all_sentences

# ==== GPU SETUP ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# ==== LOAD MODEL ====
print("[INFO] Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
if device == 'cuda':
    model.half()
print("[INFO] Model loaded successfully")

# ==== LOAD PROPAGANDA EXAMPLES ====
print("\n" + "="*80)
print("STEP 1: LOADING AND PARTITIONING PROPAGANDA DATA")
print("="*80)

if not os.path.exists(SEMEVAL_FILE):
    raise FileNotFoundError(f"{SEMEVAL_FILE} not found!")

df_propaganda = pd.read_csv(SEMEVAL_FILE)
possible_text_columns = ["text_snippet", "text", "snippet", "content", "example"]
text_column = next((col for col in possible_text_columns if col in df_propaganda.columns), None)

if not text_column:
    raise ValueError(f"[ERROR] No suitable text column found. Available: {list(df_propaganda.columns)}")

# Get article IDs if available
id_columns = ["article_id", "articleID", "id", "article", "doc_id"]
id_column = next((col for col in id_columns if col in df_propaganda.columns), None)

print(f"[INFO] Using text column: '{text_column}'")
print(f"[INFO] Using ID column: '{id_column if id_column else 'None (will generate)'}'")

# Clean and prepare propaganda texts
df_propaganda['cleaned_text'] = df_propaganda[text_column].apply(lambda x: clean_text(str(x)) if pd.notna(x) else "")
df_propaganda = df_propaganda[df_propaganda['cleaned_text'].str.len() >= 20].copy()
df_propaganda = df_propaganda[df_propaganda['cleaned_text'].str.len() <= 500].copy()

# Add article IDs if missing
if id_column:
    df_propaganda['article_id'] = df_propaganda[id_column]
else:
    df_propaganda['article_id'] = [f"article_{i}" for i in range(len(df_propaganda))]

# Remove duplicates
df_propaganda = df_propaganda.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)

print(f"[INFO] Total propaganda examples: {len(df_propaganda)}")

# Partition: 50% for similarity reference, 50% for testing
np.random.seed(42)
indices = np.random.permutation(len(df_propaganda))
split_point = len(df_propaganda) // 2

reference_indices = indices[:split_point]
test_indices = indices[split_point:]

df_propaganda_reference = df_propaganda.iloc[reference_indices].reset_index(drop=True)
df_propaganda_test = df_propaganda.iloc[test_indices].reset_index(drop=True)

# Limit test set to PROPAGANDA_TEST_SIZE
if len(df_propaganda_test) > PROPAGANDA_TEST_SIZE:
    df_propaganda_test = df_propaganda_test.sample(n=PROPAGANDA_TEST_SIZE, random_state=42).reset_index(drop=True)

print(f"[INFO] Reference set (for similarity): {len(df_propaganda_reference)}")
print(f"[INFO] Test set (for evaluation): {len(df_propaganda_test)}")

# ==== ENCODE REFERENCE PROPAGANDA ====
print("\n[INFO] Encoding reference propaganda examples...")
propaganda_reference_texts = df_propaganda_reference['cleaned_text'].tolist()
propaganda_reference_embeddings = model.encode(
    propaganda_reference_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    device=device
)
if device == 'cuda':
    propaganda_reference_embeddings = propaganda_reference_embeddings.float().cpu().numpy()
else:
    propaganda_reference_embeddings = propaganda_reference_embeddings.numpy()

print(f"[INFO] Reference embeddings shape: {propaganda_reference_embeddings.shape}")

# ==== LOAD AND ENCODE NON-PROPAGANDA TEST DATA ====
print("\n" + "="*80)
print("STEP 2: LOADING NON-PROPAGANDA TEST DATA")
print("="*80)

nonpolitical_sentences = load_nonpolitical_sentences()

# Limit to NONPROPAGANDA_TEST_SIZE for balanced evaluation
if len(nonpolitical_sentences) > NONPROPAGANDA_TEST_SIZE:
    np.random.seed(42)
    nonpolitical_sentences = np.random.choice(nonpolitical_sentences, NONPROPAGANDA_TEST_SIZE, replace=False).tolist()

print(f"[INFO] Using {len(nonpolitical_sentences)} non-propaganda sentences for testing")

# ==== ENCODE TEST DATA ====
print("\n" + "="*80)
print("STEP 3: ENCODING TEST DATA")
print("="*80)

# Encode propaganda test set
print("[INFO] Encoding propaganda test set...")
propaganda_test_texts = df_propaganda_test['cleaned_text'].tolist()
propaganda_test_embeddings = model.encode(
    propaganda_test_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    device=device
)
if device == 'cuda':
    propaganda_test_embeddings = propaganda_test_embeddings.float().cpu().numpy()
else:
    propaganda_test_embeddings = propaganda_test_embeddings.numpy()

# Encode non-propaganda test set
print("[INFO] Encoding non-propaganda test set...")
nonpropaganda_test_embeddings = model.encode(
    nonpolitical_sentences,
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    device=device
)
if device == 'cuda':
    nonpropaganda_test_embeddings = nonpropaganda_test_embeddings.float().cpu().numpy()
else:
    nonpropaganda_test_embeddings = nonpropaganda_test_embeddings.numpy()

# ==== COMPUTE SIMILARITIES ====
print("\n" + "="*80)
print("STEP 4: COMPUTING SIMILARITIES")
print("="*80)

print("[INFO] Computing similarities for propaganda test set...")
propaganda_similarities = cosine_similarity(propaganda_test_embeddings, propaganda_reference_embeddings)
propaganda_max_similarities = np.max(propaganda_similarities, axis=1)
propaganda_best_match_indices = np.argmax(propaganda_similarities, axis=1)

print("[INFO] Computing similarities for non-propaganda test set...")
nonpropaganda_similarities = cosine_similarity(nonpropaganda_test_embeddings, propaganda_reference_embeddings)
nonpropaganda_max_similarities = np.max(nonpropaganda_similarities, axis=1)
nonpropaganda_best_match_indices = np.argmax(nonpropaganda_similarities, axis=1)

# ==== EVALUATE THRESHOLDS ====
print("\n" + "="*80)
print("STEP 5: EVALUATING THRESHOLDS")
print("="*80)

# Create combined test set
y_true = np.concatenate([
    np.ones(len(propaganda_max_similarities)),
    np.zeros(len(nonpropaganda_max_similarities))
])

all_max_similarities = np.concatenate([
    propaganda_max_similarities,
    nonpropaganda_max_similarities
])

threshold_results = []

print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
print("-" * 60)

for threshold in THRESHOLDS:
    y_pred = (all_max_similarities >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    threshold_results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    })
    
    print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {accuracy:<12.4f}")

# Save threshold results
df_threshold_results = pd.DataFrame(threshold_results)
df_threshold_results.to_csv(OUTPUT_THRESHOLD_RESULTS, index=False)
print(f"\n[INFO] Threshold evaluation results saved to {OUTPUT_THRESHOLD_RESULTS}")

# Find best threshold by F1 score
best_threshold_idx = df_threshold_results['f1_score'].idxmax()
best_threshold = df_threshold_results.loc[best_threshold_idx, 'threshold']
best_f1 = df_threshold_results.loc[best_threshold_idx, 'f1_score']

print(f"\n{'='*80}")
print(f"BEST THRESHOLD: {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
print(f"{'='*80}")

# ==== ANALYZE POLITICAL SPEECHES ====
print("\n" + "="*80)
print("STEP 6: ANALYZING POLITICAL SPEECHES")
print("="*80)

speech_files = glob.glob(os.path.join(SPEECH_FOLDER, "*.txt"))

if not speech_files:
    print(f"[WARNING] No speech files found in '{SPEECH_FOLDER}' folder")
    speech_analysis_results = []
    speech_flagged_sentences = []
else:
    print(f"[INFO] Found {len(speech_files)} speech file(s)")
    
    speech_analysis_results = []
    speech_flagged_sentences = []
    
    for speech_file in speech_files:
        speech_name = os.path.basename(speech_file).replace('.txt', '')
        print(f"\n[INFO] Processing: {speech_name}")
        
        with open(speech_file, 'r', encoding='utf-8') as f:
            speech_text = f.read()
        
        cleaned_speech = clean_text(speech_text)
        sentences = split_into_sentences(cleaned_speech)
        
        print(f"[INFO] Extracted {len(sentences)} sentences")
        
        if len(sentences) == 0:
            continue
        
        # Encode
        speech_embeddings = model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device
        )
        if device == 'cuda':
            speech_embeddings = speech_embeddings.float().cpu().numpy()
        else:
            speech_embeddings = speech_embeddings.numpy()
        
        # Compute similarities
        similarities = cosine_similarity(speech_embeddings, propaganda_reference_embeddings)
        max_similarities = np.max(similarities, axis=1)
        best_match_indices = np.argmax(similarities, axis=1)
        
        # Evaluate at each threshold
        for threshold in THRESHOLDS:
            predictions = (max_similarities >= threshold).astype(int)
            n_flagged = np.sum(predictions)
            
            flagged_mask = predictions == 1
            flagged_similarities = max_similarities[flagged_mask]
            
            result = {
                'source': speech_name,
                'source_type': 'political_speech',
                'threshold': threshold,
                'total_sentences': len(sentences),
                'flagged_sentences': n_flagged,
                'coverage_percentage': n_flagged / len(sentences) if len(sentences) > 0 else 0,
                'avg_similarity': np.mean(flagged_similarities) if len(flagged_similarities) > 0 else 0,
                'min_similarity': np.min(flagged_similarities) if len(flagged_similarities) > 0 else 0,
                'max_similarity': np.max(flagged_similarities) if len(flagged_similarities) > 0 else 0
            }
            speech_analysis_results.append(result)
        
        # Record flagged sentences at best threshold
        best_threshold_predictions = (max_similarities >= best_threshold).astype(int)
        for i, (sentence, similarity, prediction, match_idx) in enumerate(zip(
            sentences, max_similarities, best_threshold_predictions, best_match_indices
        )):
            if prediction == 1:
                matched_article_id = df_propaganda_reference.loc[match_idx, 'article_id']
                matched_text = df_propaganda_reference.loc[match_idx, 'cleaned_text']
                
                speech_flagged_sentences.append({
                    'source': speech_name,
                    'source_type': 'political_speech',
                    'sentence_index': i,
                    'sentence': sentence,
                    'similarity_score': similarity,
                    'threshold_used': best_threshold,
                    'matched_article_id': matched_article_id,
                    'matched_propaganda_text': matched_text[:200]
                })

# Save speech analysis results
df_speech_analysis = pd.DataFrame(speech_analysis_results)
df_speech_flagged = pd.DataFrame(speech_flagged_sentences)

if len(df_speech_analysis) > 0:
    df_speech_analysis.to_csv(OUTPUT_POLITICAL_ANALYSIS, index=False)
    print(f"\n[INFO] Political speech analysis saved to {OUTPUT_POLITICAL_ANALYSIS}")

if len(df_speech_flagged) > 0:
    df_speech_flagged.to_csv(OUTPUT_POLITICAL_FLAGGED, index=False)
    print(f"[INFO] Flagged political sentences saved to {OUTPUT_POLITICAL_FLAGGED}")
    print(f"[INFO] Total flagged sentences: {len(df_speech_flagged)}")

# ==== ANALYZE NON-POLITICAL TEXT ====
print("\n" + "="*80)
print("STEP 7: ANALYZING NON-POLITICAL TEXT")
print("="*80)

nonpolitical_analysis_results = []
nonpolitical_flagged_sentences = []

# Evaluate at each threshold
for threshold in THRESHOLDS:
    predictions = (nonpropaganda_max_similarities >= threshold).astype(int)
    n_flagged = np.sum(predictions)
    
    flagged_mask = predictions == 1
    flagged_similarities = nonpropaganda_max_similarities[flagged_mask]
    
    result = {
        'source': 'HuggingFace_Datasets',
        'source_type': 'nonpolitical_text',
        'threshold': threshold,
        'total_sentences': len(nonpolitical_sentences),
        'flagged_sentences': n_flagged,
        'coverage_percentage': n_flagged / len(nonpolitical_sentences) if len(nonpolitical_sentences) > 0 else 0,
        'avg_similarity': np.mean(flagged_similarities) if len(flagged_similarities) > 0 else 0,
        'min_similarity': np.min(flagged_similarities) if len(flagged_similarities) > 0 else 0,
        'max_similarity': np.max(flagged_similarities) if len(flagged_similarities) > 0 else 0
    }
    nonpolitical_analysis_results.append(result)

# Record flagged sentences at best threshold
best_threshold_predictions = (nonpropaganda_max_similarities >= best_threshold).astype(int)
for i, (sentence, similarity, prediction, match_idx) in enumerate(zip(
    nonpolitical_sentences, nonpropaganda_max_similarities, best_threshold_predictions, nonpropaganda_best_match_indices
)):
    if prediction == 1:
        matched_article_id = df_propaganda_reference.loc[match_idx, 'article_id']
        matched_text = df_propaganda_reference.loc[match_idx, 'cleaned_text']
        
        nonpolitical_flagged_sentences.append({
            'source': 'HuggingFace_Datasets',
            'source_type': 'nonpolitical_text',
            'sentence_index': i,
            'sentence': sentence,
            'similarity_score': similarity,
            'threshold_used': best_threshold,
            'matched_article_id': matched_article_id,
            'matched_propaganda_text': matched_text[:200]
        })

# Save non-political analysis results
df_nonpolitical_analysis = pd.DataFrame(nonpolitical_analysis_results)
df_nonpolitical_flagged = pd.DataFrame(nonpolitical_flagged_sentences)

df_nonpolitical_analysis.to_csv(OUTPUT_NONPOLITICAL_ANALYSIS, index=False)
print(f"\n[INFO] Non-political text analysis saved to {OUTPUT_NONPOLITICAL_ANALYSIS}")

df_nonpolitical_flagged.to_csv(OUTPUT_NONPOLITICAL_FLAGGED, index=False)
print(f"[INFO] Flagged non-political sentences saved to {OUTPUT_NONPOLITICAL_FLAGGED}")
print(f"[INFO] Total flagged sentences: {len(df_nonpolitical_flagged)}")

# ==== VISUALIZATIONS ====
print("\n" + "="*80)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*80)

# Figure 1: Threshold Evaluation Metrics
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Threshold Evaluation Metrics', fontsize=16, fontweight='bold')

axes1[0, 0].plot(df_threshold_results['threshold'], df_threshold_results['precision'], marker='o', linewidth=2)
axes1[0, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes1[0, 0].set_title('Precision vs Threshold')
axes1[0, 0].set_xlabel('Threshold')
axes1[0, 0].set_ylabel('Precision')
axes1[0, 0].legend()
axes1[0, 0].grid(True)

axes1[0, 1].plot(df_threshold_results['threshold'], df_threshold_results['recall'], marker='o', linewidth=2, color='orange')
axes1[0, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes1[0, 1].set_title('Recall vs Threshold')
axes1[0, 1].set_xlabel('Threshold')
axes1[0, 1].set_ylabel('Recall')
axes1[0, 1].legend()
axes1[0, 1].grid(True)

axes1[1, 0].plot(df_threshold_results['threshold'], df_threshold_results['f1_score'], marker='o', linewidth=2, color='green')
axes1[1, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes1[1, 0].set_title('F1-Score vs Threshold')
axes1[1, 0].set_xlabel('Threshold')
axes1[1, 0].set_ylabel('F1-Score')
axes1[1, 0].legend()
axes1[1, 0].grid(True)

axes1[1, 1].plot(df_threshold_results['threshold'], df_threshold_results['accuracy'], marker='o', linewidth=2, color='purple')
axes1[1, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes1[1, 1].set_title('Accuracy vs Threshold')
axes1[1, 1].set_xlabel('Threshold')
axes1[1, 1].set_ylabel('Accuracy')
axes1[1, 1].legend()
axes1[1, 1].grid(True)

plt.tight_layout()
plt.savefig('ThresholdData/threshold_evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved: threshold_evaluation_metrics.png")
plt.close()

# Figure 2: Political Speeches Analysis
if len(df_speech_analysis) > 0:
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Political Speeches Analysis', fontsize=16, fontweight='bold')
    
    for speech_name in df_speech_analysis['source'].unique():
        speech_data = df_speech_analysis[df_speech_analysis['source'] == speech_name]
        
        axes2[0, 0].plot(speech_data['threshold'], speech_data['coverage_percentage'], marker='o', label=speech_name)
        axes2[0, 1].plot(speech_data['threshold'], speech_data['flagged_sentences'], marker='o', label=speech_name)
        axes2[1, 0].plot(speech_data['threshold'], speech_data['avg_similarity'], marker='s', label=speech_name)
    
    axes2[0, 0].axvline(best_threshold, color='r', linestyle='--', alpha=0.5, label='Best Threshold')
    axes2[0, 0].set_title('Coverage vs Threshold')
    axes2[0, 0].set_xlabel('Threshold')
    axes2[0, 0].set_ylabel('Coverage (%)')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True)
    
    axes2[0, 1].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)
    axes2[0, 1].set_title('Flagged Sentences vs Threshold')
    axes2[0, 1].set_xlabel('Threshold')
    axes2[0, 1].set_ylabel('Flagged Sentences')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True)
    
    axes2[1, 0].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)
    axes2[1, 0].set_title('Average Similarity of Flagged')
    axes2[1, 0].set_xlabel('Threshold')
    axes2[1, 0].set_ylabel('Avg Similarity')
    axes2[1, 0].legend()
    axes2[1, 0].grid(True)
    
    # Bar chart at best threshold
    best_speech_data = df_speech_analysis[df_speech_analysis['threshold'] == best_threshold]
    axes2[1, 1].bar(best_speech_data['source'], best_speech_data['flagged_sentences'])
    axes2[1, 1].set_title(f'Flagged at Best Threshold ({best_threshold:.2f})')
    axes2[1, 1].set_xlabel('Speech')
    axes2[1, 1].set_ylabel('Flagged Sentences')
    axes2[1, 1].tick_params(axis='x', rotation=45)
    axes2[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('ThresholdData/political_speeches_analysis.png', dpi=300, bbox_inches='tight')
    print("[INFO] Saved: political_speeches_analysis.png")
    plt.close()

# Figure 3: Non-Political Text Analysis
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('Non-Political Text Analysis', fontsize=16, fontweight='bold')

axes3[0, 0].plot(df_nonpolitical_analysis['threshold'], df_nonpolitical_analysis['coverage_percentage'], marker='o', linewidth=2)
axes3[0, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes3[0, 0].set_title('Coverage vs Threshold')
axes3[0, 0].set_xlabel('Threshold')
axes3[0, 0].set_ylabel('Coverage (%)')
axes3[0, 0].legend()
axes3[0, 0].grid(True)

axes3[0, 1].plot(df_nonpolitical_analysis['threshold'], df_nonpolitical_analysis['flagged_sentences'], marker='o', linewidth=2)
axes3[0, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes3[0, 1].set_title('Flagged Sentences vs Threshold')
axes3[0, 1].set_xlabel('Threshold')
axes3[0, 1].set_ylabel('Flagged Sentences')
axes3[0, 1].legend()
axes3[0, 1].grid(True)

axes3[1, 0].plot(df_nonpolitical_analysis['threshold'], df_nonpolitical_analysis['avg_similarity'], marker='s', linewidth=2)
axes3[1, 0].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes3[1, 0].set_title('Average Similarity of Flagged')
axes3[1, 0].set_xlabel('Threshold')
axes3[1, 0].set_ylabel('Avg Similarity')
axes3[1, 0].legend()
axes3[1, 0].grid(True)

# False positive rate (same as coverage for non-political)
axes3[1, 1].plot(df_nonpolitical_analysis['threshold'], df_nonpolitical_analysis['coverage_percentage'], marker='o', linewidth=2, color='red')
axes3[1, 1].axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
axes3[1, 1].set_title('False Positive Rate')
axes3[1, 1].set_xlabel('Threshold')
axes3[1, 1].set_ylabel('FP Rate')
axes3[1, 1].legend()
axes3[1, 1].grid(True)

plt.tight_layout()
plt.savefig('ThresholdData/nonpolitical_text_analysis.png', dpi=300, bbox_inches='tight')
print("[INFO] Saved: nonpolitical_text_analysis.png")
plt.close()

# ==== FINAL SUMMARY ====
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nBest Threshold (by F1-Score): {best_threshold:.2f}")
print(f"  - Precision: {df_threshold_results.loc[best_threshold_idx, 'precision']:.4f}")
print(f"  - Recall: {df_threshold_results.loc[best_threshold_idx, 'recall']:.4f}")
print(f"  - F1-Score: {df_threshold_results.loc[best_threshold_idx, 'f1_score']:.4f}")
print(f"  - Accuracy: {df_threshold_results.loc[best_threshold_idx, 'accuracy']:.4f}")

print(f"\nConfusion Matrix at Best Threshold:")
print(f"  True Positives:  {df_threshold_results.loc[best_threshold_idx, 'true_positives']}")
print(f"  False Positives: {df_threshold_results.loc[best_threshold_idx, 'false_positives']}")
print(f"  True Negatives:  {df_threshold_results.loc[best_threshold_idx, 'true_negatives']}")
print(f"  False Negatives: {df_threshold_results.loc[best_threshold_idx, 'false_negatives']}")

if len(df_speech_flagged) > 0:
    print(f"\nPolitical Speeches:")
    print(f"  Total flagged sentences: {len(df_speech_flagged)}")
    print(f"  Average similarity: {df_speech_flagged['similarity_score'].mean():.4f}")
    print(f"  Top 5 most similar sentences:")
    for idx, row in df_speech_flagged.nlargest(5, 'similarity_score').iterrows():
        print(f"    [{row['similarity_score']:.3f}] {row['sentence'][:80]}...")
        print(f"      → Matched: {row['matched_article_id']} | {row['matched_propaganda_text'][:60]}...")

print(f"\nNon-Political Text:")
print(f"  Total flagged sentences: {len(df_nonpolitical_flagged)}")
if len(df_nonpolitical_flagged) > 0:
    print(f"  Average similarity: {df_nonpolitical_flagged['similarity_score'].mean():.4f}")
    print(f"  Top 5 most similar sentences:")
    for idx, row in df_nonpolitical_flagged.nlargest(5, 'similarity_score').iterrows():
        print(f"    [{row['similarity_score']:.3f}] {row['sentence'][:80]}...")
        print(f"      → Matched: {row['matched_article_id']} | {row['matched_propaganda_text'][:60]}...")

print("\n" + "="*80)
print("OUTPUT FILES GENERATED:")
print("="*80)
print(f"1. {OUTPUT_THRESHOLD_RESULTS} - Threshold evaluation metrics")
print(f"2. {OUTPUT_POLITICAL_ANALYSIS} - Political speeches analysis by threshold")
print(f"3. {OUTPUT_POLITICAL_FLAGGED} - Flagged sentences from political speeches")
print(f"4. {OUTPUT_NONPOLITICAL_ANALYSIS} - Non-political text analysis by threshold")
print(f"5. {OUTPUT_NONPOLITICAL_FLAGGED} - Flagged sentences from non-political text")
print(f"6. threshold_evaluation_metrics.png - Visualization of metrics")
if len(df_speech_analysis) > 0:
    print(f"7. political_speeches_analysis.png - Political speeches visualization")
print(f"8. nonpolitical_text_analysis.png - Non-political text visualization")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print(f"\nBased on F1-Score optimization:")
print(f"    Use threshold: {best_threshold:.2f}")
print(f"    Expected precision: {df_threshold_results.loc[best_threshold_idx, 'precision']:.2%}")
print(f"    Expected recall: {df_threshold_results.loc[best_threshold_idx, 'recall']:.2%}")

print("\nNext Steps:")
print("  1. Review flagged sentences in CSV files")
print("  2. Manually validate a sample of flagged propaganda")
print("  3. Check matched_article_id to see which propaganda examples are matching")
print("  4. If results are unsatisfactory, consider:")
print("     - Different aggregation methods (top-5, top-10 average)")
print("     - Using percentile thresholds instead of fixed values")
print("     - Fine-tuning a classifier instead of similarity matching")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)