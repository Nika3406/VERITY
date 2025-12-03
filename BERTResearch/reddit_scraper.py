import praw
import pandas as pd
import os
import time
import random
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==== CONFIG ====
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

KEYWORDS_FILE = "keywords.txt"
SEMEVAL_FILE = "semeval_examples.csv"
OUTPUT_FILE = "C-Dataset.csv"

# Target political/news subreddits for relevant content
TARGET_SUBREDDITS = [
    "politics",
    "worldnews",
    "news",
    "PoliticalDiscussion",
    "conservative",
    "Liberal",
    "democrats",
    "Republican",
    "Libertarian",
    "socialism",
    "PoliticalCompassMemes",
    "CapitalismVSocialism",
    "NeutralPolitics",
    "changemyview",
    "unpopularopinion",
    "conspiracy",
    "JoeRogan",
    "PoliticalHumor",
    "IntellectualDarkWeb",
    "TrueOffMyChest"
]

LIMIT_PER_SEARCH = 500
MIN_SLEEP = 2
MAX_SLEEP = 4
SIMILARITY_THRESHOLD = 0.60
MAX_COMMENTS_PER_POST = 20
SCRAPE_COMMENTS = True
BATCH_SIZE = 32
USE_GPU = True
COMMENT_FETCH_DELAY = 0.3

# ==== GPU SETUP ====
device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

if device == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==== LOAD EMBEDDING MODEL ====
print("[INFO] Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
if device == 'cuda':
    model.half()
print("[INFO] Model loaded successfully on", device)

# ==== AUTH ====
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ==== LOAD DATA ====
keywords = []
if os.path.exists(KEYWORDS_FILE):
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        keywords = [line.strip().lower() for line in f if line.strip()]

propaganda_embeddings = None
if os.path.exists(SEMEVAL_FILE):
    df = pd.read_csv(SEMEVAL_FILE)
    propaganda_texts = df["text_snippet"].dropna().astype(str).tolist()
    propaganda_texts = [p for p in propaganda_texts if 10 <= len(p) <= 500]
    propaganda_texts = list(set(propaganda_texts))
    
    print(f"[INFO] Encoding {len(propaganda_texts)} propaganda examples...")
    propaganda_embeddings = model.encode(
        propaganda_texts, 
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    if device == 'cuda':
        propaganda_embeddings = propaganda_embeddings.cpu().numpy()
    else:
        propaganda_embeddings = propaganda_embeddings.numpy()
    print(f"[INFO] Embeddings shape: {propaganda_embeddings.shape}")
else:
    print("[WARN] No propaganda examples file found")

print(f"[INFO] Loaded {len(keywords)} keywords and targeting {len(TARGET_SUBREDDITS)} subreddits")

# ==== STEP 1: COLLECT ALL POSTS/COMMENTS ====
print("\n[PHASE 1: COLLECTING DATA]")
all_items = []

for subreddit_name in TARGET_SUBREDDITS:
    print(f"\n[SCRAPING SUBREDDIT] r/{subreddit_name}")
    for kw in keywords:
        print(f"  [SEARCH] '{kw}' in r/{subreddit_name}")
        try:
            submissions = list(reddit.subreddit(subreddit_name).search(kw, limit=LIMIT_PER_SEARCH, sort="new"))
            
            with tqdm(total=len(submissions), desc=f"r/{subreddit_name}/{kw}") as pbar:
                for sub in submissions:
                    title = sub.title or ""
                    body = sub.selftext or ""
                    combined = title + " " + body
                    
                    # Collect post
                    if len(combined.strip()) >= 20:
                        all_items.append({
                            "keyword": kw,
                            "subreddit": subreddit_name,
                            "post_id": sub.id,
                            "post_title": title,
                            "content_type": "post",
                            "text": body,
                            "score": sub.score,
                            "url": sub.url,
                            "created_utc": sub.created_utc,
                            "id": sub.id,
                            "author": str(sub.author) if sub.author else "[deleted]"
                        })
                    
                    # Collect comments
                    if SCRAPE_COMMENTS:
                        try:
                            sub.comments.replace_more(limit=0)
                            comment_count = 0
                            for c in sub.comments.list():
                                if comment_count >= MAX_COMMENTS_PER_POST:
                                    break
                                if hasattr(c, 'body') and c.body:
                                    if c.body not in ['[deleted]', '[removed]'] and len(c.body.strip()) >= 20:
                                        all_items.append({
                                            "keyword": kw,
                                            "subreddit": subreddit_name,
                                            "post_id": sub.id,
                                            "post_title": title,
                                            "content_type": "comment",
                                            "text": c.body,
                                            "score": c.score,
                                            "url": f"https://reddit.com{c.permalink}",
                                            "created_utc": c.created_utc,
                                            "id": c.id,
                                            "author": str(c.author) if c.author else "[deleted]"
                                        })
                                        comment_count += 1
                            
                            time.sleep(COMMENT_FETCH_DELAY)
                        except Exception as e:
                            if "429" in str(e):
                                print(f"[WARN] Rate limited, sleeping 60s...")
                                time.sleep(60)
                            else:
                                print(f"[WARN] Comment error: {e}")
                    
                    pbar.update(1)
            
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
        
        except Exception as e:
            print(f"[WARN] Error searching '{kw}' in r/{subreddit_name}:", e)
            time.sleep(10)

print(f"\n[INFO] Collected {len(all_items)} total items")

# ==== STEP 2: BATCH COMPUTE SIMILARITIES ====
print("\n[PHASE 2: COMPUTING SIMILARITIES]")

texts = [item['text'] for item in all_items]

print(f"[INFO] Encoding {len(texts)} texts in batches...")
text_embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_tensor=False,
    device=device
)

print(f"[INFO] Computing cosine similarities...")
similarities = cosine_similarity(text_embeddings, propaganda_embeddings)
max_similarities = np.max(similarities, axis=1)

# Add similarity scores to items
for i, item in enumerate(all_items):
    item['similarity_score'] = float(max_similarities[i])

# ==== STEP 3: FILTER BY THRESHOLD ====
print(f"\n[PHASE 3: FILTERING BY THRESHOLD >= {SIMILARITY_THRESHOLD}]")
filtered_items = [item for item in all_items if item['similarity_score'] >= SIMILARITY_THRESHOLD]
print(f"[INFO] {len(filtered_items)} items passed similarity threshold")

# ==== SAVE ====
if len(filtered_items) == 0:
    print(f"\n[WARNING] No items passed the similarity threshold of {SIMILARITY_THRESHOLD}")
    print(f"[INFO] Total items collected: {len(all_items)}")
    print(f"[INFO] Top 20 similarity scores from collected data:")
    all_scores = sorted([item['similarity_score'] for item in all_items], reverse=True)[:20]
    for i, score in enumerate(all_scores, 1):
        print(f"  {i}. {score:.4f}")
    
    if all_scores:
        suggested_threshold = max(0.35, all_scores[min(9, len(all_scores)-1)] * 0.95)
        print(f"\n[SUGGESTION] Consider setting SIMILARITY_THRESHOLD to around {suggested_threshold:.2f}")
        print(f"[SUGGESTION] This would give you approximately {sum(1 for item in all_items if item['similarity_score'] >= suggested_threshold)} results")
    
    df = pd.DataFrame(columns=["keyword", "subreddit", "post_id", "post_title", "content_type",
                                "text", "score", "url", "created_utc", "id", "similarity_score", "author"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[DONE] Saved empty dataset to {OUTPUT_FILE}")
else:
    df = pd.DataFrame(filtered_items)
    df.drop_duplicates(subset="id", inplace=True)
    df = df.sort_values("similarity_score", ascending=False)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    post_count = len(df[df['content_type'] == 'post'])
    comment_count = len(df[df['content_type'] == 'comment'])
    unique_subreddits = df['subreddit'].nunique()

    print(f"\n[DONE] Saved {len(df)} items to {OUTPUT_FILE}")
    print(f"[STATS] Posts: {post_count}, Comments: {comment_count}")
    print(f"[STATS] Found across {unique_subreddits} different subreddits")
    print(f"[STATS] Similarity - Min: {df['similarity_score'].min():.3f}, Max: {df['similarity_score'].max():.3f}, Mean: {df['similarity_score'].mean():.3f}")

if device == 'cuda':
    print(f"[GPU] Max memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
