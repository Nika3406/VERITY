import praw
import pandas as pd
import os
import time
import torch
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==== CONFIG ====
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

SEMEVAL_FILE = "semeval_examples.csv"
OUTPUT_FILE = "S-Dataset.csv"

# Target political/news subreddits to find relevant content
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

LIMIT_POSTS_PER_SUB = 500  # Posts per subreddit
MAX_COMMENTS_PER_POST = 20
SIMILARITY_THRESHOLD = 0.60
BATCH_SIZE = 32
USE_GPU = True
COMMENT_FETCH_DELAY = 0.3

device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# ==== LOAD MODEL ====
print("[INFO] Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
if device == 'cuda':
    model.half()
print("[INFO] Model loaded successfully.")

# ==== AUTH ====
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ==== LOAD NORMATIVE SENTENCES ====
if not os.path.exists(SEMEVAL_FILE):
    raise FileNotFoundError(f"{SEMEVAL_FILE} not found!")

df_norm = pd.read_csv(SEMEVAL_FILE)
norm_texts = df_norm["text_snippet"].dropna().astype(str).tolist()
norm_texts = [t for t in norm_texts if 10 <= len(t) <= 500]
print(f"[INFO] Loaded {len(norm_texts)} normative examples...")

print("[INFO] Encoding normative examples...")
norm_embeddings = model.encode(
    norm_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_tensor=False,
    device=device
)

# ==== STEP 1: COLLECT ALL DATA ====
print(f"\n[PHASE 1: COLLECTING DATA FROM {len(TARGET_SUBREDDITS)} POLITICAL SUBREDDITS]")
all_items = []

for subreddit_name in TARGET_SUBREDDITS:
    print(f"\n[SCRAPING] r/{subreddit_name}")
    try:
        subreddit = reddit.subreddit(subreddit_name)
        submissions = list(subreddit.hot(limit=LIMIT_POSTS_PER_SUB))
        
        for sub in tqdm(submissions, desc=f"r/{subreddit_name}"):
            title = sub.title or ""
            body = sub.selftext or ""
            
            # Collect post
            post_text = title + " " + body
            if len(post_text.strip()) >= 20:
                all_items.append({
                    "keyword": None,
                    "subreddit": subreddit_name,
                    "content_type": "post",
                    "post_id": sub.id,
                    "post_title": title,
                    "text": post_text,
                    "score": sub.score,
                    "url": sub.url,
                    "created_utc": sub.created_utc,
                    "id": sub.id,
                    "author": str(sub.author) if sub.author else "[deleted]"
                })

            # Collect comments with rate limiting
            try:
                sub.comments.replace_more(limit=0)
                comment_count = 0
                for c in sub.comments.list():
                    if comment_count >= MAX_COMMENTS_PER_POST:
                        break
                    if hasattr(c, "body") and c.body not in ["[deleted]", "[removed]"]:
                        body = c.body.strip()
                        if len(body) >= 20:
                            all_items.append({
                                "keyword": None,
                                "subreddit": subreddit_name,
                                "content_type": "comment",
                                "post_id": sub.id,
                                "post_title": title,
                                "text": body,
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
                    print(f"[WARN] Comment fetch error: {e}")
        
        time.sleep(2)
        
    except Exception as e:
        print(f"[ERROR] Failed to scrape r/{subreddit_name}: {e}")
        time.sleep(5)

print(f"\n[INFO] Collected {len(all_items)} total items")

# ==== STEP 2: BATCH COMPUTE SIMILARITIES ====
print(f"\n[PHASE 2: COMPUTING SIMILARITIES]")

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
similarities = cosine_similarity(text_embeddings, norm_embeddings)
max_similarities = np.max(similarities, axis=1)

# Add similarity scores
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
    
    df = pd.DataFrame(columns=["keyword", "subreddit", "content_type", "post_id", "post_title", 
                                "text", "score", "url", "created_utc", "id", "author", "similarity_score"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[DONE] Saved empty dataset to {OUTPUT_FILE}")
else:
    df = pd.DataFrame(filtered_items)
    df.drop_duplicates(subset="id", inplace=True)
    df = df.sort_values("similarity_score", ascending=False)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    unique_subreddits = df['subreddit'].nunique()
    post_count = len(df[df['content_type'] == 'post'])
    comment_count = len(df[df['content_type'] == 'comment'])

    print(f"\n[DONE] Saved {len(df)} items to {OUTPUT_FILE}")
    print(f"[STATS] Posts: {post_count}, Comments: {comment_count}")
    print(f"[STATS] Found across {unique_subreddits} different subreddits")
    print(f"[STATS] Similarity - Min: {df['similarity_score'].min():.3f}, Max: {df['similarity_score'].max():.3f}, Mean: {df['similarity_score'].mean():.3f}")