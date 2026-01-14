import praw
import pandas as pd
import os
import time
import torch
import hashlib
import random
import re
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# ===================== CONFIG =====================
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

SEMEVAL_FILE = "semeval_examples.csv"
DATASET_DIR = "datasets"
DISCOVERY_FILE = "discovered_subreddits.txt"

LIMIT_POSTS = 1200
MAX_COMMENTS_PER_POST = 60

SIMILARITY_THRESHOLD = 0.60
ENCODE_BATCH = 128
PIPELINE_BATCH = 512
NORM_CHUNK = 512

USE_GPU = True
COMMENT_FETCH_DELAY = 0.25

# UPDATED: More lenient thresholds to get more data
MIN_SUBS_REQUIRED = 5_000  # Lowered from 10k/50k
MIN_POSTS_REQUIRED = 30     # Lowered from 50
MIN_ACTIVE_RATIO = 0.20     # Lowered from 0.33

# ===================== DEVICE =====================
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ===================== MODEL =====================
print("[INFO] Loading sentence embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
if device == "cuda":
    model.half()
print("[INFO] Model loaded.")

# ===================== REDDIT =====================
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ===================== LOAD NORMATIVE DATA =====================
if not os.path.exists(SEMEVAL_FILE):
    raise FileNotFoundError(SEMEVAL_FILE)

df_norm = pd.read_csv(SEMEVAL_FILE)
norm_texts = df_norm["text_snippet"].dropna().astype(str).tolist()
norm_texts = [t for t in norm_texts if 10 <= len(t) <= 500]

print(f"[INFO] Encoding {len(norm_texts)} normative examples...")
norm_emb = model.encode(
    norm_texts,
    batch_size=ENCODE_BATCH,
    convert_to_tensor=True,
    device=device,
    show_progress_bar=True
)
norm_emb = F.normalize(norm_emb, dim=1)

# ===================== DATASET HISTORY =====================
os.makedirs(DATASET_DIR, exist_ok=True)

processed_subs = {
    f.split("_r_")[1].replace(".csv", "")
    for f in os.listdir(DATASET_DIR)
    if f.startswith("S-Dataset_r_")
}

print(f"[INFO] Found {len(processed_subs)} processed subreddits")

# ===================== LOAD DISCOVERED SUBREDDITS =====================
if not os.path.exists(DISCOVERY_FILE):
    print(f"[ERROR] {DISCOVERY_FILE} not found!")
    print("[ERROR] Run the discovery script first.")
    exit(1)

with open(DISCOVERY_FILE) as f:
    discovered_subs = set(line.strip() for line in f if line.strip())

print(f"[INFO] Loaded {len(discovered_subs)} discovered subreddits")

# ===================== SUBREDDIT PICKER WITH BETTER LOGIC =====================
def select_next_subreddit():
    candidates = list(discovered_subs - processed_subs)
    random.shuffle(candidates)
    
    checked = 0
    max_checks = min(100, len(candidates))  # Check up to 100 subs
    
    for name in candidates[:max_checks]:
        checked += 1
        try:
            sub = reddit.subreddit(name)
            
            # Skip if too small
            if sub.subscribers < MIN_SUBS_REQUIRED:
                continue
            
            # Check for recent activity
            posts = list(sub.new(limit=MIN_POSTS_REQUIRED))
            if len(posts) < MIN_POSTS_REQUIRED:
                continue
            
            # Check if posts have comments (active discussion)
            active = sum(1 for p in posts if p.num_comments > 0)
            if active < int(MIN_POSTS_REQUIRED * MIN_ACTIVE_RATIO):
                continue
            
            print(f"[SELECTED] r/{name} (subs={sub.subscribers:,}, checked={checked})")
            return name
            
        except Exception as e:
            continue
    
    print(f"[WARNING] No suitable sub found after checking {checked} candidates")
    return None

subreddit_name = select_next_subreddit()
if subreddit_name is None:
    print("[EXIT] No more suitable subreddits available.")
    print("[TIP] Run discovery script again to find more subs.")
    exit(0)

OUTPUT_FILE = os.path.join(DATASET_DIR, f"S-Dataset_r_{subreddit_name}.csv")

# ===================== HELPERS =====================
seen_hashes = set()
results = []

def text_hash(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()

def process_batch(items, texts):
    if not texts:
        return
    
    emb = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        convert_to_tensor=True,
        device=device
    )
    emb = F.normalize(emb, dim=1)
    
    max_sim = torch.zeros(len(texts), device=device)
    
    for i in range(0, norm_emb.size(0), NORM_CHUNK):
        sims = emb @ norm_emb[i:i + NORM_CHUNK].T
        max_sim = torch.maximum(max_sim, sims.max(dim=1).values)
    
    scores = max_sim.cpu().numpy()
    
    for i, s in enumerate(scores):
        if s >= SIMILARITY_THRESHOLD:
            item = items[i]
            item["similarity_score"] = float(s)
            results.append(item)

# ===================== SCRAPE =====================
buffer_items, buffer_texts = [], []

print(f"[SCRAPING] r/{subreddit_name}")
subreddit = reddit.subreddit(subreddit_name)

for sub in tqdm(subreddit.new(limit=LIMIT_POSTS)):
    title = sub.title or ""
    body = sub.selftext or ""
    text = f"{title} {body}".strip()
    
    if len(text) >= 20:
        h = text_hash(text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            buffer_items.append({
                "keyword": None,
                "subreddit": subreddit_name,
                "content_type": "post",
                "post_id": sub.id,
                "post_title": title,
                "text": text,
                "score": sub.score,
                "url": sub.url,
                "created_utc": sub.created_utc,
                "id": sub.id,
                "author": str(sub.author) if sub.author else "[deleted]"
            })
            buffer_texts.append(text)
    
    try:
        sub.comments.replace_more(limit=0)
        count = 0
        for c in sub.comments:
            if count >= MAX_COMMENTS_PER_POST:
                break
            if not hasattr(c, "body"):
                continue
            if c.body in ("[deleted]", "[removed]"):
                continue
            
            txt = c.body.strip()
            if len(txt) < 20:
                continue
            
            h = text_hash(txt)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            buffer_items.append({
                "keyword": None,
                "subreddit": subreddit_name,
                "content_type": "comment",
                "post_id": sub.id,
                "post_title": title,
                "text": txt,
                "score": c.score,
                "url": f"https://reddit.com{c.permalink}",
                "created_utc": c.created_utc,
                "id": c.id,
                "author": str(c.author) if c.author else "[deleted]"
            })
            buffer_texts.append(txt)
            count += 1
        
        time.sleep(COMMENT_FETCH_DELAY)
    except Exception:
        pass
    
    if len(buffer_texts) >= PIPELINE_BATCH:
        process_batch(buffer_items, buffer_texts)
        buffer_items.clear()
        buffer_texts.clear()

process_batch(buffer_items, buffer_texts)

# ===================== SAVE =====================
df = pd.DataFrame(results)

if not df.empty:
    df.drop_duplicates(subset="id", inplace=True)
    df.sort_values("similarity_score", ascending=False, inplace=True)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"\n[DONE] Saved {len(df)} rows → {OUTPUT_FILE}")
print(f"[STATS] Posts: {len(df[df.content_type == 'post'])}")
print(f"[STATS] Comments: {len(df[df.content_type == 'comment'])}")
print(f"[PROGRESS] Total processed subs: {len(processed_subs) + 1}")
print(f"[PROGRESS] Remaining: {len(discovered_subs - processed_subs) - 1}")
