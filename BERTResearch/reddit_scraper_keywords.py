import praw
import pandas as pd
import os
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm

# ==== CONFIG ====
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

KEYWORDS_FILE = "keywords.txt"
OUTPUT_FILE = "K-Dataset.csv"
SEARCH_TARGET = "all"  # Search ALL of Reddit for broader dataset
LIMIT_PER_SEARCH = 150
MIN_SLEEP = 1
MAX_SLEEP = 2
SCRAPE_COMMENTS = True
MAX_COMMENTS_PER_POST = 50

# ==== AUTH ====
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ==== LOAD KEYWORDS ====
with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]

print(f"[INFO] Loaded {len(keywords)} keywords")
print(f"[INFO] Searching across ALL of Reddit")

# ==== COLLECT ALL DATA ====
print("\n[COLLECTING DATA]")
posts = []

for kw in keywords:
    print(f"  [SEARCH] '{kw}'")
    try:
        submissions = list(reddit.subreddit(SEARCH_TARGET).search(kw, limit=LIMIT_PER_SEARCH, sort="new"))

        with tqdm(total=len(submissions), desc=f"Keyword: {kw}") as pbar:
            for sub in submissions:
                subreddit_name = str(sub.subreddit.display_name)
                title = sub.title or ""
                body = sub.selftext or ""

                posts.append({
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
                    "similarity_score": None,
                    "author": str(sub.author) if sub.author else "[deleted]"
                })

                if SCRAPE_COMMENTS:
                    try:
                        sub.comments.replace_more(limit=0)
                        for c in sub.comments.list()[:MAX_COMMENTS_PER_POST]:
                            if hasattr(c, "body") and c.body and c.body not in ["[deleted]", "[removed]"]:
                                posts.append({
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
                                    "similarity_score": None,
                                    "author": str(c.author) if c.author else "[deleted]"
                                })
                    except Exception as e:
                        print(f"[WARN] Comment error: {e}")

                pbar.update(1)

        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

    except Exception as e:
        print(f"[WARN] Error searching '{kw}': {e}")
        time.sleep(10)

df = pd.DataFrame(posts)
df.drop_duplicates(subset="id", inplace=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

unique_subreddits = df['subreddit'].nunique()
post_count = len(df[df['content_type'] == 'post'])
comment_count = len(df[df['content_type'] == 'comment'])

print(f"\n[DONE] Saved {len(df)} items to {OUTPUT_FILE}")
print(f"[STATS] Posts: {post_count}, Comments: {comment_count}")
print(f"[STATS] Found across {unique_subreddits} different subreddits")
