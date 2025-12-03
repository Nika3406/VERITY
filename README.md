# BERTPropagandaDetection

## Overview
Propaganda is not limited to news articles or political debates—it thrives in online communities. Reddit, in particular, hosts politically charged discussions filled with memes, sarcasm, and echo chambers that traditional propaganda detection models struggle with.

This project introduces a Reddit-only propaganda dataset and a BERT-based classifier fine-tuned on Reddit posts, designed to detect and visualize propaganda techniques in informal online discourse.

---

## Objectives
- Collect a large dataset of politically charged Reddit posts and replies.  
- Use existing propaganda/fallacy benchmarks only as baselines, not for training.  
- Manually annotate a Reddit dataset with propaganda categories.  
- Fine-tune BERT-base on Reddit data.  
- Deliver a tool that highlights propaganda techniques in real Reddit discussions.  

---

## Benchmarks as Baselines
While the model is trained only on Reddit, these benchmarks guide category definitions and bias framing:

- **Propaganda Techniques Corpus (PTC):** 18 techniques from news.  
- **MAFALDA Dataset:** Logical fallacies in text.  

---

## Why Reddit?
Unlike structured debates or news articles, Reddit:
- Uses informal, crowd-sourced language (memes, sarcasm, slang).  
- Reflects community dynamics (upvotes, echo chambers).  
- Provides a novel dataset for propaganda detection in conversational settings.  

---

## Legal & Ethical Considerations
- Data collected using Reddit’s API (PRAW + Pushshift).  
- No personal data—only anonymized post text.  
- Usernames, IDs, and metadata removed before dataset release.  

---

## Propaganda Categories (Reddit-tailored)
- Ad Hominem  
- Appeal to Emotion  
- Strawman  
- Bandwagon  
- Loaded Language  
- False Dilemma / False Cause  
- Slogans & Memes  
- Obfuscation / Vagueness  

---

## Data Collection Plan
- **Sources:** r/politics, r/conservative, r/worldpolitics, r/PoliticalDiscussion, r/NeutralPolitics.  
- **Initial Scrape:** ~50,000 Reddit posts + replies.  
- **Manual Annotation:** 2,000–5,000 samples.  
- **Expansion:** Semi-supervised learning to scale.  

---

## Sample Code

### 1. Collecting Reddit Data with PRAW
```python
import praw

# Initialize Reddit API (requires creating a Reddit app)
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="propaganda-detector"
)

# Example: scrape posts from r/politics
subreddit = reddit.subreddit("politics")

for post in subreddit.hot(limit=5):
    print("Title:", post.title)
    print("Text:", post.selftext)
    print("---")
