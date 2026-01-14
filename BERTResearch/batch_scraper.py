import subprocess
import os
import pandas as pd
from datetime import datetime

DATASET_DIR = "datasets"
LOG_FILE = "batch_scraping_log.txt"
TARGET_SAMPLES = 20_000  # Your target
BATCH_SIZE = 20  # How many subs to scrape before checking

def get_current_count():
    """Count total samples across all CSV files"""
    if not os.path.exists(DATASET_DIR):
        return 0
    
    total = 0
    for filename in os.listdir(DATASET_DIR):
        if filename.startswith("S-Dataset_r_") and filename.endswith(".csv"):
            filepath = os.path.join(DATASET_DIR, filename)
            try:
                df = pd.read_csv(filepath)
                total += len(df)
            except:
                continue
    return total

def log_progress(message):
    """Log progress to file and print"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

# ===================== MAIN BATCH LOOP =====================
print("\n" + "="*60)
print("BATCH REDDIT SCRAPER")
print("="*60)

start_count = get_current_count()
log_progress(f"Starting batch run with {start_count:,} existing samples")
log_progress(f"Target: {TARGET_SAMPLES:,} samples")

scraped_this_batch = 0
errors = 0

while True:
    current_count = get_current_count()
    
    # Check if we've hit target
    if current_count >= TARGET_SAMPLES:
        log_progress(f"TARGET REACHED! {current_count:,} samples collected")
        break
    
    # Check batch limit
    if scraped_this_batch >= BATCH_SIZE:
        log_progress(f"Batch limit reached ({BATCH_SIZE} subreddits)")
        log_progress("Run this script again to continue scraping")
        break
    
    remaining = TARGET_SAMPLES - current_count
    log_progress(f"\nScraping next subreddit... (Need {remaining:,} more)")
    log_progress(f"Progress: {current_count:,}/{TARGET_SAMPLES:,} ({100*current_count/TARGET_SAMPLES:.1f}%)")
    
    # Run the scraper
    try:
        result = subprocess.run(
            ["python", "reddit_scraper_sentences.py"],
            capture_output=True,
            text=True,
            timeout=1200  # 20 min timeout per subreddit
        )
        
        # Check if scraper found no more subreddits
        if "No more suitable subreddits" in result.stdout:
            log_progress("No more subreddits available")
            log_progress("Run discovery script again to find more")
            break
        
        # Check for success
        if result.returncode == 0:
            # Parse output to get subreddit name and count
            for line in result.stdout.split("\n"):
                if "[SELECTED]" in line:
                    log_progress(line.strip())
                if "[DONE] Saved" in line:
                    log_progress(line.strip())
            
            scraped_this_batch += 1
        else:
            errors += 1
            log_progress(f"Scraper failed (error #{errors})")
            if errors >= 5:
                log_progress("Too many errors, stopping batch")
                break
    
    except subprocess.TimeoutExpired:
        log_progress("Scraper timeout (10 min)")
        errors += 1
    except Exception as e:
        log_progress(f"Error: {str(e)}")
        errors += 1

# ===================== FINAL STATS =====================
final_count = get_current_count()
gained = final_count - start_count

print("\n" + "="*60)
print("BATCH COMPLETE")
print("="*60)
log_progress(f"Subreddits scraped this batch: {scraped_this_batch}")
log_progress(f"Samples gained: {gained:,}")
log_progress(f"Total samples now: {final_count:,}")
log_progress(f"Progress: {100*final_count/TARGET_SAMPLES:.1f}%")

if final_count >= TARGET_SAMPLES:
    log_progress("TARGET REACHED!")
else:
    remaining = TARGET_SAMPLES - final_count
    log_progress(f"Still need {remaining:,} more samples")
    log_progress("Run this script again to continue")
