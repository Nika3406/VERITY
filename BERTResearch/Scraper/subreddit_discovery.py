import praw
import os
import re
from dotenv import load_dotenv
from tqdm import tqdm

# ===================== CONFIG =====================
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ===================== EXPANDED SEED LIST =====================
EXPANDED_SEEDS = [
    # Core political
    "politics", "worldnews", "news", "geopolitics", "PoliticalDiscussion",
    "NeutralPolitics", "moderatepolitics", "ukpolitics", "CanadaPolitics",
    "AustralianPolitics", "europeanpolitics", "Ask_Politics",
    
    # Partisan
    "conservative", "liberal", "democrats", "Republican", "progressive",
    "Libertarian", "socialism", "capitalism", "communism", "anarchism",
    "GreenParty", "neoliberal", "SocialDemocracy", "Conservative",
    
    # Debate & opinion
    "changemyview", "unpopularopinion", "The10thDentist", "TrueUnpopularOpinion",
    "ControversialOpinions", "Vent", "rant", "offmychest", "TrueOffMyChest",
    
    # Memes & humor (often political)
    "PoliticalCompassMemes", "PoliticalHumor", "TheRightCantMeme",
    "TheLeftCantMeme", "ENLIGHTENEDCENTRISM", "ToiletPaperUSA",
    
    # Social issues
    "MensRights", "Feminism", "TwoXChromosomes", "racism", "BlackLivesMatter",
    "immigration", "climate", "environment", "Economics", "economy",
    
    # Conspiracy & controversial
    "conspiracy", "conspiracytheories", "JoeRogan", "TimPool",
    "IntellectualDarkWeb", "stupidpol",
    
    # International
    "worldpolitics", "anime_titties", "China", "Russia", "iran", "Israel",
    "Palestine", "syriancivilwar", "MiddleEastNews",
    
    # Media & analysis
    "media_criticism", "propaganda", "chomsky", "BreadTube", "TheMajorityReport",
    "DaveRubin", "ChapoTrapHouse", "KotakuInAction", "GamerGhazi",
    
    # Ask communities (often political)
    "AskReddit", "AskAnAmerican", "AskEurope", "AskALiberal",
    "AskConservatives", "AskTrumpSupporters", "AskThe_Donald",
    
    # Discussion
    "TrueReddit", "NeutralTalk", "InsightfulQuestions", "self",
    "CasualConversation", "SeriousConversation", "Discussion",
    
    # Current events
    "OutOfTheLoop", "explainlikeimfive", "NoStupidQuestions",
    "TooAfraidToAsk", "AmItheAsshole", "AmITheButtface",
    
    # Philosophy & ethics
    "philosophy", "ethics", "religion", "atheism", "Christianity",
    "islam", "DebateReligion", "DebateAnAtheist",
    
    # Country specific (high activity)
    "unitedkingdom", "canada", "australia", "germany", "france",
    "india", "brasil", "mexico", "unitedstates", "USPolitics",
    
    # Meta & specific movements
    "antiwork", "WorkReform", "LateStageCapitalism", "Capitalism",
    "Anarcho_Capitalism", "Shitstatistssay", "EnoughLibertarianSpam",
    
    # Tech/society intersection
    "technology", "Futurology", "privacy", "ABoringDystopia",
    "LeopardsAteMyFace", "SelfAwarewolves",
    
    # Education/research
    "science", "AskHistorians", "PoliticalScience", "Ask_Lawyers",
    "legaladviceofftopic", "AskEconomics", "AcademicPhilosophy"
]

DISCOVERY_FILE = "discovered_subreddits.txt"
STATS_FILE = "discovery_stats.txt"

# How many subs to scan (don't scan ALL of them)
MAX_SUBS_TO_SCAN = 50  # Only scan 50 subs instead of all
POSTS_PER_SUB = 100    # Reduced from 300-500

# ===================== LOAD EXISTING =====================
if os.path.exists(DISCOVERY_FILE):
    with open(DISCOVERY_FILE) as f:
        discovered = set(line.strip() for line in f if line.strip())
    print(f"[INFO] Loaded {len(discovered)} existing subreddits")
else:
    discovered = set()

# ===================== ADD SEEDS =====================
discovered.update(EXPANDED_SEEDS)
print(f"[INFO] Total after adding seeds: {len(discovered)}")

# Save immediately so seeds are written
with open(DISCOVERY_FILE, "w") as f:
    for sub in sorted(discovered):
        f.write(sub + "\n")

# ===================== FAST DISCOVERY (POSTS ONLY) =====================
def discover_fast(subreddit_name, limit=100):
    """Fast discovery: only scan post titles/text for r/mentions"""
    found = set()
    try:
        sub = reddit.subreddit(subreddit_name)
        
        # Method 1: Sidebar (very fast, single API call)
        try:
            sidebar = sub.description or ""
            for match in re.findall(r"r/([A-Za-z0-9_]+)", sidebar):
                found.add(match)
        except:
            pass
        
        # Method 2: Posts (fast-ish)
        try:
            for post in sub.hot(limit=limit):
                text = f"{post.title} {post.selftext}"
                for match in re.findall(r"r/([A-Za-z0-9_]+)", text):
                    found.add(match)
                
                # Crossposts (bonus mentions)
                if hasattr(post, "crosspost_parent_list"):
                    for cp in post.crosspost_parent_list:
                        name = cp.get("subreddit")
                        if name:
                            found.add(name)
        except:
            pass
            
    except Exception:
        pass
    
    return found

# ===================== SELECT SUBS TO SCAN =====================
# Prioritize: seeds first, then existing discovered subs
to_scan = list(EXPANDED_SEEDS)

# Add some already-discovered subs if we have room
remaining_slots = MAX_SUBS_TO_SCAN - len(to_scan)
if remaining_slots > 0:
    other_subs = list(discovered - set(EXPANDED_SEEDS))
    to_scan.extend(other_subs[:remaining_slots])

to_scan = to_scan[:MAX_SUBS_TO_SCAN]

print(f"\n[DISCOVERY] Scanning {len(to_scan)} subreddits (fast mode)")
print(f"[INFO] This should take 5-15 minutes\n")

# ===================== RUN DISCOVERY =====================
stats = {
    "scanned": 0,
    "new_found": 0,
    "failed": 0
}

new_subs = set()

for sub_name in tqdm(to_scan):
    try:
        found = discover_fast(sub_name, limit=POSTS_PER_SUB)
        new_found = found - discovered
        
        if new_found:
            new_subs.update(new_found)
            stats["new_found"] += len(new_found)
        
        stats["scanned"] += 1
        
    except Exception:
        stats["failed"] += 1
        continue

# ===================== UPDATE DISCOVERY FILE =====================
discovered.update(new_subs)

with open(DISCOVERY_FILE, "w") as f:
    for sub in sorted(discovered):
        f.write(sub + "\n")

# ===================== SAVE STATS =====================
with open(STATS_FILE, "w") as f:
    f.write(f"Total subreddits discovered: {len(discovered)}\n")
    f.write(f"Scanned: {stats['scanned']}\n")
    f.write(f"New found this run: {stats['new_found']}\n")
    f.write(f"Failed: {stats['failed']}\n")
    f.write(f"\nSample of new discoveries:\n")
    for sub in sorted(new_subs)[:100]:
        f.write(f"  - r/{sub}\n")

# ===================== RESULTS =====================
print(f"\n{'='*60}")
print(f"[COMPLETE] Discovery finished!")
print(f"{'='*60}")
print(f"Total discovered: {len(discovered)}")
print(f"New this run: {len(new_subs)}")
print(f"Scanned: {stats['scanned']}")
print(f"Failed: {stats['failed']}")
print(f"\nSaved to: {DISCOVERY_FILE}")
print(f"Stats saved to: {STATS_FILE}")
print(f"\n[TIP] Run this script multiple times to discover more")
print(f"[TIP] Each run scans different subreddits from your list")
