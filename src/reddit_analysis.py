#!/usr/bin/env python3
"""
Reddit Analysis for Romance Book Trends

Analyzes r/RomanceBooks subreddit to identify:
- Trending tropes and settings
- Unmet demand (request posts)
- Community sentiment on different niches
"""

import argparse
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

try:
    import praw
except ImportError:
    print("Please install praw: pip install praw")
    exit(1)

# Load environment variables
load_dotenv()

# Project paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Keywords to track
TROPES = [
    "enemies to lovers", "friends to lovers", "fake dating", "forced proximity",
    "grumpy sunshine", "second chance", "forbidden love", "slow burn",
    "one bed", "marriage of convenience", "secret baby", "age gap",
    "brother's best friend", "boss employee", "celebrity romance",
]

SETTINGS = [
    "small town", "hockey", "f1", "racing", "sports", "mafia", "dark romance",
    "billionaire", "cowboy", "western", "historical", "regency", "scottish",
    "paranormal", "shifter", "vampire", "alien", "fantasy", "sci-fi",
    "office", "college", "military", "rockstar", "royal", "arranged marriage",
]


def get_reddit_client() -> praw.Reddit:
    """Create authenticated Reddit client from environment variables."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "RomanceTrends/1.0")

    if not client_id or not client_secret:
        print("Reddit API credentials not found.")
        print("Please create a Reddit app at https://www.reddit.com/prefs/apps")
        print("Then set these environment variables:")
        print("  REDDIT_CLIENT_ID=your_client_id")
        print("  REDDIT_CLIENT_SECRET=your_client_secret")
        print("\nOr create a .env file in the project root with these values.")
        exit(1)

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def fetch_posts(reddit: praw.Reddit, subreddit: str, limit: int = 1000,
                time_filter: str = "year") -> list[dict]:
    """
    Fetch posts from a subreddit.

    Args:
        reddit: Authenticated Reddit client
        subreddit: Subreddit name
        limit: Maximum posts to fetch
        time_filter: One of: hour, day, week, month, year, all
    """
    sub = reddit.subreddit(subreddit)
    posts = []

    print(f"Fetching up to {limit} posts from r/{subreddit}...")

    for i, post in enumerate(sub.top(time_filter=time_filter, limit=limit)):
        posts.append({
            "id": post.id,
            "title": post.title,
            "selftext": post.selftext or "",
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": datetime.fromtimestamp(post.created_utc),
            "url": post.url,
            "flair": post.link_flair_text,
            "upvote_ratio": post.upvote_ratio,
        })

        if (i + 1) % 100 == 0:
            print(f"  Fetched {i + 1} posts...")
            time.sleep(1)  # Rate limiting

    print(f"Fetched {len(posts)} posts")
    return posts


def count_keyword_mentions(posts: list[dict], keywords: list[str]) -> Counter:
    """Count mentions of keywords across all posts."""
    counts = Counter()

    for post in posts:
        text = f"{post['title']} {post['selftext']}".lower()
        for keyword in keywords:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                counts[keyword] += 1

    return counts


def analyze_request_posts(posts: list[dict]) -> list[dict]:
    """
    Identify "looking for" / request posts - indicators of unmet demand.

    These posts often contain phrases like:
    - "looking for"
    - "recommendation"
    - "suggest"
    - "any books with"
    """
    request_patterns = [
        r'\blooking for\b',
        r'\brecommend',
        r'\bsuggest',
        r'\bany books? (with|like|similar)',
        r'\bwhat.*(read|book)',
        r'\bhelp.*find',
    ]

    requests = []
    for post in posts:
        text = f"{post['title']} {post['selftext']}".lower()
        for pattern in request_patterns:
            if re.search(pattern, text):
                requests.append(post)
                break

    return requests


def calculate_engagement_by_keyword(posts: list[dict], keywords: list[str]) -> pd.DataFrame:
    """
    Calculate average engagement metrics for posts mentioning each keyword.

    This helps identify which niches generate the most community interest.
    """
    keyword_stats = defaultdict(lambda: {"posts": 0, "total_score": 0, "total_comments": 0})

    for post in posts:
        text = f"{post['title']} {post['selftext']}".lower()
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                keyword_stats[keyword]["posts"] += 1
                keyword_stats[keyword]["total_score"] += post["score"]
                keyword_stats[keyword]["total_comments"] += post["num_comments"]

    results = []
    for keyword, stats in keyword_stats.items():
        if stats["posts"] > 0:
            results.append({
                "keyword": keyword,
                "post_count": stats["posts"],
                "avg_score": round(stats["total_score"] / stats["posts"], 1),
                "avg_comments": round(stats["total_comments"] / stats["posts"], 1),
                "engagement_score": round(
                    (stats["total_score"] + stats["total_comments"] * 2) / stats["posts"], 1
                ),
            })

    return pd.DataFrame(results).sort_values("engagement_score", ascending=False)


def analyze_temporal_trends(posts: list[dict], keywords: list[str]) -> pd.DataFrame:
    """
    Track keyword mentions over time to identify growing vs declining interest.
    """
    # Group posts by month
    monthly_counts = defaultdict(lambda: Counter())

    for post in posts:
        month_key = post["created_utc"].strftime("%Y-%m")
        text = f"{post['title']} {post['selftext']}".lower()

        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                monthly_counts[month_key][keyword] += 1

    # Convert to DataFrame
    df = pd.DataFrame(monthly_counts).T.fillna(0).sort_index()
    df.index = pd.to_datetime(df.index)

    return df


def plot_keyword_mentions(counts: Counter, title: str, output_path: Path):
    """Bar chart of keyword mention counts."""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    keywords = list(counts.keys())
    values = list(counts.values())

    # Sort by count
    sorted_pairs = sorted(zip(values, keywords), reverse=True)
    values, keywords = zip(*sorted_pairs) if sorted_pairs else ([], [])

    colors = sns.color_palette("viridis", len(keywords))
    plt.barh(range(len(keywords)), values, color=colors)
    plt.yticks(range(len(keywords)), keywords)
    plt.xlabel("Number of Mentions")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_engagement_comparison(df: pd.DataFrame, output_path: Path):
    """Scatter plot of engagement vs frequency."""
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    plt.scatter(df["post_count"], df["engagement_score"], s=100, alpha=0.7)

    for _, row in df.iterrows():
        plt.annotate(row["keyword"], (row["post_count"], row["engagement_score"]),
                    fontsize=8, alpha=0.8)

    plt.xlabel("Number of Posts")
    plt.ylabel("Engagement Score")
    plt.title("Reddit Engagement: Frequency vs Engagement", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def run_analysis(subreddit: str = "RomanceBooks", limit: int = 1000,
                time_filter: str = "year"):
    """Run full Reddit analysis."""
    reddit = get_reddit_client()

    print(f"\n{'='*60}")
    print(f"Reddit Analysis: r/{subreddit}")
    print(f"Time filter: {time_filter}, Limit: {limit}")
    print(f"{'='*60}")

    # Fetch posts
    posts = fetch_posts(reddit, subreddit, limit, time_filter)

    if not posts:
        print("No posts fetched")
        return

    # Save raw posts
    posts_df = pd.DataFrame(posts)
    posts_df.to_csv(DATA_DIR / f"reddit_{subreddit}_posts.csv", index=False)

    # Analyze trope mentions
    print("\nAnalyzing trope mentions...")
    trope_counts = count_keyword_mentions(posts, TROPES)
    print(f"\nTop Tropes by Mention Count:")
    for trope, count in trope_counts.most_common(10):
        print(f"  {trope}: {count}")

    plot_keyword_mentions(trope_counts, f"r/{subreddit} - Trope Mentions",
                         OUTPUT_DIR / f"reddit_tropes.png")

    # Analyze setting mentions
    print("\nAnalyzing setting mentions...")
    setting_counts = count_keyword_mentions(posts, SETTINGS)
    print(f"\nTop Settings by Mention Count:")
    for setting, count in setting_counts.most_common(10):
        print(f"  {setting}: {count}")

    plot_keyword_mentions(setting_counts, f"r/{subreddit} - Setting Mentions",
                         OUTPUT_DIR / f"reddit_settings.png")

    # Analyze engagement
    print("\nCalculating engagement scores...")
    all_keywords = TROPES + SETTINGS
    engagement_df = calculate_engagement_by_keyword(posts, all_keywords)
    engagement_df.to_csv(DATA_DIR / "reddit_engagement.csv", index=False)

    print("\nTop Keywords by Engagement Score:")
    print(engagement_df.head(15).to_string(index=False))

    plot_engagement_comparison(engagement_df.head(20),
                              OUTPUT_DIR / "reddit_engagement.png")

    # Analyze request posts (unmet demand)
    print("\nAnalyzing request posts (unmet demand indicators)...")
    requests = analyze_request_posts(posts)
    print(f"Found {len(requests)} request posts out of {len(posts)} total")

    if requests:
        request_tropes = count_keyword_mentions(requests, TROPES)
        request_settings = count_keyword_mentions(requests, SETTINGS)

        print("\nMost Requested Tropes (unmet demand):")
        for trope, count in request_tropes.most_common(5):
            print(f"  {trope}: {count}")

        print("\nMost Requested Settings (unmet demand):")
        for setting, count in request_settings.most_common(5):
            print(f"  {setting}: {count}")

    # Temporal analysis
    print("\nAnalyzing temporal trends...")
    temporal_df = analyze_temporal_trends(posts, all_keywords)
    temporal_df.to_csv(DATA_DIR / "reddit_temporal.csv")

    # Calculate growth for each keyword
    if len(temporal_df) >= 6:
        growth_data = []
        for col in temporal_df.columns:
            recent = temporal_df[col].tail(3).mean()
            prior = temporal_df[col].head(3).mean()
            growth = ((recent - prior) / max(prior, 1)) * 100 if prior > 0 else 0
            growth_data.append({"keyword": col, "growth_pct": round(growth, 1)})

        growth_df = pd.DataFrame(growth_data).sort_values("growth_pct", ascending=False)
        print("\nFastest Growing Keywords on Reddit:")
        print(growth_df.head(10).to_string(index=False))
        growth_df.to_csv(DATA_DIR / "reddit_growth.csv", index=False)

    print(f"\nAnalysis complete! Data saved to {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Reddit for romance book trends")
    parser.add_argument(
        "--subreddit",
        default="RomanceBooks",
        help="Subreddit to analyze (default: RomanceBooks)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum posts to fetch (default: 1000)",
    )
    parser.add_argument(
        "--time",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="year",
        help="Time filter for posts (default: year)",
    )

    args = parser.parse_args()
    run_analysis(args.subreddit, args.limit, args.time)


if __name__ == "__main__":
    main()
