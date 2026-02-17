#!/usr/bin/env python3
"""
Google Books API Integration for Romance Book Analysis

Analyzes book metadata from Google Books to identify:
- Publishing velocity (new releases per month)
- Description keyword analysis
- Emerging themes via topic modeling
"""

import argparse
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

try:
    from googleapiclient.discovery import build
except ImportError:
    print("Please install: pip install google-api-python-client")
    exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
except ImportError:
    print("Please install: pip install scikit-learn")
    exit(1)

# Load environment variables
load_dotenv()

# Project paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Search queries for different romance niches
NICHE_QUERIES = {
    # Sports
    "hockey_romance": "hockey romance novel",
    "f1_romance": "formula one racing romance",
    "racing_romance": "racing romance novel",
    "sports_romance": "sports romance contemporary",
    "baseball_romance": "baseball romance novel",
    "football_romance": "football romance novel",
    "soccer_romance": "soccer romance novel",
    "mma_romance": "MMA fighter romance",
    # Settings
    "small_town_romance": "small town romance novel",
    "billionaire_romance": "billionaire romance",
    "mafia_romance": "mafia romance novel",
    "cowboy_romance": "cowboy western romance",
    "dark_romance": "dark romance novel",
    "college_romance": "college romance new adult",
    "military_romance": "military romance novel",
    "rockstar_romance": "rockstar musician romance",
    "royal_romance": "royal prince romance",
    "office_romance": "office workplace romance",
    "bodyguard_romance": "bodyguard protector romance",
    "single_dad_romance": "single dad romance",
    # Tropes
    "enemies_to_lovers": "enemies to lovers romance",
    "fake_dating": "fake dating romance novel",
    "grumpy_sunshine": "grumpy sunshine romance",
    "forced_proximity": "forced proximity romance",
    "second_chance": "second chance romance novel",
    "secret_baby": "secret baby romance",
    "forbidden_romance": "forbidden romance novel",
    "age_gap": "age gap romance novel",
    "marriage_convenience": "marriage of convenience romance",
    "friends_to_lovers": "friends to lovers romance",
    "brothers_best_friend": "brother's best friend romance",
    "slow_burn": "slow burn romance novel",
    "why_choose": "why choose reverse harem romance",
    # Paranormal/Fantasy
    "shifter_romance": "shifter romance novel",
    "vampire_romance": "vampire romance novel",
    "werewolf_romance": "werewolf romance novel",
    "fae_romance": "fae fantasy romance",
    "dragon_romance": "dragon shifter romance",
    "fantasy_romance": "fantasy romance novel",
    "romantasy": "romantasy novel",
    "alien_romance": "alien sci-fi romance",
    "monster_romance": "monster romance novel",
    # Historical
    "regency_romance": "regency romance novel",
    "historical_romance": "historical romance novel",
    "scottish_romance": "scottish highlander romance",
    "viking_romance": "viking romance novel",
    "western_romance": "western frontier romance",
}


def get_books_service():
    """Create Google Books API service (no auth required for basic searches)."""
    # API key is optional but increases quota
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")

    if api_key:
        return build("books", "v1", developerKey=api_key)
    else:
        print("Note: Running without API key (limited to 1000 requests/day)")
        print("Set GOOGLE_BOOKS_API_KEY for higher quota")
        return build("books", "v1")


def search_books(service, query: str, max_results: int = 40) -> list[dict]:
    """
    Search for books matching a query.

    Args:
        service: Google Books API service
        query: Search query
        max_results: Maximum results (max 40 per request)

    Returns:
        List of book metadata dicts
    """
    books = []
    start_index = 0

    while len(books) < max_results:
        try:
            result = service.volumes().list(
                q=query,
                startIndex=start_index,
                maxResults=min(40, max_results - len(books)),
                orderBy="relevance",
                printType="BOOKS",
                langRestrict="en",
            ).execute()

            items = result.get("items", [])
            if not items:
                break

            for item in items:
                info = item.get("volumeInfo", {})
                books.append({
                    "id": item.get("id"),
                    "title": info.get("title", ""),
                    "authors": ", ".join(info.get("authors", [])),
                    "publisher": info.get("publisher", ""),
                    "published_date": info.get("publishedDate", ""),
                    "description": info.get("description", ""),
                    "categories": ", ".join(info.get("categories", [])),
                    "page_count": info.get("pageCount"),
                    "average_rating": info.get("averageRating"),
                    "ratings_count": info.get("ratingsCount", 0),
                    "language": info.get("language", ""),
                })

            start_index += len(items)
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"Error fetching books: {e}")
            break

    return books


def parse_publish_year(date_str: str) -> int | None:
    """Extract year from various date formats."""
    if not date_str:
        return None

    # Try different formats
    patterns = [
        r"^(\d{4})-\d{2}-\d{2}$",  # 2024-01-15
        r"^(\d{4})-\d{2}$",        # 2024-01
        r"^(\d{4})$",              # 2024
    ]

    for pattern in patterns:
        match = re.match(pattern, date_str)
        if match:
            return int(match.group(1))

    return None


def analyze_publishing_velocity(books: list[dict]) -> pd.DataFrame:
    """
    Calculate publishing velocity (books per year) for a niche.
    """
    years = []
    for book in books:
        year = parse_publish_year(book["published_date"])
        if year and 2015 <= year <= 2026:
            years.append(year)

    if not years:
        return pd.DataFrame()

    counts = Counter(years)
    df = pd.DataFrame([
        {"year": year, "count": count}
        for year, count in sorted(counts.items())
    ])

    return df


def extract_keywords_from_descriptions(books: list[dict], top_n: int = 30) -> list[tuple]:
    """
    Extract most common meaningful keywords from book descriptions
    using TF-IDF.
    """
    descriptions = [b["description"] for b in books if b["description"]]

    if len(descriptions) < 5:
        return []

    # TF-IDF to find important terms
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    try:
        tfidf = vectorizer.fit_transform(descriptions)
        feature_names = vectorizer.get_feature_names_out()

        # Average TF-IDF scores across documents
        avg_scores = tfidf.mean(axis=0).A1
        top_indices = avg_scores.argsort()[-top_n:][::-1]

        return [(feature_names[i], round(avg_scores[i], 4)) for i in top_indices]

    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        return []


def run_topic_modeling(books: list[dict], n_topics: int = 5) -> list[list[str]]:
    """
    Run LDA topic modeling on book descriptions to identify themes.
    """
    descriptions = [b["description"] for b in books if b["description"]]

    if len(descriptions) < 10:
        return []

    # Vectorize
    vectorizer = CountVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    try:
        doc_term = vectorizer.fit_transform(descriptions)
        feature_names = vectorizer.get_feature_names_out()

        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
        )
        lda.fit(doc_term)

        # Extract top words per topic
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-8:][::-1]]
            topics.append(top_words)

        return topics

    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return []


def analyze_niche(service, niche_name: str, query: str, max_books: int = 100) -> dict:
    """
    Run full analysis for a single niche.
    """
    print(f"\nAnalyzing: {niche_name}")
    print(f"  Query: {query}")

    books = search_books(service, query, max_books)
    print(f"  Found {len(books)} books")

    if not books:
        return {}

    # Save raw data
    books_df = pd.DataFrame(books)
    books_df.to_csv(DATA_DIR / f"books_{niche_name}.csv", index=False)

    # Publishing velocity
    velocity_df = analyze_publishing_velocity(books)
    if not velocity_df.empty:
        recent_velocity = velocity_df[velocity_df["year"] >= 2023]["count"].mean()
        early_velocity = velocity_df[velocity_df["year"] <= 2020]["count"].mean()
        velocity_growth = ((recent_velocity - early_velocity) / max(early_velocity, 1)) * 100
    else:
        velocity_growth = 0
        recent_velocity = 0

    # Keywords
    keywords = extract_keywords_from_descriptions(books, top_n=15)

    # Topics
    topics = run_topic_modeling(books, n_topics=3)

    # Ratings analysis
    rated_books = [b for b in books if b["average_rating"]]
    avg_rating = sum(b["average_rating"] for b in rated_books) / len(rated_books) if rated_books else 0

    return {
        "niche": niche_name,
        "total_books": len(books),
        "recent_velocity": round(recent_velocity, 1),
        "velocity_growth_pct": round(velocity_growth, 1),
        "avg_rating": round(avg_rating, 2),
        "top_keywords": keywords[:10],
        "topics": topics,
    }


def run_full_analysis(max_books_per_niche: int = 100):
    """Run analysis across all romance niches."""
    service = get_books_service()

    all_results = []

    for niche_name, query in NICHE_QUERIES.items():
        result = analyze_niche(service, niche_name, query, max_books_per_niche)
        if result:
            all_results.append(result)
        time.sleep(1)  # Rate limiting between niches

    if not all_results:
        print("No results collected")
        return

    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            "niche": r["niche"],
            "total_books": r["total_books"],
            "recent_velocity": r["recent_velocity"],
            "velocity_growth_pct": r["velocity_growth_pct"],
            "avg_rating": r["avg_rating"],
        }
        for r in all_results
    ]).sort_values("velocity_growth_pct", ascending=False)

    summary_df.to_csv(DATA_DIR / "books_niche_summary.csv", index=False)

    print("\n" + "="*60)
    print("GOOGLE BOOKS ANALYSIS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

    # Top keywords across all niches
    print("\nTop Keywords by Niche:")
    for r in all_results:
        print(f"\n{r['niche']}:")
        for kw, score in r["top_keywords"][:5]:
            print(f"  - {kw} ({score})")

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in summary_df["velocity_growth_pct"]]
    plt.barh(summary_df["niche"], summary_df["velocity_growth_pct"], color=colors)
    plt.axvline(x=0, color="black", linewidth=0.5)
    plt.xlabel("Publishing Velocity Growth (%)")
    plt.title("Romance Niche Publishing Growth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "books_velocity_growth.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'books_velocity_growth.png'}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze Google Books for romance trends")
    parser.add_argument(
        "--niche",
        choices=list(NICHE_QUERIES.keys()) + ["all"],
        default="all",
        help="Niche to analyze (default: all)",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=100,
        help="Max books per niche (default: 100)",
    )
    parser.add_argument(
        "--query",
        help="Custom search query",
    )

    args = parser.parse_args()

    print(f"Google Books Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    service = get_books_service()

    if args.query:
        analyze_niche(service, "custom", args.query, args.max_books)
    elif args.niche == "all":
        run_full_analysis(args.max_books)
    else:
        analyze_niche(service, args.niche, NICHE_QUERIES[args.niche], args.max_books)


if __name__ == "__main__":
    main()
