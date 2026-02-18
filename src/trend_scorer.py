#!/usr/bin/env python3
"""
Trend Scorer - Composite Index for Romance Niche Analysis

Combines data from multiple sources to calculate a composite trend score:
- Google Trends search volume growth (40%)
- Reddit community engagement growth (30%)
- Google Books publishing velocity (20%)
- Market saturation index (10%)

Higher scores = more promising emerging niches
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Project paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# Keyword mapping between different sources
# Maps canonical niche names to their variants across data sources
NICHE_MAPPING = {
    # Sports
    "hockey_romance": {"google_trends": "hockey romance", "reddit": "hockey", "google_books": "hockey_romance"},
    "f1_romance": {"google_trends": "F1 romance", "reddit": "f1", "google_books": "f1_romance"},
    "racing_romance": {"google_trends": "racing romance", "reddit": "racing", "google_books": "racing_romance"},
    "sports_romance": {"google_trends": "sports romance", "reddit": "sports", "google_books": "sports_romance"},
    "baseball_romance": {"google_trends": "baseball romance", "reddit": "baseball", "google_books": "baseball_romance"},
    "football_romance": {"google_trends": "football romance", "reddit": "football", "google_books": "football_romance"},
    "soccer_romance": {"google_trends": "soccer romance", "reddit": "soccer", "google_books": "soccer_romance"},
    "mma_romance": {"google_trends": "MMA romance", "reddit": "mma", "google_books": "mma_romance"},
    "boxing_romance": {"google_trends": "boxing romance", "reddit": "boxing", "google_books": None},
    "basketball_romance": {"google_trends": "basketball romance", "reddit": "basketball", "google_books": None},
    # Settings
    "small_town_romance": {"google_trends": "small town romance", "reddit": "small town", "google_books": "small_town_romance"},
    "billionaire_romance": {"google_trends": "billionaire romance", "reddit": "billionaire", "google_books": "billionaire_romance"},
    "mafia_romance": {"google_trends": "mafia romance", "reddit": "mafia", "google_books": "mafia_romance"},
    "cowboy_romance": {"google_trends": "cowboy romance", "reddit": "cowboy", "google_books": "cowboy_romance"},
    "dark_romance": {"google_trends": "dark romance", "reddit": "dark romance", "google_books": "dark_romance"},
    "college_romance": {"google_trends": "college romance", "reddit": "college", "google_books": "college_romance"},
    "military_romance": {"google_trends": "military romance", "reddit": "military", "google_books": "military_romance"},
    "rockstar_romance": {"google_trends": "rockstar romance", "reddit": "rockstar", "google_books": "rockstar_romance"},
    "royal_romance": {"google_trends": "royal romance", "reddit": "royal", "google_books": "royal_romance"},
    "office_romance": {"google_trends": "office romance", "reddit": "office", "google_books": "office_romance"},
    "bodyguard_romance": {"google_trends": "bodyguard romance", "reddit": "bodyguard", "google_books": "bodyguard_romance"},
    "single_dad_romance": {"google_trends": "single dad romance", "reddit": "single dad", "google_books": "single_dad_romance"},
    "bully_romance": {"google_trends": "bully romance", "reddit": "bully", "google_books": None},
    "boss_romance": {"google_trends": "boss romance", "reddit": "boss", "google_books": None},
    # Tropes
    "enemies_to_lovers": {"google_trends": "enemies to lovers", "reddit": "enemies to lovers", "google_books": "enemies_to_lovers"},
    "fake_dating": {"google_trends": "fake dating romance", "reddit": "fake dating", "google_books": "fake_dating"},
    "grumpy_sunshine": {"google_trends": "grumpy sunshine", "reddit": "grumpy sunshine", "google_books": "grumpy_sunshine"},
    "forced_proximity": {"google_trends": "forced proximity", "reddit": "forced proximity", "google_books": "forced_proximity"},
    "second_chance": {"google_trends": "second chance romance", "reddit": "second chance", "google_books": "second_chance"},
    "secret_baby": {"google_trends": "secret baby romance", "reddit": "secret baby", "google_books": "secret_baby"},
    "forbidden_romance": {"google_trends": "forbidden romance", "reddit": "forbidden", "google_books": "forbidden_romance"},
    "age_gap": {"google_trends": "age gap romance", "reddit": "age gap", "google_books": "age_gap"},
    "marriage_convenience": {"google_trends": "marriage of convenience", "reddit": "marriage of convenience", "google_books": "marriage_convenience"},
    "friends_to_lovers": {"google_trends": "friends to lovers", "reddit": "friends to lovers", "google_books": "friends_to_lovers"},
    "brothers_best_friend": {"google_trends": "brother's best friend", "reddit": "brother's best friend", "google_books": "brothers_best_friend"},
    "slow_burn": {"google_trends": "slow burn romance", "reddit": "slow burn", "google_books": "slow_burn"},
    "one_bed": {"google_trends": "one bed trope", "reddit": "one bed", "google_books": None},
    "love_triangle": {"google_trends": "love triangle", "reddit": "love triangle", "google_books": None},
    "revenge_romance": {"google_trends": "revenge romance", "reddit": "revenge", "google_books": None},
    "why_choose": {"google_trends": "why choose romance", "reddit": "why choose", "google_books": "why_choose"},
    # Paranormal/Fantasy
    "shifter_romance": {"google_trends": "shifter romance", "reddit": "shifter", "google_books": "shifter_romance"},
    "vampire_romance": {"google_trends": "vampire romance", "reddit": "vampire", "google_books": "vampire_romance"},
    "werewolf_romance": {"google_trends": "werewolf romance", "reddit": "werewolf", "google_books": "werewolf_romance"},
    "fae_romance": {"google_trends": "fae romance", "reddit": "fae", "google_books": "fae_romance"},
    "dragon_romance": {"google_trends": "dragon romance", "reddit": "dragon", "google_books": "dragon_romance"},
    "fantasy_romance": {"google_trends": "fantasy romance", "reddit": "fantasy", "google_books": "fantasy_romance"},
    "romantasy": {"google_trends": "romantasy", "reddit": "romantasy", "google_books": "romantasy"},
    "alien_romance": {"google_trends": "alien romance", "reddit": "alien", "google_books": "alien_romance"},
    "monster_romance": {"google_trends": "monster romance", "reddit": "monster", "google_books": "monster_romance"},
    "time_travel": {"google_trends": "time travel romance", "reddit": "time travel", "google_books": None},
    # Historical
    "regency_romance": {"google_trends": "regency romance", "reddit": "regency", "google_books": "regency_romance"},
    "historical_romance": {"google_trends": "historical romance", "reddit": "historical", "google_books": "historical_romance"},
    "scottish_romance": {"google_trends": "scottish romance", "reddit": "scottish", "google_books": "scottish_romance"},
    "viking_romance": {"google_trends": "viking romance", "reddit": "viking", "google_books": "viking_romance"},
    "western_romance": {"google_trends": "western romance", "reddit": "western", "google_books": "western_romance"},
    # Emerging
    "reverse_harem": {"google_trends": "reverse harem", "reddit": "reverse harem", "google_books": None},
    "spicy_romance": {"google_trends": "spicy romance", "reddit": "spicy", "google_books": None},
    "booktok_romance": {"google_trends": "booktok romance", "reddit": "booktok", "google_books": None},
    # Additional niches
    "queer_romance": {"google_trends": "queer romance", "reddit": "queer", "google_books": "queer_romance"},
    "cozy_romance": {"google_trends": "cozy romance", "reddit": "cozy", "google_books": "cozy_romance"},
    "ghost_romance": {"google_trends": "ghost romance", "reddit": "ghost", "google_books": "ghost_romance"},
    "academic_romance": {"google_trends": "academic romance", "reddit": "academic", "google_books": "academic_romance"},
    "sapphic_romance": {"google_trends": "sapphic romance", "reddit": "sapphic", "google_books": "sapphic_romance"},
    "mm_romance": {"google_trends": "mm romance", "reddit": "mm", "google_books": "mm_romance"},
    "ff_romance": {"google_trends": "ff romance", "reddit": "ff", "google_books": None},
    "lgbtq_romance": {"google_trends": "lgbtq romance", "reddit": "lgbtq", "google_books": None},
    "haunted_romance": {"google_trends": "haunted romance", "reddit": "haunted", "google_books": "haunted_romance"},
    "professor_romance": {"google_trends": "professor romance", "reddit": "professor", "google_books": None},
}


def load_google_trends_data() -> pd.DataFrame | None:
    """Load Google Trends growth data."""
    path = DATA_DIR / "all_niches_growth.csv"
    if not path.exists():
        print(f"Google Trends data not found: {path}")
        return None

    df = pd.read_csv(path)
    return df


def load_reddit_data() -> pd.DataFrame | None:
    """Load Reddit engagement and growth data."""
    growth_path = DATA_DIR / "reddit_growth.csv"
    engagement_path = DATA_DIR / "reddit_engagement.csv"

    dfs = []
    if growth_path.exists():
        growth_df = pd.read_csv(growth_path)
        growth_df = growth_df.rename(columns={"growth_pct": "reddit_growth_pct"})
        dfs.append(growth_df)

    if engagement_path.exists():
        eng_df = pd.read_csv(engagement_path)
        if "engagement_score" in eng_df.columns:
            eng_df = eng_df[["keyword", "engagement_score"]]
            eng_df = eng_df.rename(columns={"engagement_score": "reddit_engagement"})
            dfs.append(eng_df)

    if not dfs:
        print("Reddit data not found")
        return None

    if len(dfs) == 2:
        return dfs[0].merge(dfs[1], on="keyword", how="outer")
    return dfs[0]


def load_books_data() -> pd.DataFrame | None:
    """Load Google Books publishing velocity data."""
    path = DATA_DIR / "books_niche_summary.csv"
    if not path.exists():
        print(f"Google Books data not found: {path}")
        return None

    df = pd.read_csv(path)
    return df


def normalize_scores(series: pd.Series) -> pd.Series:
    """
    Normalize a series to 0-100 scale using min-max normalization.
    Handles negative values by shifting.
    """
    if series.isna().all():
        return series

    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        return pd.Series([50] * len(series), index=series.index)

    return ((series - min_val) / (max_val - min_val)) * 100


def calculate_composite_score(row: pd.Series) -> float:
    """
    Calculate composite trend score for a niche.

    Weights:
    - Google Trends growth: 40%
    - Reddit growth/engagement: 30%
    - Publishing velocity growth: 20%
    - Inverse saturation bonus: 10%
    """
    score = 0
    weights_used = 0

    # Google Trends (40%)
    if pd.notna(row.get("trends_score")):
        score += row["trends_score"] * 0.4
        weights_used += 0.4

    # Reddit (30%)
    if pd.notna(row.get("reddit_score")):
        score += row["reddit_score"] * 0.3
        weights_used += 0.3

    # Books (20%)
    if pd.notna(row.get("books_score")):
        score += row["books_score"] * 0.2
        weights_used += 0.2

    # Saturation bonus (10%) - lower saturation = higher bonus
    if pd.notna(row.get("saturation_score")):
        score += (100 - row["saturation_score"]) * 0.1
        weights_used += 0.1

    # Normalize by weights actually used
    if weights_used > 0:
        return score / weights_used * (weights_used / 1.0)

    return 0


def build_combined_dataset() -> pd.DataFrame:
    """
    Combine all data sources into a single dataset for scoring.
    """
    # Load all data sources
    trends_df = load_google_trends_data()
    reddit_df = load_reddit_data()
    books_df = load_books_data()

    # Start with niche mapping as base
    niches = list(NICHE_MAPPING.keys())
    combined = pd.DataFrame({"niche": niches})

    # Add Google Trends data
    if trends_df is not None:
        trends_map = {}
        for niche, mapping in NICHE_MAPPING.items():
            keyword = mapping.get("google_trends")
            if keyword:
                match = trends_df[trends_df["keyword"].str.lower() == keyword.lower()]
                if not match.empty:
                    trends_map[niche] = match["yoy_growth_pct"].values[0]
        combined["trends_growth"] = combined["niche"].map(trends_map)

    # Add Reddit data
    if reddit_df is not None:
        reddit_map = {}
        for niche, mapping in NICHE_MAPPING.items():
            keyword = mapping.get("reddit")
            if keyword:
                match = reddit_df[reddit_df["keyword"].str.lower() == keyword.lower()]
                if not match.empty:
                    if "reddit_growth_pct" in match.columns:
                        reddit_map[niche] = match["reddit_growth_pct"].values[0]
                    elif "reddit_engagement" in match.columns:
                        reddit_map[niche] = match["reddit_engagement"].values[0]
        combined["reddit_growth"] = combined["niche"].map(reddit_map)

    # Add Books data
    if books_df is not None:
        books_map = {}
        saturation_map = {}
        for niche, mapping in NICHE_MAPPING.items():
            book_niche = mapping.get("google_books")
            if book_niche:
                match = books_df[books_df["niche"] == book_niche]
                if not match.empty:
                    books_map[niche] = match["velocity_growth_pct"].values[0]
                    # Use total_books as rough saturation proxy
                    saturation_map[niche] = match["total_books"].values[0]

        combined["books_growth"] = combined["niche"].map(books_map)
        combined["saturation"] = combined["niche"].map(saturation_map)

    return combined


def calculate_trend_scores(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate normalized scores and composite trend score.
    """
    # Normalize each metric to 0-100 scale
    if "trends_growth" in combined.columns:
        combined["trends_score"] = normalize_scores(combined["trends_growth"])

    if "reddit_growth" in combined.columns:
        combined["reddit_score"] = normalize_scores(combined["reddit_growth"])

    if "books_growth" in combined.columns:
        combined["books_score"] = normalize_scores(combined["books_growth"])

    if "saturation" in combined.columns:
        combined["saturation_score"] = normalize_scores(combined["saturation"])

    # Calculate composite score
    combined["trend_score"] = combined.apply(calculate_composite_score, axis=1)

    # Sort by trend score
    combined = combined.sort_values("trend_score", ascending=False)

    return combined


def classify_opportunity(row: pd.Series) -> str:
    """
    Classify a niche into opportunity categories.
    """
    score = row.get("trend_score", 0)
    saturation = row.get("saturation_score", 50)

    if score >= 70 and saturation < 40:
        return "High Opportunity"  # Growing + undersaturated
    elif score >= 70:
        return "Competitive"       # Growing but saturated
    elif score >= 40 and saturation < 40:
        return "Emerging"          # Moderate growth, low saturation
    elif score < 40 and saturation >= 60:
        return "Saturated"         # Low growth, high saturation
    else:
        return "Watch"             # Moderate/uncertain


def generate_report(df: pd.DataFrame) -> str:
    """Generate a text summary report."""
    report = []
    report.append("=" * 70)
    report.append("ROMANCE TREND ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)

    # Top opportunities
    report.append("\nTOP 5 EMERGING NICHES (Highest Trend Scores):")
    report.append("-" * 50)
    for i, row in df.head(5).iterrows():
        report.append(f"\n{row['niche'].replace('_', ' ').title()}")
        report.append(f"  Trend Score: {row['trend_score']:.1f}")
        report.append(f"  Opportunity: {row.get('opportunity', 'N/A')}")
        if pd.notna(row.get("trends_growth")):
            report.append(f"  Google Trends Growth: {row['trends_growth']:.1f}%")
        if pd.notna(row.get("reddit_growth")):
            report.append(f"  Reddit Growth: {row['reddit_growth']:.1f}%")
        if pd.notna(row.get("books_growth")):
            report.append(f"  Publishing Velocity Growth: {row['books_growth']:.1f}%")

    # High opportunity niches
    high_opp = df[df["opportunity"] == "High Opportunity"]
    if not high_opp.empty:
        report.append("\n\nHIGH OPPORTUNITY NICHES (Growing + Undersaturated):")
        report.append("-" * 50)
        for _, row in high_opp.iterrows():
            report.append(f"  - {row['niche'].replace('_', ' ').title()} (Score: {row['trend_score']:.1f})")

    # Emerging niches
    emerging = df[df["opportunity"] == "Emerging"]
    if not emerging.empty:
        report.append("\n\nEMERGING NICHES (Moderate Growth, Low Saturation):")
        report.append("-" * 50)
        for _, row in emerging.iterrows():
            report.append(f"  - {row['niche'].replace('_', ' ').title()} (Score: {row['trend_score']:.1f})")

    # Data availability note
    report.append("\n\nDATA SOURCES:")
    report.append("-" * 50)
    has_trends = "trends_growth" in df.columns and df["trends_growth"].notna().any()
    has_reddit = "reddit_growth" in df.columns and df["reddit_growth"].notna().any()
    has_books = "books_growth" in df.columns and df["books_growth"].notna().any()
    report.append(f"  Google Trends: {'Available' if has_trends else 'Run: python src/google_trends.py'}")
    report.append(f"  Reddit: {'Available' if has_reddit else 'Run: python src/reddit_analysis.py'}")
    report.append(f"  Google Books: {'Available' if has_books else 'Run: python src/google_books.py'}")

    return "\n".join(report)


def plot_trend_scores(df: pd.DataFrame, output_path: Path):
    """Create a horizontal bar chart of trend scores with opportunity color coding."""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Color by opportunity type
    colors = {
        "High Opportunity": "#27ae60",
        "Competitive": "#f39c12",
        "Emerging": "#3498db",
        "Saturated": "#e74c3c",
        "Watch": "#95a5a6",
    }

    bar_colors = [colors.get(row["opportunity"], "#95a5a6") for _, row in df.iterrows()]

    plt.barh(range(len(df)), df["trend_score"], color=bar_colors)
    plt.yticks(range(len(df)), [n.replace("_", " ").title() for n in df["niche"]])
    plt.xlabel("Composite Trend Score")
    plt.title("Romance Niche Trend Scores", fontsize=14, fontweight="bold")

    # Legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors.values()]
    plt.legend(legend_handles, colors.keys(), loc="lower right")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_matrix(df: pd.DataFrame, output_path: Path):
    """Create a scatter plot showing growth vs saturation."""
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    # Use available metrics
    x_col = "trends_growth" if "trends_growth" in df.columns else "reddit_growth"
    y_col = "saturation" if "saturation" in df.columns else "books_growth"

    if x_col not in df.columns or y_col not in df.columns:
        print("Insufficient data for scatter plot")
        return

    valid_df = df.dropna(subset=[x_col, y_col])
    if valid_df.empty:
        print("No valid data for scatter plot")
        return

    # Color by opportunity
    colors = {
        "High Opportunity": "#27ae60",
        "Competitive": "#f39c12",
        "Emerging": "#3498db",
        "Saturated": "#e74c3c",
        "Watch": "#95a5a6",
    }

    for _, row in valid_df.iterrows():
        plt.scatter(
            row[x_col],
            row[y_col],
            c=colors.get(row["opportunity"], "#95a5a6"),
            s=row["trend_score"] * 3,
            alpha=0.7,
        )
        plt.annotate(
            row["niche"].replace("_", " "),
            (row[x_col], row[y_col]),
            fontsize=8,
            alpha=0.8,
        )

    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title("Niche Opportunity Matrix", fontsize=14, fontweight="bold")

    # Add quadrant lines
    plt.axhline(y=valid_df[y_col].median(), color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=valid_df[x_col].median(), color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def run_scoring():
    """Run the full trend scoring analysis."""
    print("\n" + "="*60)
    print("ROMANCE TREND SCORER")
    print("="*60)

    # Build combined dataset
    print("\nLoading data from all sources...")
    combined = build_combined_dataset()

    # Calculate scores
    print("Calculating trend scores...")
    scored = calculate_trend_scores(combined)

    # Classify opportunities
    scored["opportunity"] = scored.apply(classify_opportunity, axis=1)

    # Save results
    scored.to_csv(DATA_DIR / "trend_scores.csv", index=False)
    print(f"\nSaved scores to: {DATA_DIR / 'trend_scores.csv'}")

    # Print results
    print("\n" + "="*60)
    print("TREND SCORES BY NICHE")
    print("="*60)
    display_cols = ["niche", "trend_score", "opportunity"]
    if "trends_growth" in scored.columns:
        display_cols.insert(2, "trends_growth")
    print(scored[display_cols].to_string(index=False))

    # Generate report
    report = generate_report(scored)
    print("\n" + report)

    report_path = OUTPUT_DIR / "trend_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved report to: {report_path}")

    # Create visualizations
    plot_trend_scores(scored, OUTPUT_DIR / "trend_scores.png")
    plot_scatter_matrix(scored, OUTPUT_DIR / "opportunity_matrix.png")

    return scored


def main():
    parser = argparse.ArgumentParser(description="Calculate composite trend scores")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recalculate all scores from source data",
    )

    args = parser.parse_args()
    run_scoring()


if __name__ == "__main__":
    main()
