#!/usr/bin/env python3
"""
Google Trends Analysis for Romance Book Niches

Analyzes search volume trends for different romance subgenres to identify
emerging niches and calculate growth rates.
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytrends.request import TrendReq

# Project paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Keyword groups for comparison (max 5 per group for Google Trends API)
KEYWORD_GROUPS = {
    # Sports niches
    "sports_romance_1": [
        "hockey romance",
        "F1 romance",
        "racing romance",
        "baseball romance",
        "football romance",
    ],
    "sports_romance_2": [
        "soccer romance",
        "basketball romance",
        "MMA romance",
        "boxing romance",
        "sports romance",
    ],
    # Popular tropes
    "tropes_1": [
        "enemies to lovers",
        "fake dating romance",
        "grumpy sunshine",
        "forced proximity",
        "second chance romance",
    ],
    "tropes_2": [
        "secret baby romance",
        "forbidden romance",
        "age gap romance",
        "one bed trope",
        "marriage of convenience",
    ],
    "tropes_3": [
        "brother's best friend",
        "slow burn romance",
        "friends to lovers",
        "love triangle",
        "revenge romance",
    ],
    # Settings and subgenres
    "settings_1": [
        "small town romance",
        "billionaire romance",
        "mafia romance",
        "cowboy romance",
        "office romance",
    ],
    "settings_2": [
        "dark romance",
        "bully romance",
        "college romance",
        "military romance",
        "rockstar romance",
    ],
    "settings_3": [
        "royal romance",
        "arranged marriage romance",
        "bodyguard romance",
        "boss romance",
        "single dad romance",
    ],
    # Paranormal and fantasy
    "paranormal": [
        "shifter romance",
        "vampire romance",
        "werewolf romance",
        "fae romance",
        "dragon romance",
    ],
    "fantasy_scifi": [
        "fantasy romance",
        "romantasy",
        "alien romance",
        "sci-fi romance",
        "time travel romance",
    ],
    # Historical
    "historical": [
        "regency romance",
        "historical romance",
        "scottish romance",
        "viking romance",
        "western romance",
    ],
    # Emerging/niche
    "emerging": [
        "why choose romance",
        "reverse harem",
        "monster romance",
        "spicy romance",
        "booktok romance",
    ],
}


def fetch_trends(keywords: list[str], timeframe: str = "today 5-y") -> pd.DataFrame:
    """
    Fetch Google Trends data for a list of keywords.

    Args:
        keywords: List of search terms (max 5)
        timeframe: Time range for data

    Returns:
        DataFrame with interest over time
    """
    pytrends = TrendReq(hl="en-US", tz=420)  # Arizona timezone (UTC-7)

    # Build payload and fetch
    pytrends.build_payload(keywords[:5], cat=0, timeframe=timeframe, geo="US")
    data = pytrends.interest_over_time()

    if "isPartial" in data.columns:
        data = data.drop(columns=["isPartial"])

    return data


def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various growth rate metrics for each keyword.

    Returns DataFrame with:
    - Overall growth (first vs last period)
    - YoY growth (last 12 months vs prior 12 months)
    - Recent momentum (last 3 months vs prior 3 months)
    """
    results = []

    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 12:
            continue

        # Overall growth (first 3 months avg vs last 3 months avg)
        first_period = series.head(3).mean()
        last_period = series.tail(3).mean()
        overall_growth = ((last_period - first_period) / max(first_period, 1)) * 100

        # YoY growth
        last_12 = series.tail(12).mean()
        prior_12 = series.iloc[-24:-12].mean() if len(series) >= 24 else series.head(12).mean()
        yoy_growth = ((last_12 - prior_12) / max(prior_12, 1)) * 100

        # Recent momentum (last 3 vs prior 3 months)
        last_3 = series.tail(3).mean()
        prior_3 = series.iloc[-6:-3].mean() if len(series) >= 6 else series.head(3).mean()
        momentum = ((last_3 - prior_3) / max(prior_3, 1)) * 100

        # Current interest level
        current_interest = series.tail(1).values[0]
        avg_interest = series.mean()

        results.append({
            "keyword": col,
            "current_interest": current_interest,
            "avg_interest": round(avg_interest, 1),
            "overall_growth_pct": round(overall_growth, 1),
            "yoy_growth_pct": round(yoy_growth, 1),
            "momentum_3m_pct": round(momentum, 1),
        })

    return pd.DataFrame(results).sort_values("yoy_growth_pct", ascending=False)


def identify_breakouts(growth_df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Identify keywords with >threshold% growth (breakout niches)."""
    return growth_df[growth_df["yoy_growth_pct"] > threshold]


def plot_trends(df: pd.DataFrame, title: str, output_path: Path):
    """Create a line chart of trends over time."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=2)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Search Interest (0-100)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_growth_comparison(growth_df: pd.DataFrame, output_path: Path):
    """Create a bar chart comparing growth rates."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in growth_df["yoy_growth_pct"]]

    plt.barh(growth_df["keyword"], growth_df["yoy_growth_pct"], color=colors)
    plt.axvline(x=0, color="black", linewidth=0.5)
    plt.xlabel("Year-over-Year Growth (%)")
    plt.title("Romance Niche Growth Rates", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def analyze_group(group_name: str, keywords: list[str], timeframe: str = "today 5-y"):
    """Run full analysis for a keyword group."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {group_name}")
    print(f"{'='*60}")

    # Fetch data
    print(f"Fetching trends for: {keywords}")
    df = fetch_trends(keywords, timeframe)

    if df.empty:
        print("No data returned from Google Trends")
        return None, None

    # Save raw data
    csv_path = DATA_DIR / f"{group_name}_trends.csv"
    df.to_csv(csv_path)
    print(f"Saved data: {csv_path}")

    # Calculate growth rates
    growth_df = calculate_growth_rates(df)
    growth_csv = DATA_DIR / f"{group_name}_growth.csv"
    growth_df.to_csv(growth_csv, index=False)
    print(f"Saved growth rates: {growth_csv}")

    # Print summary
    print(f"\nGrowth Rate Summary:")
    print(growth_df.to_string(index=False))

    # Identify breakouts
    breakouts = identify_breakouts(growth_df)
    if not breakouts.empty:
        print(f"\nBreakout Niches (>50% YoY growth):")
        print(breakouts[["keyword", "yoy_growth_pct"]].to_string(index=False))

    # Create visualizations
    plot_trends(df, f"{group_name.replace('_', ' ').title()} - Search Trends",
                OUTPUT_DIR / f"{group_name}_trends.png")
    plot_growth_comparison(growth_df, OUTPUT_DIR / f"{group_name}_growth.png")

    return df, growth_df


def run_full_analysis(timeframe: str = "today 5-y"):
    """Run analysis across all keyword groups and create summary."""
    all_growth = []

    for group_name, keywords in KEYWORD_GROUPS.items():
        _, growth_df = analyze_group(group_name, keywords, timeframe)
        if growth_df is not None:
            growth_df["category"] = group_name
            all_growth.append(growth_df)

    # Combine all results
    if all_growth:
        combined = pd.concat(all_growth, ignore_index=True)
        combined = combined.sort_values("yoy_growth_pct", ascending=False)

        # Save combined results
        combined.to_csv(DATA_DIR / "all_niches_growth.csv", index=False)

        print("\n" + "="*60)
        print("TOP 10 FASTEST GROWING NICHES")
        print("="*60)
        print(combined.head(10).to_string(index=False))

        # Create summary visualization
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        top_15 = combined.head(15)
        colors = sns.color_palette("viridis", len(top_15))

        plt.barh(range(len(top_15)), top_15["yoy_growth_pct"], color=colors)
        plt.yticks(range(len(top_15)), top_15["keyword"])
        plt.xlabel("Year-over-Year Growth (%)")
        plt.title("Top 15 Growing Romance Niches", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_niches_summary.png", dpi=150)
        plt.close()

        return combined

    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze Google Trends for romance niches")
    parser.add_argument(
        "--group",
        choices=list(KEYWORD_GROUPS.keys()) + ["all"],
        default="all",
        help="Keyword group to analyze",
    )
    parser.add_argument(
        "--timeframe",
        default="today 5-y",
        help="Time range (e.g., 'today 5-y', 'today 12-m', '2020-01-01 2025-01-01')",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Custom keywords to analyze (max 5)",
    )

    args = parser.parse_args()

    print(f"Romance Trends Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Timeframe: {args.timeframe}")

    if args.keywords:
        analyze_group("custom", args.keywords[:5], args.timeframe)
    elif args.group == "all":
        run_full_analysis(args.timeframe)
    else:
        analyze_group(args.group, KEYWORD_GROUPS[args.group], args.timeframe)


if __name__ == "__main__":
    main()
