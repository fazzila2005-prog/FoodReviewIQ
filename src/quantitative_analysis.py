
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\FoodReviewIQ")

LOG_ODDS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rating_log_odds.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_rating_analysis():

    df = pd.read_csv(
        LOG_ODDS_FILE
    )

    return df


# ============================================================
# TOP ASPECTS ASSOCIATED WITH LOW RATINGS
# ============================================================

def get_low_rating_aspects(
    top_n=10,
    min_reviews=100
):

    df = load_rating_analysis()

    # Remove very rare aspects.
    # This prevents tiny samples such as
    # "nashville hot cauliflower" from
    # dominating the results.

    filtered = df[
        df["total_reviews"] >= min_reviews
    ].copy()

    filtered = filtered.sort_values(
        "log_odds_low_vs_high",
        ascending=False
    )

    return filtered.head(top_n)


# ============================================================
# TOP ASPECTS ASSOCIATED WITH HIGH RATINGS
# ============================================================

def get_high_rating_aspects(
    top_n=10,
    min_reviews=100
):

    df = load_rating_analysis()

    filtered = df[
        df["total_reviews"] >= min_reviews
    ].copy()

    filtered = filtered.sort_values(
        "log_odds_low_vs_high",
        ascending=True
    )

    return filtered.head(top_n)


# ============================================================
# GET ONE SPECIFIC ASPECT
# ============================================================

def get_aspect_analysis(
    aspect_name
):

    df = load_rating_analysis()

    # Exact match first

    exact = df[
        df["analysis_aspect"].str.lower()
        == aspect_name.lower()
    ]

    if not exact.empty:

        return exact.iloc[0]

    # Partial match if exact match
    # was not found

    partial = df[
        df["analysis_aspect"]
        .str.lower()
        .str.contains(
            aspect_name.lower(),
            regex=False
        )
    ]

    if not partial.empty:

        return partial.iloc[0]

    return None


# ============================================================
# FORMAT LOW-RATING RESULTS
# ============================================================

def format_low_rating_results(
    top_n=10,
    min_reviews=100
):

    df = get_low_rating_aspects(
        top_n=top_n,
        min_reviews=min_reviews
    )

    lines = []

    for _, row in df.iterrows():

        lines.append(
            f"- {row['analysis_aspect']}: "
            f"log-odds={row['log_odds_low_vs_high']:.3f}, "
            f"low-rating rate={row['low_rating_rate']:.3f}, "
            f"high-rating rate={row['high_rating_rate']:.3f}, "
            f"low-rating reviews={int(row['low_rating_reviews'])}, "
            f"high-rating reviews={int(row['high_rating_reviews'])}, "
            f"total={int(row['total_reviews'])}"
        )

    return "\n".join(lines)


# ============================================================
# FORMAT HIGH-RATING RESULTS
# ============================================================

def format_high_rating_results(
    top_n=10,
    min_reviews=100
):

    df = get_high_rating_aspects(
        top_n=top_n,
        min_reviews=min_reviews
    )

    lines = []

    for _, row in df.iterrows():

        lines.append(
            f"- {row['analysis_aspect']}: "
            f"log-odds={row['log_odds_low_vs_high']:.3f}, "
            f"low-rating rate={row['low_rating_rate']:.3f}, "
            f"high-rating rate={row['high_rating_rate']:.3f}, "
            f"low-rating reviews={int(row['low_rating_reviews'])}, "
            f"high-rating reviews={int(row['high_rating_reviews'])}, "
            f"total={int(row['total_reviews'])}"
        )

    return "\n".join(lines)


# ============================================================
# FORMAT SPECIFIC ASPECT
# ============================================================

def format_specific_aspect(
    aspect_name
):

    result = get_aspect_analysis(
        aspect_name
    )

    if result is None:

        return (
            f"No quantitative analysis "
            f"was found for '{aspect_name}'."
        )

    return f"""
Aspect: {result['analysis_aspect']}

Low-rating reviews:
{int(result['low_rating_reviews'])}

High-rating reviews:
{int(result['high_rating_reviews'])}

Total reviews:
{int(result['total_reviews'])}

Low-rating rate:
{result['low_rating_rate']:.4f}

High-rating rate:
{result['high_rating_rate']:.4f}

Log-odds (low vs high):
{result['log_odds_low_vs_high']:.4f}
""".strip()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TOP ASPECTS ASSOCIATED WITH LOW RATINGS")
    print("=" * 70)

    print(
        format_low_rating_results(
            top_n=10,
            min_reviews=100
        )
    )

    print()

    print("=" * 70)
    print("TOP ASPECTS ASSOCIATED WITH HIGH RATINGS")
    print("=" * 70)

    print(
        format_high_rating_results(
            top_n=10,
            min_reviews=100
        )
    )

    print()

    print("=" * 70)
    print("WAITING TIME")
    print("=" * 70)

    print(
        format_specific_aspect(
            "Waiting Time"
        )
    )

