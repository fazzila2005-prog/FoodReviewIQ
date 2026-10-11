import math

import pandas as pd


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\review_aspects.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\rating_log_odds.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LOW_RATINGS = {1.0, 2.0}
HIGH_RATINGS = {4.0, 5.0}

# Laplace/additive smoothing.
ALPHA = 1.0

# Ignore extremely rare aspects.
MIN_REVIEWS = 20


# ============================================================
# LOAD
# ============================================================

print("Loading review-aspect data...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# CLEAN RATINGS
# ============================================================

df["stars"] = pd.to_numeric(
    df["stars"],
    errors="coerce"
)

df = df.dropna(
    subset=["stars"]
)


# ============================================================
# KEEP HIGH / LOW RATINGS ONLY
# ============================================================

df = df[
    df["stars"].isin(
        LOW_RATINGS | HIGH_RATINGS
    )
].copy()

print(
    f"Rows after rating filter: "
    f"{len(df):,}"
)


# ============================================================
# REMOVE CONTEXT-ONLY RECORDS
# ============================================================

df = df[
    df["aspect_type"] != "context_tag"
].copy()


# ============================================================
# CREATE ANALYSIS ASPECT
#
# Core aspects:
#     canonical_aspect
#
# Specific food/beverage items:
#     actual aspect_value
#
# This prevents thousands of specific foods from all
# appearing merely as "Specific Food Item".
# ============================================================

def get_analysis_aspect(row):

    aspect_type = str(
        row["aspect_type"]
    ).strip()

    aspect_value = str(
        row["aspect_value"]
    ).strip()

    canonical_aspect = str(
        row["canonical_aspect"]
    ).strip()

    if (
        aspect_type == "specific_food_item"
        and aspect_value
    ):
        return (
            "Food Item: "
            + aspect_value
        )

    if (
        aspect_type == "specific_beverage"
        and aspect_value
    ):
        return (
            "Beverage: "
            + aspect_value
        )

    return canonical_aspect


df["analysis_aspect"] = (
    df.apply(
        get_analysis_aspect,
        axis=1
    )
)


# ============================================================
# DEDUPLICATE
#
# Each review should count only once for an aspect.
# ============================================================

df = df.drop_duplicates(
    subset=[
        "review_id",
        "analysis_aspect"
    ]
)


# ============================================================
# TOTAL HIGH / LOW REVIEWS
# ============================================================

low_reviews = set(
    df.loc[
        df["stars"].isin(
            LOW_RATINGS
        ),
        "review_id"
    ]
)

high_reviews = set(
    df.loc[
        df["stars"].isin(
            HIGH_RATINGS
        ),
        "review_id"
    ]
)


total_low = len(
    low_reviews
)

total_high = len(
    high_reviews
)


print(
    f"\nLow-rated reviews (1–2 stars): "
    f"{total_low:,}"
)

print(
    f"High-rated reviews (4–5 stars): "
    f"{total_high:,}"
)


# ============================================================
# ASPECT COUNTS
# ============================================================

rows = []

all_aspects = sorted(
    df["analysis_aspect"]
    .dropna()
    .unique()
)


for aspect in all_aspects:

    aspect_df = df[
        df["analysis_aspect"]
        == aspect
    ]

    low_count = (
        aspect_df[
            aspect_df["stars"].isin(
                LOW_RATINGS
            )
        ]["review_id"]
        .nunique()
    )

    high_count = (
        aspect_df[
            aspect_df["stars"].isin(
                HIGH_RATINGS
            )
        ]["review_id"]
        .nunique()
    )

    total_count = (
        low_count
        + high_count
    )

    if total_count < MIN_REVIEWS:
        continue


    # --------------------------------------------------------
    # Smoothed odds
    #
    # odds_low =
    #   P(aspect | low) / P(no aspect | low)
    #
    # odds_high =
    #   P(aspect | high) / P(no aspect | high)
    # --------------------------------------------------------

    low_odds = (
        (low_count + ALPHA)
        /
        (
            total_low
            - low_count
            + ALPHA
        )
    )

    high_odds = (
        (high_count + ALPHA)
        /
        (
            total_high
            - high_count
            + ALPHA
        )
    )


    # --------------------------------------------------------
    # Log-odds ratio
    #
    # > 0 → associated with low ratings
    # < 0 → associated with high ratings
    # --------------------------------------------------------

    log_odds = math.log(
        low_odds
        /
        high_odds
    )


    # --------------------------------------------------------
    # Review rates
    # --------------------------------------------------------

    low_rate = (
        low_count
        /
        total_low
    )

    high_rate = (
        high_count
        /
        total_high
    )


    rows.append({

        "analysis_aspect":
            aspect,

        "low_rating_reviews":
            low_count,

        "high_rating_reviews":
            high_count,

        "total_reviews":
            total_count,

        "low_rating_rate":
            low_rate,

        "high_rating_rate":
            high_rate,

        "log_odds_low_vs_high":
            log_odds
    })


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

result = pd.DataFrame(
    rows
)


# ============================================================
# SORT
# ============================================================

result = result.sort_values(
    "log_odds_low_vs_high",
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n========================================")
print("RATING LOG-ODDS ANALYSIS COMPLETE")
print("========================================")

print(
    f"Aspect groups analyzed: "
    f"{len(result):,}"
)


# ============================================================
# LOW-RATING ASSOCIATIONS
# ============================================================

print("\n========================================")
print("MOST ASSOCIATED WITH LOW RATINGS")
print("========================================")

low_result = (
    result
    .sort_values(
        "log_odds_low_vs_high",
        ascending=False
    )
    .head(20)
)

print(
    low_result[
        [
            "analysis_aspect",
            "low_rating_reviews",
            "high_rating_reviews",
            "low_rating_rate",
            "high_rating_rate",
            "log_odds_low_vs_high"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# HIGH-RATING ASSOCIATIONS
# ============================================================

print("\n========================================")
print("MOST ASSOCIATED WITH HIGH RATINGS")
print("========================================")

high_result = (
    result
    .sort_values(
        "log_odds_low_vs_high",
        ascending=True
    )
    .head(20)
)

print(
    high_result[
        [
            "analysis_aspect",
            "low_rating_reviews",
            "high_rating_reviews",
            "low_rating_rate",
            "high_rating_rate",
            "log_odds_low_vs_high"
        ]
    ].to_string(
        index=False
    )
)


print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)