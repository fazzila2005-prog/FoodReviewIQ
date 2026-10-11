import os
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


# ============================================================
# FoodReviewIQ - Statistical Significance Analysis
# ============================================================

print("=" * 80)
print("FoodReviewIQ - Step 16")
print("Statistical Significance Analysis")
print("=" * 80)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\FoodReviewIQ"

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "rating_log_odds.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "statistical_significance_results.csv"
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

MIN_REVIEWS = 100
ALPHA = 0.05


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading rating analysis data...")

df = pd.read_csv(INPUT_PATH)

print(f"Rows loaded: {len(df):,}")


# ------------------------------------------------------------
# TOTAL LOW / HIGH RATING REVIEWS
# ------------------------------------------------------------

# The original analysis defines:
#
# LOW  = 1-2 stars
# HIGH = 4-5 stars
# 3-star reviews are excluded.


TOTAL_LOW = int(df["low_rating_reviews"].sum())

TOTAL_HIGH = int(df["high_rating_reviews"].sum())

print(f"\nTotal low-rating aspect counts: {TOTAL_LOW:,}")
print(f"Total high-rating aspect counts: {TOTAL_HIGH:,}")


# IMPORTANT:
# The totals above are sums across aspects and therefore are NOT
# the actual number of unique low/high reviews.
#
# The actual totals from the original rating analysis are:
# Low-rated reviews  = 9,236
# High-rated reviews = 33,614

TOTAL_LOW_REVIEWS = 9236
TOTAL_HIGH_REVIEWS = 33614

print("\nUsing actual review-group totals:")
print(f"Low-rating reviews : {TOTAL_LOW_REVIEWS:,}")
print(f"High-rating reviews: {TOTAL_HIGH_REVIEWS:,}")


# ------------------------------------------------------------
# BENJAMINI-HOCHBERG FUNCTION
# ------------------------------------------------------------

def calculate_bh_adjustment(p_values):
    """
    Benjamini-Hochberg correction for multiple comparisons.
    Controls the False Discovery Rate (FDR).
    """

    p_values = np.array(p_values, dtype=float)

    rejected, corrected_p, _, _ = multipletests(
        p_values,
        alpha=ALPHA,
        method="fdr_bh"
    )

    return rejected, corrected_p


# ------------------------------------------------------------
# ANALYZE EACH ASPECT
# ------------------------------------------------------------

results = []

print("\n")
print("=" * 80)
print("RUNNING STATISTICAL TESTS")
print("=" * 80)


filtered_df = df[df["total_reviews"] >= MIN_REVIEWS].copy()

print(
    f"\nAspects with at least {MIN_REVIEWS} reviews: "
    f"{len(filtered_df)}"
)


for _, row in filtered_df.iterrows():

    aspect = str(row["analysis_aspect"])

    low_mentions = int(row["low_rating_reviews"])
    high_mentions = int(row["high_rating_reviews"])

    # Reviews in each rating group that DO NOT mention the aspect
    low_not_mentioned = TOTAL_LOW_REVIEWS - low_mentions
    high_not_mentioned = TOTAL_HIGH_REVIEWS - high_mentions

    # --------------------------------------------------------
    # 2 x 2 CONTINGENCY TABLE
    # --------------------------------------------------------

    table = np.array([
        [low_mentions, low_not_mentioned],
        [high_mentions, high_not_mentioned]
    ])

    # --------------------------------------------------------
    # CHI-SQUARE TEST
    # --------------------------------------------------------

    chi2, chi_p, dof, expected = chi2_contingency(
        table,
        correction=False
    )

    # --------------------------------------------------------
    # FISHER'S EXACT TEST
    # --------------------------------------------------------

    # Fisher is especially useful when expected counts are small.
    if np.min(expected) < 5:
        odds_ratio_fisher, fisher_p = fisher_exact(table)
        test_used = "Fisher's exact"
        p_value = fisher_p
    else:
        odds_ratio_fisher, fisher_p = fisher_exact(table)
        test_used = "Chi-square"
        p_value = chi_p

    # --------------------------------------------------------
    # RAW ODDS RATIO
    # --------------------------------------------------------

    # Add a tiny correction if a cell is zero.
    a = low_mentions
    b = low_not_mentioned
    c = high_mentions
    d = high_not_mentioned

    if a == 0 or b == 0 or c == 0 or d == 0:

        a_adj = a + 0.5
        b_adj = b + 0.5
        c_adj = c + 0.5
        d_adj = d + 0.5

    else:

        a_adj = a
        b_adj = b
        c_adj = c
        d_adj = d

    odds_ratio = (a_adj * d_adj) / (b_adj * c_adj)

    # --------------------------------------------------------
    # 95% CONFIDENCE INTERVAL FOR ODDS RATIO
    # --------------------------------------------------------

    log_or = np.log(odds_ratio)

    standard_error = np.sqrt(
        (1 / a_adj)
        + (1 / b_adj)
        + (1 / c_adj)
        + (1 / d_adj)
    )

    ci_low = np.exp(
        log_or - 1.96 * standard_error
    )

    ci_high = np.exp(
        log_or + 1.96 * standard_error
    )

    # --------------------------------------------------------
    # RATES
    # --------------------------------------------------------

    low_rate = low_mentions / TOTAL_LOW_REVIEWS

    high_rate = high_mentions / TOTAL_HIGH_REVIEWS

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    if odds_ratio > 1:
        association = "More associated with low ratings"

    elif odds_ratio < 1:
        association = "More associated with high ratings"

    else:
        association = "No difference"


    results.append({
        "analysis_aspect": aspect,

        "low_rating_reviews": low_mentions,
        "high_rating_reviews": high_mentions,
        "total_reviews": int(row["total_reviews"]),

        "low_rating_rate": low_rate,
        "high_rating_rate": high_rate,

        "odds_ratio": odds_ratio,

        "ci_95_low": ci_low,
        "ci_95_high": ci_high,

        "chi_square": chi2,
        "chi_square_p_value": chi_p,

        "fisher_p_value": fisher_p,

        "test_used": test_used,
        "p_value": p_value,

        "association": association
    })


# ------------------------------------------------------------
# CREATE RESULTS DATAFRAME
# ------------------------------------------------------------

results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# MULTIPLE TESTING CORRECTION
# ------------------------------------------------------------

print("\nApplying Benjamini-Hochberg correction...")

rejected, corrected_p = calculate_bh_adjustment(
    results_df["p_value"].values
)

results_df["adjusted_p_value"] = corrected_p

results_df["significant_after_bh"] = rejected


# ------------------------------------------------------------
# SORT BY STATISTICAL SIGNIFICANCE
# ------------------------------------------------------------

results_df = results_df.sort_values(
    "adjusted_p_value",
    ascending=True
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# DISPLAY SIGNIFICANT ASPECTS
# ------------------------------------------------------------

significant = results_df[
    results_df["significant_after_bh"] == True
].copy()


print("\n")
print("=" * 80)
print("STATISTICALLY SIGNIFICANT ASPECTS")
print("=" * 80)

print(
    f"\nSignificant aspects after Benjamini-Hochberg correction: "
    f"{len(significant)}"
)


if len(significant) > 0:

    for _, row in significant.iterrows():

        print(
            f"\n{row['analysis_aspect']}"
        )

        print(
            f"  Odds Ratio      : "
            f"{row['odds_ratio']:.3f}"
        )

        print(
            f"  95% CI          : "
            f"{row['ci_95_low']:.3f} - "
            f"{row['ci_95_high']:.3f}"
        )

        print(
            f"  Raw p-value     : "
            f"{row['p_value']:.6g}"
        )

        print(
            f"  Adjusted p-value: "
            f"{row['adjusted_p_value']:.6g}"
        )

        print(
            f"  Low-rating rate : "
            f"{row['low_rating_rate']:.2%}"
        )

        print(
            f"  High-rating rate: "
            f"{row['high_rating_rate']:.2%}"
        )

        print(
            f"  Association     : "
            f"{row['association']}"
        )


# ------------------------------------------------------------
# WAITING TIME SPECIFIC ANALYSIS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("WAITING TIME - STATISTICAL VALIDATION")
print("=" * 80)


waiting_rows = results_df[
    results_df["analysis_aspect"].astype(str).str.lower()
    == "waiting time"
]


if not waiting_rows.empty:

    waiting = waiting_rows.iloc[0]

    print(
        f"\nAspect: Waiting Time"
    )

    print(
        f"Low-rating reviews mentioning it : "
        f"{int(waiting['low_rating_reviews']):,}"
    )

    print(
        f"High-rating reviews mentioning it: "
        f"{int(waiting['high_rating_reviews']):,}"
    )

    print(
        f"Low-rating rate                  : "
        f"{waiting['low_rating_rate']:.2%}"
    )

    print(
        f"High-rating rate                 : "
        f"{waiting['high_rating_rate']:.2%}"
    )

    print(
        f"Odds ratio                       : "
        f"{waiting['odds_ratio']:.3f}"
    )

    print(
        f"95% confidence interval          : "
        f"{waiting['ci_95_low']:.3f} - "
        f"{waiting['ci_95_high']:.3f}"
    )

    print(
        f"Statistical test                 : "
        f"{waiting['test_used']}"
    )

    print(
        f"Raw p-value                      : "
        f"{waiting['p_value']:.6g}"
    )

    print(
        f"Adjusted p-value                 : "
        f"{waiting['adjusted_p_value']:.6g}"
    )

    print(
        f"Significant after correction     : "
        f"{'YES' if waiting['significant_after_bh'] else 'NO'}"
    )

    print(
        f"Association                      : "
        f"{waiting['association']}"
    )


# ------------------------------------------------------------
# TOP LOW-RATING ASSOCIATIONS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("TOP SIGNIFICANT LOW-RATING ASSOCIATIONS")
print("=" * 80)


low_significant = results_df[
    (results_df["significant_after_bh"] == True)
    & (results_df["odds_ratio"] > 1)
].sort_values(
    "odds_ratio",
    ascending=False
)


for _, row in low_significant.head(15).iterrows():

    print(
        f"{row['analysis_aspect']:<35} "
        f"OR={row['odds_ratio']:.3f} | "
        f"p_adj={row['adjusted_p_value']:.4g}"
    )


# ------------------------------------------------------------
# TOP HIGH-RATING ASSOCIATIONS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("TOP SIGNIFICANT HIGH-RATING ASSOCIATIONS")
print("=" * 80)


high_significant = results_df[
    (results_df["significant_after_bh"] == True)
    & (results_df["odds_ratio"] < 1)
].sort_values(
    "odds_ratio",
    ascending=True
)


for _, row in high_significant.head(15).iterrows():

    print(
        f"{row['analysis_aspect']:<35} "
        f"OR={row['odds_ratio']:.3f} | "
        f"p_adj={row['adjusted_p_value']:.4g}"
    )


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("STATISTICAL ANALYSIS SUMMARY")
print("=" * 80)

print(
    f"\nTotal aspects tested: "
    f"{len(results_df)}"
)

print(
    f"Statistically significant after BH correction: "
    f"{len(significant)}"
)

print(
    f"Significant low-rating associations: "
    f"{len(low_significant)}"
)

print(
    f"Significant high-rating associations: "
    f"{len(high_significant)}"
)

print("\nResults saved to:")

print(OUTPUT_PATH)

print("\n")
print("=" * 80)
print("STEP 16 COMPLETE")
print("=" * 80)