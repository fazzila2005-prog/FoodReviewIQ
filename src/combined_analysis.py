
from quantitative_analysis import (
    get_low_rating_aspects,
    get_high_rating_aspects,
    get_aspect_analysis
)


# ============================================================
# BUILD QUANTITATIVE CONTEXT
# ============================================================

def build_quantitative_context():

    low = get_low_rating_aspects(
        top_n=10,
        min_reviews=100
    )

    high = get_high_rating_aspects(
        top_n=10,
        min_reviews=100
    )

    context = []

    context.append(
        "ASPECTS ASSOCIATED WITH LOW RATINGS:"
    )

    for _, row in low.iterrows():

        context.append(
            f"- {row['analysis_aspect']}: "
            f"log-odds={row['log_odds_low_vs_high']:.3f}, "
            f"low-rating rate="
            f"{row['low_rating_rate']:.3f}, "
            f"high-rating rate="
            f"{row['high_rating_rate']:.3f}, "
            f"total reviews="
            f"{int(row['total_reviews'])}"
        )

    context.append("")

    context.append(
        "ASPECTS ASSOCIATED WITH HIGH RATINGS:"
    )

    for _, row in high.iterrows():

        context.append(
            f"- {row['analysis_aspect']}: "
            f"log-odds={row['log_odds_low_vs_high']:.3f}, "
            f"low-rating rate="
            f"{row['low_rating_rate']:.3f}, "
            f"high-rating rate="
            f"{row['high_rating_rate']:.3f}, "
            f"total reviews="
            f"{int(row['total_reviews'])}"
        )

    return "\n".join(context)


# ============================================================
# FIND SPECIFIC ASPECT
# ============================================================

def get_specific_aspect_context(
    aspect_name
):

    result = get_aspect_analysis(
        aspect_name
    )

    if result is None:

        return (
            f"No quantitative data found "
            f"for {aspect_name}."
        )

    return f"""
Aspect:
{result['analysis_aspect']}

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

Log-odds low vs high:
{result['log_odds_low_vs_high']:.4f}
""".strip()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("QUANTITATIVE CONTEXT")
    print("=" * 70)

    print(
        build_quantitative_context()
    )

    print()

    print("=" * 70)
    print("SPECIFIC ASPECT: WAITING TIME")
    print("=" * 70)

    print(
        get_specific_aspect_context(
            "Waiting Time"
        )
    )

