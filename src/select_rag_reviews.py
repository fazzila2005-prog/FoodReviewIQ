import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\FoodReviewIQ")

REVIEWS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "processed_reviews.csv"
)

ASPECTS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "review_aspects.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_selected_reviews.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_REVIEWS = 5000

# Maximum number of selected reviews from one restaurant
MAX_PER_RESTAURANT = 5

# Minimum number of reviews we want for each sentiment
SENTIMENT_TARGET = 300

# Random seed makes the selection reproducible
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("FOODREVIEWIQ - REPRESENTATIVE RAG REVIEW SELECTION")
print("=" * 60)

print("\nLoading reviews...")

reviews = pd.read_csv(
    REVIEWS_FILE
)

print(
    f"Reviews loaded: {len(reviews):,}"
)


print("\nLoading aspect information...")

aspects = pd.read_csv(
    ASPECTS_FILE
)

print(
    f"Aspect records loaded: {len(aspects):,}"
)


# ============================================================
# CLEAN IDs
# ============================================================

reviews["review_id"] = (
    reviews["review_id"]
    .astype(str)
)

reviews["business_id"] = (
    reviews["business_id"]
    .astype(str)
)

aspects["review_id"] = (
    aspects["review_id"]
    .astype(str)
)


# ============================================================
# CREATE REVIEW-LEVEL ASPECT SUMMARY
# ============================================================

print("\nCreating review-level aspect summaries...")


# Keep one occurrence of each aspect/sentiment combination
aspect_unique = (
    aspects[
        [
            "review_id",
            "canonical_parent",
            "canonical_aspect",
            "aspect_type",
            "overall_sentiment",
        ]
    ]
    .drop_duplicates()
)


def build_aspect_summary(group):

    items = []

    for _, row in group.iterrows():

        aspect = str(
            row["canonical_aspect"]
        )

        sentiment = str(
            row["overall_sentiment"]
        )

        items.append(
            f"{aspect} ({sentiment})"
        )

    # Remove duplicates while preserving order
    items = list(
        dict.fromkeys(items)
    )

    return "; ".join(items)


aspect_summary = (
    aspect_unique
    .groupby("review_id")
    .apply(build_aspect_summary)
    .reset_index(
        name="aspect_summary"
    )
)


# ============================================================
# CREATE REVIEW-LEVEL SENTIMENT
# ============================================================

print("\nCreating review-level sentiment...")


def determine_review_sentiment(group):

    sentiments = (
        group["overall_sentiment"]
        .astype(str)
        .str.lower()
        .value_counts()
    )

    positive = sentiments.get(
        "positive",
        0
    )

    negative = sentiments.get(
        "negative",
        0
    )

    neutral = sentiments.get(
        "neutral",
        0
    )

    # Use the dominant aspect sentiment
    if positive >= negative and positive >= neutral:
        return "positive"

    elif negative >= positive and negative >= neutral:
        return "negative"

    else:
        return "neutral"


review_sentiment = (
    aspect_unique
    .groupby("review_id")
    .apply(determine_review_sentiment)
    .reset_index(
        name="review_sentiment"
    )
)


# ============================================================
# MERGE INFORMATION
# ============================================================

print("\nMerging review information...")

df = reviews.merge(
    aspect_summary,
    on="review_id",
    how="left"
)

df = df.merge(
    review_sentiment,
    on="review_id",
    how="left"
)


df["aspect_summary"] = (
    df["aspect_summary"]
    .fillna(
        "No specific aspect identified"
    )
)


df["review_sentiment"] = (
    df["review_sentiment"]
    .fillna("unknown")
)


# ============================================================
# CLEAN STAR RATINGS
# ============================================================

df["review_stars"] = pd.to_numeric(
    df["review_stars"],
    errors="coerce"
)


# ============================================================
# CREATE RATING GROUP
# ============================================================

def rating_group(stars):

    if stars <= 2:
        return "low"

    elif stars == 3:
        return "middle"

    else:
        return "high"


df["rating_group"] = (
    df["review_stars"]
    .apply(rating_group)
)


# ============================================================
# SHOW AVAILABLE DATA
# ============================================================

print("\nAvailable review groups:")

print(
    df["rating_group"]
    .value_counts()
)

print("\nAvailable sentiment groups:")

print(
    df["review_sentiment"]
    .value_counts()
)


# ============================================================
# SELECTION FUNCTION
# ============================================================

selected_parts = []


def select_reviews(
    source,
    number,
    label
):

    if number <= 0:
        return

    if len(source) == 0:
        print(
            f"WARNING: No reviews available "
            f"for {label}"
        )
        return

    # Shuffle first
    source = source.sample(
        frac=1,
        random_state=RANDOM_SEED
    )

    # Limit number from one restaurant
    selected = (
        source
        .groupby(
            "business_id",
            group_keys=False
        )
        .head(
            MAX_PER_RESTAURANT
        )
    )

    # If we still need more, sample from remaining
    if len(selected) < number:

        remaining = source[
            ~source["review_id"].isin(
                selected["review_id"]
            )
        ]

        extra_needed = (
            number - len(selected)
        )

        if len(remaining) > 0:

            extra = remaining.sample(
                n=min(
                    extra_needed,
                    len(remaining)
                ),
                random_state=RANDOM_SEED
            )

            selected = pd.concat(
                [
                    selected,
                    extra
                ]
            )

    # Final size
    selected = selected.head(
        number
    )

    print(
        f"Selected {len(selected):,} "
        f"reviews for {label}"
    )

    selected_parts.append(
        selected
    )


# ============================================================
# PHASE 1
# BALANCED RATING DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1 - RATING BALANCE")
print("=" * 60)


# Target approximately:
#
# 30% low ratings
# 20% middle ratings
# 50% high ratings


low_target = int(
    TARGET_REVIEWS * 0.30
)

middle_target = int(
    TARGET_REVIEWS * 0.20
)

high_target = int(
    TARGET_REVIEWS * 0.50
)


select_reviews(
    df[
        df["rating_group"] == "low"
    ],
    low_target,
    "low-rated reviews (1-2 stars)"
)


select_reviews(
    df[
        df["rating_group"] == "middle"
    ],
    middle_target,
    "middle-rated reviews (3 stars)"
)


select_reviews(
    df[
        df["rating_group"] == "high"
    ],
    high_target,
    "high-rated reviews (4-5 stars)"
)


# ============================================================
# COMBINE PHASE 1
# ============================================================

selected = pd.concat(
    selected_parts,
    ignore_index=True
)


# Remove duplicate reviews
selected = (
    selected
    .drop_duplicates(
        subset="review_id"
    )
)


print(
    f"\nReviews after rating selection: "
    f"{len(selected):,}"
)


# ============================================================
# PHASE 2
# ADD ASPECT DIVERSITY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 - ASPECT DIVERSITY")
print("=" * 60)


# Get important aspects based on frequency
aspect_counts = (
    aspect_unique[
        "canonical_aspect"
    ]
    .value_counts()
)


# Take the top 50 aspects
important_aspects = (
    aspect_counts
    .head(50)
    .index
    .tolist()
)


print(
    f"Important aspects considered: "
    f"{len(important_aspects)}"
)


# Remaining reviews not already selected
remaining = df[
    ~df["review_id"].isin(
        selected["review_id"]
    )
].copy()


# We want additional aspect diversity
# without making the dataset much larger.

aspect_additions = []


for aspect in important_aspects:

    if len(selected) >= TARGET_REVIEWS:
        break

    matching_ids = set(
        aspect_unique[
            aspect_unique[
                "canonical_aspect"
            ] == aspect
        ]["review_id"]
    )

    candidates = remaining[
        remaining["review_id"].isin(
            matching_ids
        )
    ]

    if len(candidates) == 0:
        continue

    # Pick up to 10 reviews per important aspect
    # from different restaurants when possible.

    candidates = (
        candidates
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .groupby(
            "business_id",
            group_keys=False
        )
        .head(1)
    )

    number_needed = min(
        10,
        TARGET_REVIEWS - len(selected)
    )

    candidates = candidates.head(
        number_needed
    )

    if len(candidates) > 0:

        aspect_additions.append(
            candidates
        )

        # Remove them from remaining
        remaining = remaining[
            ~remaining["review_id"].isin(
                candidates["review_id"]
            )
        ]

        selected = pd.concat(
            [
                selected,
                candidates
            ],
            ignore_index=True
        )

        selected = (
            selected
            .drop_duplicates(
                subset="review_id"
            )
        )


print(
    f"Reviews after aspect diversity: "
    f"{len(selected):,}"
)


# ============================================================
# PHASE 3
# FILL REMAINING SLOTS
# ============================================================

print("\n" + "=" * 60)
print("PHASE 3 - FILL REMAINING SLOTS")
print("=" * 60)


remaining_needed = (
    TARGET_REVIEWS
    - len(selected)
)


if remaining_needed > 0:

    remaining = df[
        ~df["review_id"].isin(
            selected["review_id"]
        )
    ].copy()

    remaining = (
        remaining
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
    )

    # Prefer restaurants not already heavily represented
    current_counts = (
        selected["business_id"]
        .value_counts()
    )

    remaining["current_count"] = (
        remaining["business_id"]
        .map(current_counts)
        .fillna(0)
    )

    remaining = remaining.sort_values(
        "current_count"
    )

    extra = (
        remaining
        .groupby(
            "business_id",
            group_keys=False
        )
        .head(
            MAX_PER_RESTAURANT
        )
    )

    extra = extra.head(
        remaining_needed
    )

    selected = pd.concat(
        [
            selected,
            extra
        ],
        ignore_index=True
    )


# ============================================================
# FINALIZE
# ============================================================

selected = (
    selected
    .drop_duplicates(
        subset="review_id"
    )
)


# If slightly over target, randomly trim
if len(selected) > TARGET_REVIEWS:

    selected = selected.sample(
        n=TARGET_REVIEWS,
        random_state=RANDOM_SEED
    )


# ============================================================
# CREATE EMBEDDING TEXT
# ============================================================

selected["embedding_text"] = (
    "Review aspects: "
    + selected["aspect_summary"].astype(str)
    + "\nReview: "
    + selected["text"].astype(str)
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "review_id",
    "business_id",
    "restaurant_name",
    "city",
    "state",
    "business_stars",
    "review_stars",
    "date",
    "useful",
    "funny",
    "cool",
    "aspect_summary",
    "review_sentiment",
    "rating_group",
    "text",
    "embedding_text",
]


selected = selected[
    output_columns
]


selected.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("REPRESENTATIVE RAG DATASET CREATED")
print("=" * 60)

print(
    f"Final reviews: "
    f"{len(selected):,}"
)

print(
    f"Unique restaurants: "
    f"{selected['business_id'].nunique():,}"
)

print(
    f"Output file:\n{OUTPUT_FILE}"
)


print("\nRating distribution:")

print(
    selected["review_stars"]
    .value_counts()
    .sort_index()
)


print("\nRating group distribution:")

print(
    selected["rating_group"]
    .value_counts()
)


print("\nReview sentiment distribution:")

print(
    selected["review_sentiment"]
    .value_counts()
)


print("\nTop aspects represented:")

aspect_frequency = {}

for value in selected[
    "aspect_summary"
].astype(str):

    for item in value.split(";"):

        aspect = item.split("(")[0].strip()

        if aspect:
            aspect_frequency[aspect] = (
                aspect_frequency.get(
                    aspect,
                    0
                ) + 1
            )


top_aspects = sorted(
    aspect_frequency.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]


for aspect, count in top_aspects:

    print(
        f"{aspect}: {count:,}"
    )


print("\n" + "=" * 60)
print("SELECTION COMPLETE")
print("=" * 60)