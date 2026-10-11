import re

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# FILES
# ============================================================

OCCURRENCES_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_candidates.csv"
)

FINAL_MAPPING_FILE = (
    r"C:\FoodReviewIQ\data\processed\final_cluster_aspect_mapping.csv"
)

REVIEWS_FILE = (
    r"C:\FoodReviewIQ\data\processed\processed_reviews.csv"
)

OUTPUT_OCCURRENCES = (
    r"C:\FoodReviewIQ\data\processed\aspect_sentiment_occurrences.csv"
)

OUTPUT_REVIEW_ASPECTS = (
    r"C:\FoodReviewIQ\data\processed\review_aspects.csv"
)


# ============================================================
# SENTIMENT THRESHOLDS
# ============================================================

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


# ============================================================
# 1. LOAD REVIEW-ASPECT OCCURRENCES
# ============================================================

print("Loading aspect occurrences...")

occurrences = pd.read_csv(
    OCCURRENCES_FILE,
    usecols=[
        "review_id",
        "surface_phrase",
        "aspect_candidate"
    ]
)

occurrences["review_id"] = (
    occurrences["review_id"]
    .astype(str)
)

occurrences["surface_phrase"] = (
    occurrences["surface_phrase"]
    .fillna("")
    .astype(str)
    .str.strip()
)

occurrences["aspect_candidate"] = (
    occurrences["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(
    f"Aspect occurrences: "
    f"{len(occurrences):,}"
)


# ============================================================
# 2. LOAD FINAL DIRECT ASPECT MAPPING
# ============================================================

print(
    "\nLoading final candidate → aspect mapping..."
)

mapping = pd.read_csv(
    FINAL_MAPPING_FILE,
    usecols=[
        "aspect_candidate",
        "canonical_parent",
        "canonical_aspect",
        "aspect_type",
        "aspect_value"
    ]
)

mapping["aspect_candidate"] = (
    mapping["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

mapping["canonical_parent"] = (
    mapping["canonical_parent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

mapping["canonical_aspect"] = (
    mapping["canonical_aspect"]
    .fillna("")
    .astype(str)
    .str.strip()
)

mapping["aspect_type"] = (
    mapping["aspect_type"]
    .fillna("")
    .astype(str)
    .str.strip()
)

mapping["aspect_value"] = (
    mapping["aspect_value"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# IMPORTANT:
# One candidate must have only one final mapping.
mapping = mapping.drop_duplicates(
    subset=["aspect_candidate"],
    keep="first"
)

print(
    f"Unique candidate mappings: "
    f"{len(mapping):,}"
)


# ============================================================
# 3. VALIDATE MAPPING
# ============================================================

unique_aspects = (
    mapping["canonical_aspect"]
    .nunique()
)

print(
    f"Unique canonical aspects available: "
    f"{unique_aspects}"
)

if unique_aspects < 10:

    raise ValueError(
        "Final mapping still looks incorrect. "
        "Expected many canonical aspects."
    )


# ============================================================
# 4. DIRECT MERGE
# ============================================================

print(
    "\nConnecting review phrases directly "
    "to canonical aspects..."
)

aspect_records = occurrences.merge(
    mapping,
    on="aspect_candidate",
    how="left",
    validate="many_to_one"
)

print(
    f"Records after mapping: "
    f"{len(aspect_records):,}"
)


# ============================================================
# 5. KEEP ONLY SENTIMENT-BEARING ASPECTS
# ============================================================

allowed_types = {
    "core_aspect",
    "specific_food_item",
    "specific_beverage"
}

before_filter = len(
    aspect_records
)

aspect_records = aspect_records[
    aspect_records["aspect_type"].isin(
        allowed_types
    )
].copy()

aspect_records = aspect_records.dropna(
    subset=[
        "canonical_parent",
        "canonical_aspect"
    ]
)

print(
    f"\nSentiment-bearing occurrences: "
    f"{len(aspect_records):,}"
)

print(
    f"Removed context/non-sentiment records: "
    f"{before_filter - len(aspect_records):,}"
)


# ============================================================
# 6. LOAD REVIEW FILE HEADER
# ============================================================

print("\nLoading review information...")

review_columns = pd.read_csv(
    REVIEWS_FILE,
    nrows=0
).columns.tolist()

print(
    "Review file columns:"
)

print(
    review_columns
)


# ============================================================
# 7. FIND COLUMNS AUTOMATICALLY
# ============================================================

def find_column(
    candidates,
    available,
    description
):

    for candidate in candidates:

        if candidate in available:
            return candidate

    raise ValueError(
        f"\nCould not find {description}.\n"
        f"Available columns:\n{available}"
    )


review_id_column = find_column(
    [
        "review_id",
        "reviewid",
        "id"
    ],
    review_columns,
    "review ID column"
)


text_column = find_column(
    [
        "text",
        "review_text",
        "review",
        "content"
    ],
    review_columns,
    "review text column"
)


rating_column = find_column(
    [
        "stars",
        "rating",
        "review_rating",
        "review_stars",
        "score"
    ],
    review_columns,
    "rating column"
)


business_id_column = None

for candidate in [
    "business_id",
    "businessid"
]:

    if candidate in review_columns:
        business_id_column = candidate
        break


print(
    f"\nUsing review ID: {review_id_column}"
)

print(
    f"Using text: {text_column}"
)

print(
    f"Using rating: {rating_column}"
)


# ============================================================
# 8. LOAD REVIEWS
# ============================================================

review_usecols = [
    review_id_column,
    text_column,
    rating_column
]

if business_id_column:

    review_usecols.append(
        business_id_column
    )


reviews = pd.read_csv(
    REVIEWS_FILE,
    usecols=review_usecols
)


# ============================================================
# 9. STANDARDIZE REVIEW COLUMNS
# ============================================================

rename_map = {
    review_id_column: "review_id",
    text_column: "text",
    rating_column: "stars"
}

if business_id_column:

    rename_map[
        business_id_column
    ] = "business_id"


reviews = reviews.rename(
    columns=rename_map
)


if "business_id" not in reviews.columns:

    reviews["business_id"] = ""


reviews["review_id"] = (
    reviews["review_id"]
    .astype(str)
)

reviews["business_id"] = (
    reviews["business_id"]
    .fillna("")
    .astype(str)
)

reviews["text"] = (
    reviews["text"]
    .fillna("")
    .astype(str)
)

reviews["stars"] = pd.to_numeric(
    reviews["stars"],
    errors="coerce"
)

reviews = reviews.dropna(
    subset=["stars"]
)


print(
    f"\nReviews loaded: "
    f"{len(reviews):,}"
)


# ============================================================
# 10. ATTACH REVIEW TEXT/RATING
# ============================================================

print(
    "\nAttaching review information..."
)

aspect_records = aspect_records.merge(
    reviews,
    on="review_id",
    how="left",
    validate="many_to_one"
)

print(
    f"Records after review merge: "
    f"{len(aspect_records):,}"
)


# ============================================================
# 11. EXTRACT LOCAL ASPECT CONTEXT
# ============================================================

def extract_context(
    review_text,
    surface_phrase
):

    review_text = str(
        review_text
    ).strip()

    surface_phrase = str(
        surface_phrase
    ).strip()

    if not review_text:
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        review_text
    )

    if surface_phrase:

        for sentence in sentences:

            if (
                surface_phrase.lower()
                in sentence.lower()
            ):

                return sentence.strip()[:500]

    return review_text[:500]


print(
    "\nExtracting aspect contexts..."
)

tqdm.pandas()

aspect_records["context"] = (
    aspect_records.progress_apply(
        lambda row:
        extract_context(
            row["text"],
            row["surface_phrase"]
        ),
        axis=1
    )
)


# ============================================================
# 12. INITIALIZE VADER
# ============================================================

print(
    "\nInitializing VADER..."
)

analyzer = SentimentIntensityAnalyzer()

print(
    "VADER ready."
)


# ============================================================
# 13. SENTIMENT
# ============================================================

print(
    "\nRunning aspect-level sentiment..."
)


def sentiment_score(text):

    if not text:
        return 0.0

    return analyzer.polarity_scores(
        text
    )["compound"]


def sentiment_label(score):

    if score >= POSITIVE_THRESHOLD:
        return "positive"

    if score <= NEGATIVE_THRESHOLD:
        return "negative"

    return "neutral"


aspect_records["sentiment_score"] = (
    aspect_records["context"]
    .apply(sentiment_score)
)

aspect_records["sentiment"] = (
    aspect_records["sentiment_score"]
    .apply(sentiment_label)
)


# ============================================================
# 14. SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    "review_id",
    "business_id",
    "stars",
    "surface_phrase",
    "aspect_candidate",
    "canonical_parent",
    "canonical_aspect",
    "aspect_type",
    "aspect_value",
    "context",
    "sentiment_score",
    "sentiment"
]

aspect_records = aspect_records[
    output_columns
]


# ============================================================
# 15. REMOVE DUPLICATE OCCURRENCES
# ============================================================

aspect_records = (
    aspect_records
    .drop_duplicates(
        subset=[
            "review_id",
            "canonical_aspect",
            "surface_phrase"
        ]
    )
)


# ============================================================
# 16. SAVE OCCURRENCE DATA
# ============================================================

print(
    "\nSaving occurrence-level data..."
)

aspect_records.to_csv(
    OUTPUT_OCCURRENCES,
    index=False
)

print(
    f"Saved:\n"
    f"{OUTPUT_OCCURRENCES}"
)


# ============================================================
# 17. BUILD REVIEW × ASPECT DATASET
# ============================================================

print(
    "\nBuilding review-level aspect dataset..."
)

review_aspects = (
    aspect_records
    .groupby(
        [
            "review_id",
            "business_id",
            "stars",
            "canonical_parent",
            "canonical_aspect",
            "aspect_type",
            "aspect_value"
        ],
        as_index=False
    )
    .agg(
        mention_count=(
            "surface_phrase",
            "count"
        ),

        average_sentiment=(
            "sentiment_score",
            "mean"
        ),

        positive_mentions=(
            "sentiment",
            lambda x:
            (x == "positive").sum()
        ),

        negative_mentions=(
            "sentiment",
            lambda x:
            (x == "negative").sum()
        ),

        neutral_mentions=(
            "sentiment",
            lambda x:
            (x == "neutral").sum()
        )
    )
)


review_aspects["overall_sentiment"] = (
    review_aspects[
        "average_sentiment"
    ]
    .apply(sentiment_label)
)


# ============================================================
# 18. SAVE REVIEW-ASPECT DATA
# ============================================================

print(
    "\nSaving review-level aspect data..."
)

review_aspects.to_csv(
    OUTPUT_REVIEW_ASPECTS,
    index=False
)


# ============================================================
# 19. FINAL RESULTS
# ============================================================

print("\n========================================")
print("ASPECT SENTIMENT COMPLETE")
print("========================================")

print(
    f"Occurrence records:       "
    f"{len(aspect_records):,}"
)

print(
    f"Review × aspect records:  "
    f"{len(review_aspects):,}"
)

print(
    f"Unique canonical aspects: "
    f"{review_aspects['canonical_aspect'].nunique():,}"
)


print(
    "\nCanonical aspect distribution:"
)

print(
    review_aspects[
        "canonical_aspect"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nSentiment distribution:"
)

print(
    aspect_records[
        "sentiment"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nRating distribution:"
)

print(
    reviews[
        "stars"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    f"\nOccurrence data:\n"
    f"{OUTPUT_OCCURRENCES}"
)

print(
    f"\nReview-aspect data:\n"
    f"{OUTPUT_REVIEW_ASPECTS}"
)