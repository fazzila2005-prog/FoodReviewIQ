import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\FoodReviewIQ")
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REVIEWS_FILE = PROCESSED_DIR / "processed_reviews.csv"
ASPECTS_FILE = PROCESSED_DIR / "review_aspects.csv"
OUTPUT_FILE = PROCESSED_DIR / "rag_documents.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed reviews...")
reviews = pd.read_csv(REVIEWS_FILE)

print("Loading review aspects...")
aspects = pd.read_csv(ASPECTS_FILE)

print(f"Reviews loaded: {len(reviews):,}")
print(f"Aspect records loaded: {len(aspects):,}")


# ============================================================
# CLEAN IMPORTANT COLUMNS
# ============================================================

reviews["review_id"] = reviews["review_id"].astype(str)
aspects["review_id"] = aspects["review_id"].astype(str)

reviews["text"] = reviews["text"].fillna("").astype(str)

reviews["restaurant_name"] = reviews["restaurant_name"].fillna("Unknown")
reviews["city"] = reviews["city"].fillna("Unknown")
reviews["state"] = reviews["state"].fillna("Unknown")

reviews["review_stars"] = pd.to_numeric(
    reviews["review_stars"],
    errors="coerce"
)


# ============================================================
# CREATE ASPECT SUMMARY FOR EACH REVIEW
# ============================================================

print("\nBuilding aspect summaries...")

def create_aspect_summary(group):

    items = []

    for _, row in group.iterrows():

        aspect = str(row["canonical_aspect"])

        sentiment = str(row["overall_sentiment"])

        items.append(
            f"{aspect} ({sentiment})"
        )

    # Remove duplicates while preserving order
    items = list(dict.fromkeys(items))

    return "; ".join(items)


aspect_summary = (
    aspects
    .groupby("review_id")
    .apply(create_aspect_summary)
    .reset_index(name="aspect_summary")
)


print(
    f"Reviews with discovered aspects: "
    f"{len(aspect_summary):,}"
)


# ============================================================
# MERGE REVIEWS + ASPECT INFORMATION
# ============================================================

print("\nMerging review and aspect information...")

rag = reviews[
    [
        "review_id",
        "business_id",
        "restaurant_name",
        "city",
        "state",
        "business_stars",
        "review_stars",
        "date",
        "text",
        "useful",
        "funny",
        "cool",
    ]
].copy()

rag = rag.merge(
    aspect_summary,
    on="review_id",
    how="left"
)


rag["aspect_summary"] = rag["aspect_summary"].fillna(
    "No specific aspect identified"
)


# ============================================================
# CREATE RAG DOCUMENT TEXT
# ============================================================

print("\nCreating RAG documents...")

def create_document(row):

    return (
        f"Restaurant: {row['restaurant_name']}\n"
        f"Location: {row['city']}, {row['state']}\n"
        f"Restaurant Rating: {row['business_stars']}\n"
        f"Review Rating: {row['review_stars']}\n"
        f"Review Date: {row['date']}\n"
        f"Detected Aspects: {row['aspect_summary']}\n"
        f"Review:\n{row['text']}"
    )


tqdm.pandas(desc="Creating documents")

rag["document"] = rag.progress_apply(
    create_document,
    axis=1
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
    "text",
    "document",
]

rag = rag[output_columns]

rag.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RAG DOCUMENT BUILD COMPLETE")
print("=" * 60)

print(f"Documents created: {len(rag):,}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Columns: {list(rag.columns)}")

print("\nExample document:")
print("-" * 60)
print(rag.iloc[0]["document"][:1500])