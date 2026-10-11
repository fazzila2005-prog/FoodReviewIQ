import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\FoodReviewIQ")

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_selected_reviews.csv"
)

CHROMA_DIR = (
    BASE_DIR
    / "data"
    / "chroma_db"
)

COLLECTION_NAME = "foodreviewiq_reviews"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 128


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("FOODREVIEWIQ VECTOR DATABASE")
print("=" * 60)

print("\nLoading selected RAG reviews...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Reviews loaded: {len(df):,}"
)


# ============================================================
# CLEAN DATA
# ============================================================

text_columns = [
    "review_id",
    "business_id",
    "restaurant_name",
    "city",
    "state",
    "date",
    "aspect_summary",
    "review_sentiment",
    "rating_group",
    "text",
    "embedding_text",
]

for column in text_columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

print(
    f"Model: {MODEL_NAME}"
)

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Embedding model loaded.")


# ============================================================
# CHECK EMBEDDING DIMENSION
# ============================================================

test_embedding = model.encode(
    ["test sentence"],
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(
    f"Embedding dimension: "
    f"{test_embedding.shape[1]}"
)


# ============================================================
# CREATE CHROMADB
# ============================================================

print("\nCreating ChromaDB...")

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# RESET COLLECTION
# ============================================================

try:

    client.delete_collection(
        name=COLLECTION_NAME
    )

    print("Existing collection deleted.")

except Exception:

    pass


collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description":
        "FoodReviewIQ representative Yelp review RAG collection"
    }
)

print(
    f"Collection created: "
    f"{COLLECTION_NAME}"
)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

documents = (
    df["embedding_text"]
    .tolist()
)

embeddings = model.encode(
    documents,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\nEmbedding generation complete.")

print(
    f"Embedding matrix shape: "
    f"{embeddings.shape}"
)


# ============================================================
# PREPARE METADATA
# ============================================================

print("\nPreparing metadata...")

ids = (
    df["review_id"]
    .tolist()
)

metadata_list = []

for _, row in df.iterrows():

    metadata = {

        "business_id":
            str(row["business_id"]),

        "restaurant_name":
            str(row["restaurant_name"]),

        "city":
            str(row["city"]),

        "state":
            str(row["state"]),

        "review_stars":
            float(row["review_stars"])
            if pd.notna(row["review_stars"])
            else 0.0,

        "business_stars":
            float(row["business_stars"])
            if pd.notna(row["business_stars"])
            else 0.0,

        "date":
            str(row["date"]),

        "review_sentiment":
            str(row["review_sentiment"]),

        "rating_group":
            str(row["rating_group"]),

        "aspect_summary":
            str(row["aspect_summary"]),
    }

    metadata_list.append(
        metadata
    )


# ============================================================
# STORE IN CHROMADB
# ============================================================

print("\nStoring vectors in ChromaDB...")

for start in tqdm(
    range(
        0,
        len(df),
        BATCH_SIZE
    ),
    desc="Storing vectors"
):

    end = min(
        start + BATCH_SIZE,
        len(df)
    )

    collection.add(

        ids=ids[start:end],

        embeddings=(
            embeddings[start:end]
            .tolist()
        ),

        documents=(
            documents[start:end]
        ),

        metadatas=(
            metadata_list[start:end]
        )
    )


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 60)
print("VECTOR DATABASE BUILD COMPLETE")
print("=" * 60)

print(
    f"Documents in ChromaDB: "
    f"{collection.count():,}"
)

print(
    f"Database location: "
    f"{CHROMA_DIR}"
)


# ============================================================
# TEST SEMANTIC SEARCH
# ============================================================

print("\nTesting semantic search...")

query = (
    "customers complaining about "
    "long waiting times and slow service"
)

print(
    f"\nQuery: {query}"
)

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)

results = collection.query(
    query_embeddings=
        query_embedding.tolist(),

    n_results=5,

    include=[
        "documents",
        "metadatas",
        "distances",
    ]
)


print("\nTop 5 retrieved reviews:")
print("=" * 60)


for i in range(5):

    metadata = (
        results["metadatas"][0][i]
    )

    document = (
        results["documents"][0][i]
    )

    review_id = (
        results["ids"][0][i]
    )

    distance = (
        results["distances"][0][i]
    )

    print(
        f"\nRESULT {i + 1}"
    )

    print("-" * 60)

    print(
        f"Review ID: {review_id}"
    )

    print(
        f"Restaurant: "
        f"{metadata['restaurant_name']}"
    )

    print(
        f"Review rating: "
        f"{metadata['review_stars']}"
    )

    print(
        f"Sentiment: "
        f"{metadata['review_sentiment']}"
    )

    print(
        f"Distance: "
        f"{distance:.4f}"
    )

    print("\nReview:")

    print(
        document[:600]
        .replace("\n", " ")
    )


print("\n" + "=" * 60)
print("RAG VECTOR DATABASE READY")
print("=" * 60)