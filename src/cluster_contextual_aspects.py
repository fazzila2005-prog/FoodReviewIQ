import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# PATHS
# ============================================================

CANDIDATES_FILE = (
    r"C:\FoodReviewIQ\data\processed\refined_aspect_candidates.csv"
)

OCCURRENCES_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_candidates.csv"
)

REVIEWS_FILE = (
    r"C:\FoodReviewIQ\data\processed\processed_reviews.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_context_clusters.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# Number of intermediate semantic clusters.
N_CLUSTERS = 200

# Maximum number of real review examples used
# to represent one candidate aspect.
CONTEXTS_PER_ASPECT = 3

BATCH_SIZE = 256

RANDOM_STATE = 42


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("Loading refined candidates...")

candidates = pd.read_csv(CANDIDATES_FILE)

candidates["aspect_candidate"] = (
    candidates["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

candidates = candidates[
    candidates["aspect_candidate"] != ""
].copy()

valid_candidates = set(
    candidates["aspect_candidate"]
)

print(
    f"Refined candidates: "
    f"{len(valid_candidates):,}"
)


# ============================================================
# LOAD REVIEW TEXT
# ============================================================

print("\nLoading review texts...")

reviews = pd.read_csv(
    REVIEWS_FILE,
    usecols=["review_id", "text"]
)

reviews["review_id"] = (
    reviews["review_id"]
    .astype(str)
)

reviews["text"] = (
    reviews["text"]
    .fillna("")
    .astype(str)
)

print(
    f"Reviews loaded: "
    f"{len(reviews):,}"
)


# ============================================================
# LOAD ASPECT OCCURRENCES
# ============================================================

print("\nLoading aspect occurrences...")

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

occurrences["aspect_candidate"] = (
    occurrences["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Keep only our refined candidates.
occurrences = occurrences[
    occurrences["aspect_candidate"].isin(valid_candidates)
].copy()

print(
    f"Useful occurrences: "
    f"{len(occurrences):,}"
)


# ============================================================
# MERGE REVIEW TEXT
# ============================================================

print("\nJoining review contexts...")

occurrences = occurrences.merge(
    reviews,
    on="review_id",
    how="left"
)

occurrences["text"] = (
    occurrences["text"]
    .fillna("")
    .astype(str)
)

occurrences["surface_phrase"] = (
    occurrences["surface_phrase"]
    .fillna("")
    .astype(str)
)


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def get_context(review_text, surface_phrase):

    review_text = review_text.strip()

    if not review_text:
        return ""

    # Split review into rough sentences.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        review_text
    )

    phrase = surface_phrase.strip()

    # Look for the sentence containing the
    # original surface phrase.
    if phrase:

        for sentence in sentences:

            if phrase.lower() in sentence.lower():

                return sentence.strip()[:500]

    # If no matching sentence was found,
    # use a short part of the review.
    return review_text[:500]


print("\nExtracting aspect contexts...")

occurrences["context"] = occurrences.apply(
    lambda row: get_context(
        row["text"],
        row["surface_phrase"]
    ),
    axis=1
)

occurrences = occurrences[
    occurrences["context"].str.len() > 0
].copy()


# ============================================================
# SELECT REPRESENTATIVE CONTEXTS
# ============================================================

print("\nSelecting representative contexts...")

sampled = (
    occurrences
    .groupby(
        "aspect_candidate",
        group_keys=False
    )
    .apply(
        lambda x: x.sample(
            n=min(
                len(x),
                CONTEXTS_PER_ASPECT
            ),
            random_state=RANDOM_STATE
        )
    )
    .reset_index(drop=True)
)

print(
    f"Context rows selected: "
    f"{len(sampled):,}"
)


# ============================================================
# BUILD ONE REPRESENTATION PER ASPECT
# ============================================================

print("\nBuilding contextual representations...")

context_records = []

for aspect, group in sampled.groupby(
    "aspect_candidate"
):

    contexts = group["context"].tolist()

    representation = (
        f"aspect: {aspect}\n"
        f"contexts: "
        + " | ".join(contexts)
    )

    context_records.append({
        "aspect_candidate": aspect,
        "context_representation": representation
    })


context_df = pd.DataFrame(context_records)

print(
    f"Aspect representations: "
    f"{len(context_df):,}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# ============================================================
# CREATE CONTEXTUAL EMBEDDINGS
# ============================================================

print(
    "\nCreating contextual embeddings..."
)

representations = (
    context_df["context_representation"]
    .tolist()
)

embeddings = model.encode(
    representations,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}"
)


# ============================================================
# MINI-BATCH KMEANS
# ============================================================

print(
    f"\nCreating {N_CLUSTERS} semantic clusters..."
)

cluster_model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    batch_size=BATCH_SIZE,
    random_state=RANDOM_STATE,
    n_init=10
)

labels = cluster_model.fit_predict(
    embeddings
)

context_df["cluster_id"] = labels


# ============================================================
# ADD FREQUENCY INFORMATION
# ============================================================

frequency_lookup = candidates[
    [
        "aspect_candidate",
        "frequency",
        "unique_reviews"
    ]
].copy()

context_df = context_df.merge(
    frequency_lookup,
    on="aspect_candidate",
    how="left"
)


# ============================================================
# CLUSTER SIZE
# ============================================================

cluster_sizes = (
    context_df["cluster_id"]
    .value_counts()
    .rename("cluster_size")
)

context_df["cluster_size"] = (
    context_df["cluster_id"]
    .map(cluster_sizes)
)


# ============================================================
# SORT
# ============================================================

context_df = context_df.sort_values(
    by=[
        "cluster_id",
        "frequency"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# SAVE
# ============================================================

context_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("CONTEXTUAL CLUSTERING COMPLETE")
print("========================================")

print(
    f"Aspect candidates: "
    f"{len(context_df):,}"
)

print(
    f"Clusters created: "
    f"{N_CLUSTERS}"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# SAMPLE CLUSTERS
# ============================================================

print("\nSample clusters:\n")

cluster_sizes_sorted = (
    context_df
    .groupby("cluster_id")
    .size()
    .sort_values(
        ascending=False
    )
)

for cluster_id in cluster_sizes_sorted.head(25).index:

    cluster_df = context_df[
        context_df["cluster_id"] == cluster_id
    ].sort_values(
        "frequency",
        ascending=False
    )

    phrases = cluster_df.head(12)[
        "aspect_candidate"
    ].tolist()

    print(
        f"\nCluster {cluster_id}"
    )

    print(
        f"Size: "
        f"{len(cluster_df)}"
    )

    print(
        " | ".join(phrases)
    )