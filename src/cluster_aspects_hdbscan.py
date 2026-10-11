import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import hdbscan


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"C:\FoodReviewIQ\data\processed\refined_aspect_candidates.csv"

OUTPUT_FILE = r"C:\FoodReviewIQ\data\processed\aspect_clusters_hdbscan.csv"


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# Minimum number of phrases required to form a cluster.
MIN_CLUSTER_SIZE = 15

# Controls how conservative clustering is.
MIN_SAMPLES = 5

BATCH_SIZE = 256


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("Loading refined candidates...")

df = pd.read_csv(INPUT_FILE)

df["aspect_candidate"] = (
    df["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["aspect_candidate"] != ""].copy()

print(f"Candidates loaded: {len(df):,}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

phrases = df["aspect_candidate"].tolist()

print(f"\nCreating embeddings for {len(phrases):,} candidates...")

embeddings = model.encode(
    phrases,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings)

print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# HDBSCAN
# ============================================================

print("\nRunning HDBSCAN clustering...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom"
)

labels = clusterer.fit_predict(embeddings)

df["cluster_id"] = labels


# ============================================================
# CLUSTER PROBABILITY
# ============================================================

df["cluster_probability"] = clusterer.probabilities_


# ============================================================
# CLUSTER SIZES
# ============================================================

cluster_sizes = (
    df[df["cluster_id"] != -1]["cluster_id"]
    .value_counts()
    .rename("cluster_size")
)

df["cluster_size"] = (
    df["cluster_id"]
    .map(cluster_sizes)
    .fillna(0)
    .astype(int)
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    by=["cluster_id", "frequency"],
    ascending=[True, False]
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

clustered = df[df["cluster_id"] != -1]
noise = df[df["cluster_id"] == -1]

print("\n========================================")
print("HDBSCAN CLUSTERING COMPLETE")
print("========================================")

print(f"Candidates:          {len(df):,}")
print(f"Meaningful clusters: {clustered['cluster_id'].nunique():,}")
print(f"Clustered candidates:{len(clustered):,}")
print(f"Noise / outliers:    {len(noise):,}")

print(f"\nSaved to:")
print(OUTPUT_FILE)


# ============================================================
# DISPLAY CLUSTERS
# ============================================================

print("\nSample clusters:\n")

cluster_ids = (
    clustered["cluster_id"]
    .value_counts()
    .head(25)
    .index
)

for cluster_id in cluster_ids:

    cluster_df = clustered[
        clustered["cluster_id"] == cluster_id
    ].sort_values(
        "frequency",
        ascending=False
    )

    top_phrases = cluster_df.head(12)[
        "aspect_candidate"
    ].tolist()

    print(f"\nCluster {cluster_id}")
    print(f"Size: {len(cluster_df)}")
    print(" | ".join(top_phrases))


# ============================================================
# NOISE SAMPLE
# ============================================================

print("\n========================================")
print("NOISE SAMPLE")
print("========================================")

print(
    noise.sort_values(
        "frequency",
        ascending=False
    )[
        ["aspect_candidate", "frequency"]
    ].head(50).to_string(index=False)
)