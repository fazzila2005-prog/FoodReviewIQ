import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"C:\FoodReviewIQ\data\processed\filtered_aspect_candidates.csv"

OUTPUT_FILE = r"C:\FoodReviewIQ\data\processed\aspect_clusters.csv"


# ============================================================
# SETTINGS
# ============================================================

# Number of semantic groups to create.
#
# This is NOT our final number of restaurant aspects.
# These are just intermediate semantic clusters.
N_CLUSTERS = 100

RANDOM_STATE = 42

BATCH_SIZE = 256


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("Loading candidate aspects...")

df = pd.read_csv(INPUT_FILE)

df["aspect_candidate"] = (
    df["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove empty rows just in case.
df = df[df["aspect_candidate"] != ""].copy()

print(f"Candidates loaded: {len(df):,}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

phrases = df["aspect_candidate"].tolist()

print(f"\nCreating embeddings for {len(phrases):,} phrases...")

embeddings = model.encode(
    phrases,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings)

print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# CLUSTER EMBEDDINGS
# ============================================================

print(f"\nCreating {N_CLUSTERS} semantic clusters...")

cluster_model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    n_init=10
)

cluster_labels = cluster_model.fit_predict(embeddings)

df["cluster_id"] = cluster_labels


# ============================================================
# ADD CLUSTER SIZE
# ============================================================

cluster_sizes = (
    df["cluster_id"]
    .value_counts()
    .rename("cluster_size")
)

df = df.merge(
    cluster_sizes,
    left_on="cluster_id",
    right_index=True,
    how="left"
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
# DISPLAY SAMPLE CLUSTERS
# ============================================================

print("\n========================================")
print("SEMANTIC CLUSTERING COMPLETE")
print("========================================")

print(f"Candidates clustered: {len(df):,}")
print(f"Clusters created: {N_CLUSTERS}")
print(f"Saved to: {OUTPUT_FILE}")


print("\nSample cluster contents:\n")

for cluster_id in sorted(df["cluster_id"].unique())[:20]:

    cluster_df = df[df["cluster_id"] == cluster_id]

    top_phrases = cluster_df.head(10)["aspect_candidate"].tolist()

    print(f"\nCluster {cluster_id}")
    print(f"Size: {len(cluster_df)}")
    print(" | ".join(top_phrases))