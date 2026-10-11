import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_context_clusters.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\cluster_labeling_input.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of representative phrases we want the LLM to see
# for each cluster.
TOP_PHRASES_PER_CLUSTER = 20


# ============================================================
# LOAD
# ============================================================

print("Loading contextual clusters...")

df = pd.read_csv(INPUT_FILE)

df["aspect_candidate"] = (
    df["aspect_candidate"]
    .fillna("")
    .astype(str)
)

df["context_representation"] = (
    df["context_representation"]
    .fillna("")
    .astype(str)
)

print(
    f"Candidates loaded: {len(df):,}"
)


# ============================================================
# BUILD CLUSTER SUMMARIES
# ============================================================

cluster_rows = []

for cluster_id, group in df.groupby("cluster_id"):

    group = group.sort_values(
        "frequency",
        ascending=False
    )

    # Take the most frequent phrases.
    phrases = (
        group["aspect_candidate"]
        .head(TOP_PHRASES_PER_CLUSTER)
        .tolist()
    )

    # Remove duplicates while preserving order.
    phrases = list(
        dict.fromkeys(phrases)
    )

    # Take a few representative contexts.
    contexts = (
        group["context_representation"]
        .head(5)
        .tolist()
    )

    cluster_rows.append({
        "cluster_id": int(cluster_id),

        "cluster_size": int(
            len(group)
        ),

        "top_phrases": " | ".join(
            phrases
        ),

        "representative_contexts": "\n".join(
            contexts
        )
    })


# ============================================================
# SAVE
# ============================================================

summary_df = pd.DataFrame(
    cluster_rows
)

summary_df = summary_df.sort_values(
    "cluster_id"
)

summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("CLUSTER LABELING INPUT CREATED")
print("========================================")

print(
    f"Clusters: "
    f"{len(summary_df):,}"
)

print(
    f"Saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\nExample:\n")

print(
    summary_df.head(5).to_string(
        index=False
    )
)