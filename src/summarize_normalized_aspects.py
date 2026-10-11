import pandas as pd


# ============================================================
# PATHS
# ============================================================

LABELS_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_cluster_labels.csv"
)

CLUSTERS_FILE = (
    r"C:\FoodReviewIQ\data\processed\cluster_labeling_input.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\normalized_aspect_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

print("Loading cluster labels...")

labels = pd.read_csv(LABELS_FILE)

clusters = pd.read_csv(CLUSTERS_FILE)

print(f"Labels loaded: {len(labels):,}")


# ============================================================
# KEEP ONLY USEFUL CLUSTERS
# ============================================================

labels["keep"] = (
    labels["keep"]
    .fillna(False)
    .astype(bool)
)

useful = labels[
    labels["keep"] == True
].copy()

print(
    f"Useful clusters: {len(useful):,}"
)


# ============================================================
# MERGE CLUSTER INFORMATION
# ============================================================

clusters_small = clusters[
    [
        "cluster_id",
        "cluster_size",
        "top_phrases"
    ]
].copy()

useful = useful.merge(
    clusters_small,
    on="cluster_id",
    how="left"
)


# ============================================================
# CLEAN LABEL TEXT
# ============================================================

for column in [
    "parent_aspect",
    "normalized_aspect"
]:

    useful[column] = (
        useful[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# GROUP BY NORMALIZED ASPECT
# ============================================================

summary = (
    useful
    .groupby(
        [
            "parent_aspect",
            "normalized_aspect"
        ],
        dropna=False
    )
    .agg(
        cluster_count=("cluster_id", "count"),
        total_cluster_candidates=(
            "cluster_size",
            "sum"
        ),
        cluster_ids=(
            "cluster_id",
            lambda x: ", ".join(
                str(int(v))
                for v in x
            )
        ),
        example_phrases=(
            "top_phrases",
            lambda x: " || ".join(
                str(v)
                for v in x.head(5)
            )
        )
    )
    .reset_index()
)


# ============================================================
# SORT
# ============================================================

summary = summary.sort_values(
    by=[
        "parent_aspect",
        "total_cluster_candidates"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n========================================")
print("NORMALIZED ASPECT SUMMARY")
print("========================================")

print(
    f"Unique parent/aspect combinations: "
    f"{len(summary):,}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\n\nASPECTS BY PARENT:\n")

for parent, group in summary.groupby(
    "parent_aspect"
):

    print(
        f"\n### {parent}"
    )

    for _, row in group.iterrows():

        print(
            f"  - {row['normalized_aspect']}"
            f"  "
            f"({int(row['cluster_count'])} clusters)"
        )