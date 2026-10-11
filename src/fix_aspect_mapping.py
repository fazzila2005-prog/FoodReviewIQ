import pandas as pd


# ============================================================
# FILES
# ============================================================

NORMALIZED_FILE = (
    r"C:\FoodReviewIQ\data\processed\normalized_aspect_summary.csv"
)

FINAL_MAPPING_FILE = (
    r"C:\FoodReviewIQ\data\processed\final_aspect_mapping.csv"
)

CLEAN_TAXONOMY_FILE = (
    r"C:\FoodReviewIQ\data\processed\clean_final_taxonomy.csv"
)

CLUSTERS_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_context_clusters.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\cluster_to_aspect_mapping.csv"
)


# ============================================================
# LOAD NORMALIZED SUMMARY
# ============================================================

print("Loading normalized aspect summary...")

normalized = pd.read_csv(
    NORMALIZED_FILE
)

print(
    f"Normalized rows: {len(normalized):,}"
)


# ============================================================
# VERY IMPORTANT:
# mapping_id in final_aspect_mapping.csv was created from
# the ROW POSITION of normalized_aspect_summary.csv.
#
# Recreate exactly the same IDs.
# ============================================================

normalized = normalized.reset_index(drop=True)

normalized["mapping_id"] = normalized.index


# ============================================================
# LOAD FINAL GROQ MAPPING
# ============================================================

print("\nLoading Groq final mapping...")

final_mapping = pd.read_csv(
    FINAL_MAPPING_FILE
)

final_mapping["mapping_id"] = (
    final_mapping["mapping_id"]
    .astype(int)
)

print(
    f"Final mapping rows: "
    f"{len(final_mapping):,}"
)


# ============================================================
# LOAD CLEAN TAXONOMY
# ============================================================

print("\nLoading clean taxonomy...")

clean_taxonomy = pd.read_csv(
    CLEAN_TAXONOMY_FILE
)

clean_taxonomy["mapping_id"] = (
    clean_taxonomy["mapping_id"]
    .astype(int)
)

print(
    f"Clean taxonomy rows: "
    f"{len(clean_taxonomy):,}"
)


# ============================================================
# VALIDATE MAPPING IDS
# ============================================================

normalized_ids = set(
    normalized["mapping_id"]
)

final_ids = set(
    final_mapping["mapping_id"]
)

clean_ids = set(
    clean_taxonomy["mapping_id"]
)

print("\n========================================")
print("MAPPING ID VALIDATION")
print("========================================")

print(
    f"Normalized IDs: {len(normalized_ids):,}"
)

print(
    f"Final mapping IDs: {len(final_ids):,}"
)

print(
    f"Clean taxonomy IDs: {len(clean_ids):,}"
)

if normalized_ids != final_ids:

    raise ValueError(
        "Normalized summary and final mapping IDs "
        "do not match."
    )

if final_ids != clean_ids:

    raise ValueError(
        "Final mapping and clean taxonomy IDs "
        "do not match."
    )

print(
    "Mapping IDs are consistent."
)


# ============================================================
# BUILD MAPPING ID → CANONICAL ASPECT
# ============================================================

taxonomy_map = (
    final_mapping[
        [
            "mapping_id",
            "canonical_parent",
            "canonical_aspect"
        ]
    ]
    .merge(
        clean_taxonomy[
            [
                "mapping_id",
                "aspect_type",
                "aspect_value"
            ]
        ],
        on="mapping_id",
        how="left"
    )
)


# ============================================================
# EXPAND CLUSTER IDS
# ============================================================

print(
    "\nExpanding cluster IDs..."
)

cluster_rows = []

for _, row in normalized.iterrows():

    mapping_id = int(
        row["mapping_id"]
    )

    cluster_ids = str(
        row["cluster_ids"]
    ).split(",")

    for cluster_id in cluster_ids:

        cluster_id = cluster_id.strip()

        if not cluster_id:
            continue

        try:
            cluster_id = int(
                cluster_id
            )
        except ValueError:
            continue

        cluster_rows.append({
            "cluster_id": cluster_id,
            "mapping_id": mapping_id
        })


cluster_map = pd.DataFrame(
    cluster_rows
)


print(
    f"Expanded cluster mappings: "
    f"{len(cluster_map):,}"
)


# ============================================================
# CHECK DUPLICATE CLUSTER ASSIGNMENTS
# ============================================================

duplicate_clusters = (
    cluster_map[
        cluster_map["cluster_id"].duplicated(
            keep=False
        )
    ]
)

if len(duplicate_clusters) > 0:

    print(
        "\nWARNING:"
    )

    print(
        duplicate_clusters
        .sort_values("cluster_id")
        .head(30)
        .to_string(index=False)
    )

    raise ValueError(
        "Some cluster IDs map to multiple taxonomy rows."
    )


# ============================================================
# ATTACH CANONICAL TAXONOMY
# ============================================================

cluster_mapping = (
    cluster_map
    .merge(
        taxonomy_map,
        on="mapping_id",
        how="left"
    )
)


# ============================================================
# CHECK MISSING TAXONOMY
# ============================================================

missing_taxonomy = (
    cluster_mapping[
        cluster_mapping["canonical_aspect"]
        .isna()
    ]
)

print(
    f"\nClusters without taxonomy: "
    f"{len(missing_taxonomy):,}"
)

if len(missing_taxonomy) > 0:

    print(
        missing_taxonomy[
            [
                "cluster_id",
                "mapping_id"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# CHECK HOW MANY CLUSTERS ARE MAPPED
# ============================================================

print("\n========================================")
print("CLUSTER → ASPECT VALIDATION")
print("========================================")

print(
    f"Total clusters in mapping: "
    f"{cluster_mapping['cluster_id'].nunique():,}"
)

print(
    f"Mapped canonical aspects: "
    f"{cluster_mapping['canonical_aspect'].notna().sum():,}"
)

print(
    f"Unique canonical aspects: "
    f"{cluster_mapping['canonical_aspect'].nunique():,}"
)


# ============================================================
# DISPLAY ASPECT DISTRIBUTION
# ============================================================

print(
    "\nCanonical aspect distribution:"
)

distribution = (
    cluster_mapping[
        cluster_mapping["canonical_aspect"].notna()
    ]
    .groupby(
        [
            "canonical_parent",
            "canonical_aspect"
        ]
    )
    .agg(
        cluster_count=(
            "cluster_id",
            "count"
        )
    )
    .reset_index()
    .sort_values(
        "cluster_count",
        ascending=False
    )
)

print(
    distribution.to_string(index=False)
)


# ============================================================
# SAVE AUTHORITATIVE MAPPING
# ============================================================

cluster_mapping.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved authoritative mapping to:"
)

print(
    OUTPUT_FILE
)