import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\cluster_labeling_input.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_cluster_labels.csv"
)

MODEL_NAME = "openai/gpt-oss-20b"

# IMPORTANT:
# Only 2 clusters per request.
# This keeps the request comfortably small.
BATCH_SIZE = 1

# Wait between requests so we stay below the
# Free plan request-per-minute limit.
SLEEP_SECONDS = 3


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found.\n"
        "Check:\n"
        "C:\\FoodReviewIQ\\.env"
    )

client = Groq(api_key=api_key)


# ============================================================
# LOAD CLUSTERS
# ============================================================

print("Loading cluster data...")

df = pd.read_csv(INPUT_FILE)

df["top_phrases"] = (
    df["top_phrases"]
    .fillna("")
    .astype(str)
)

df["representative_contexts"] = (
    df["representative_contexts"]
    .fillna("")
    .astype(str)
)

df = df.sort_values("cluster_id").reset_index(drop=True)

print(
    f"Total clusters available: {len(df):,}"
)


# ============================================================
# RESUME SUPPORT
# ============================================================

if os.path.exists(OUTPUT_FILE):

    existing = pd.read_csv(OUTPUT_FILE)

    if "cluster_id" in existing.columns:

        completed_ids = set(
            existing["cluster_id"]
            .astype(int)
            .tolist()
        )

        results = existing.to_dict("records")

    else:

        completed_ids = set()
        results = []

    print(
        f"Already completed: "
        f"{len(completed_ids):,}"
    )

else:

    completed_ids = set()
    results = []


remaining = df[
    ~df["cluster_id"].astype(int).isin(
        completed_ids
    )
].copy()

print(
    f"Clusters remaining: "
    f"{len(remaining):,}"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You label clusters of restaurant-review aspect phrases.

For EACH cluster, return exactly ONE result.

KEEP a cluster when it represents a useful restaurant concept:
food, menu items, food quality, taste, portions, service staff,
waiting, orders, reservations, price/value, beverages, ambience,
noise, seating, decor, cleanliness, parking, or similar.

REJECT a cluster when it is mainly:
- a person's name
- a specific city/place/street/business name
- a date, time, number
- generic conversational language
- meaningless noise
- unrelated to restaurant experience
- too ambiguous

A specific place such as "Exton" is NOT an aspect.

Do not assign sentiment.

For kept clusters, provide:
parent_aspect
normalized_aspect

Ignore noisy phrases inside an otherwise useful cluster.

Return valid JSON matching the schema.
"""


# ============================================================
# JSON SCHEMA
# ============================================================

schema = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "integer"
                    },
                    "keep": {
                        "type": "boolean"
                    },
                    "parent_aspect": {
                        "type": "string"
                    },
                    "normalized_aspect": {
                        "type": "string"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": [
                            "high",
                            "medium",
                            "low"
                        ]
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": [
                    "cluster_id",
                    "keep",
                    "parent_aspect",
                    "normalized_aspect",
                    "confidence",
                    "reason"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}


# ============================================================
# COMPACT CLUSTER REPRESENTATION
# ============================================================

def compact_cluster(row):

    # Keep only the first ~12 useful phrases.
    phrases = [
        x.strip()
        for x in row["top_phrases"].split("|")
        if x.strip()
    ]

    phrases = phrases[:12]

    # Use only ONE representative context.
    contexts = row["representative_contexts"]

    # The saved CSV contains literal "\n" separators.
    context_parts = contexts.split("\\n")

    first_context = ""

    for part in context_parts:

        part = part.strip()

        if part.startswith("contexts:"):
            part = part.replace(
                "contexts:",
                "",
                1
            ).strip()

        if part:
            first_context = part
            break

    # Keep the context deliberately short.
    first_context = first_context[:450]

    return f"""
CLUSTER ID: {int(row["cluster_id"])}

TOP PHRASES:
{", ".join(phrases)}

ONE REPRESENTATIVE CONTEXT:
{first_context}
"""


# ============================================================
# BUILD BATCH PROMPT
# ============================================================

def build_prompt(batch):

    cluster_sections = []

    for _, row in batch.iterrows():

        cluster_sections.append(
            compact_cluster(row)
        )

    return """
Label the following restaurant-review clusters.

Return exactly one result for each cluster ID.

""" + "\n--------------------\n".join(
        cluster_sections
    )


# ============================================================
# PROCESS BATCHES
# ============================================================

total = len(remaining)

for start in range(
    0,
    total,
    BATCH_SIZE
):

    batch = remaining.iloc[
        start:start + BATCH_SIZE
    ]

    batch_ids = [
        int(x)
        for x in batch["cluster_id"].tolist()
    ]

    print(
        f"\nProcessing clusters "
        f"{start + 1}-{min(start + BATCH_SIZE, total)} "
        f"of {total}: {batch_ids}"
    )

    prompt = build_prompt(batch)

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "aspect_cluster_labels",
                    "strict": True,
                    "schema": schema
                }
            },

            reasoning_effort="low"
        )

        content = (
            response.choices[0]
            .message.content
        )

        data = json.loads(content)

        batch_results = data["results"]

        # ----------------------------------------------------
        # VALIDATE CLUSTER IDS
        # ----------------------------------------------------

        returned_ids = [
            int(item["cluster_id"])
            for item in batch_results
        ]

        expected_ids = set(batch_ids)
        actual_ids = set(returned_ids)

        if (
            len(returned_ids)
            != len(actual_ids)
        ):
            raise ValueError(
                "Groq returned duplicate cluster IDs."
            )

        if actual_ids != expected_ids:
            raise ValueError(
                f"Cluster ID mismatch.\n"
                f"Expected: {sorted(expected_ids)}\n"
                f"Returned: {sorted(actual_ids)}"
            )

        # ----------------------------------------------------
        # ADD RESULTS
        # ----------------------------------------------------

        for item in batch_results:
            results.append(item)

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        result_df = pd.DataFrame(results)

        result_df = result_df.drop_duplicates(
            subset=["cluster_id"],
            keep="last"
        )

        result_df = result_df.sort_values(
            "cluster_id"
        )

        result_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"Saved {len(result_df)} total labels."
        )

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)

        print(
            "\nAlready completed labels were saved."
        )

        raise

    # Stay below free request-per-minute limit.
    time.sleep(SLEEP_SECONDS)


# ============================================================
# FINAL SUMMARY
# ============================================================

final_df = pd.read_csv(
    OUTPUT_FILE
)

print("\n========================================")
print("ALL CLUSTER LABELING COMPLETE")
print("========================================")

print(
    f"Total labeled clusters: "
    f"{len(final_df):,}"
)

print(
    f"Useful clusters: "
    f"{int(final_df['keep'].sum()):,}"
)

print(
    f"Rejected clusters: "
    f"{int((~final_df['keep']).sum()):,}"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)