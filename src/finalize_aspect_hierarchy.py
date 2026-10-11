import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\normalized_aspect_summary.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\final_aspect_mapping.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

# One mapping per request = much safer structured output.
SLEEP_SECONDS = 3


# ============================================================
# LOAD GROQ
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Check C:\\FoodReviewIQ\\.env"
    )

client = Groq(api_key=api_key)


# ============================================================
# LOAD INPUT
# ============================================================

print("Loading current aspect labels...")

df = pd.read_csv(INPUT_FILE)

df["parent_aspect"] = (
    df["parent_aspect"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["normalized_aspect"] = (
    df["normalized_aspect"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df.reset_index(drop=True)

# Stable ID for each input row.
df["mapping_id"] = df.index

print(
    f"Aspect combinations loaded: {len(df):,}"
)


# ============================================================
# RESUME FROM SAVED RESULTS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    existing = pd.read_csv(OUTPUT_FILE)

    if "mapping_id" in existing.columns:

        completed_ids = set(
            existing["mapping_id"]
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
    ~df["mapping_id"].isin(completed_ids)
].copy()

print(
    f"Remaining: "
    f"{len(remaining):,}"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are creating the final aspect taxonomy for FoodReviewIQ,
a restaurant review intelligence system.

Normalize ONE restaurant-review concept.

Merge synonymous or nearly identical labels into a clean,
consistent taxonomy.

Preferred parent categories:

Food
Menu
Beverages
Service
Delivery
Price & Value
Ambience
Location & Access
Cleanliness & Maintenance
Dining Experience
Events & Group Dining
Restaurant Type
Other

Examples:

service staff / staff / employee / busser
→ Service / Service Staff

friendly staff / staff friendliness / warm welcome
→ Service / Staff Attitude

waiting / waiting time / service timing / long wait
→ Service / Waiting Time

price / pricing / cost / bill
→ Price & Value / Pricing

value / price value / good value
→ Price & Value / Value for Money

taste / flavor
→ Food / Taste & Flavor

portion / portion size
→ Food / Portion Size

ambience / atmosphere
→ Ambience / Overall Ambience

noise / music / noise level
→ Ambience / Noise Level

menu variety / selection / choices
→ Menu / Menu Variety

pizza / burger / wrap / sushi
→ Food / Specific Food Item

Preserve meaningful distinctions.

Do NOT assign sentiment.

Do NOT invent an aspect that is unsupported by the supplied
label and example phrases.

Return exactly one mapping.
"""


# ============================================================
# SIMPLE OBJECT SCHEMA
# ============================================================

schema = {
    "type": "object",

    "properties": {

        "mapping_id": {
            "type": "integer"
        },

        "canonical_parent": {
            "type": "string"
        },

        "canonical_aspect": {
            "type": "string"
        },

        "merge_reason": {
            "type": "string"
        }
    },

    "required": [
        "mapping_id",
        "canonical_parent",
        "canonical_aspect",
        "merge_reason"
    ],

    "additionalProperties": False
}


# ============================================================
# PROCESS ONE ROW AT A TIME
# ============================================================

total = len(remaining)

for position, (_, row) in enumerate(
    remaining.iterrows(),
    start=1
):

    mapping_id = int(
        row["mapping_id"]
    )

    print(
        f"\nProcessing "
        f"{position}/{total} "
        f"(mapping_id={mapping_id})"
    )

    user_prompt = f"""
Normalize this restaurant-review concept.

MAPPING ID:
{mapping_id}

CURRENT PARENT:
{row["parent_aspect"]}

CURRENT ASPECT:
{row["normalized_aspect"]}

NUMBER OF SOURCE CLUSTERS:
{row["cluster_count"]}

EXAMPLE PHRASES:
{row["example_phrases"]}

Return exactly one mapping for mapping_id {mapping_id}.
"""

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
                    "content": user_prompt
                }
            ],

            response_format={
                "type": "json_schema",

                "json_schema": {
                    "name": "aspect_mapping",

                    "strict": True,

                    "schema": schema
                }
            },

            reasoning_effort="low"
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        result = json.loads(content)

        # ----------------------------------------------------
        # VALIDATE ID
        # ----------------------------------------------------

        returned_id = int(
            result["mapping_id"]
        )

        if returned_id != mapping_id:

            raise ValueError(
                f"Wrong mapping ID returned. "
                f"Expected {mapping_id}, "
                f"got {returned_id}"
            )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(result)

        result_df = pd.DataFrame(
            results
        )

        result_df = (
            result_df
            .drop_duplicates(
                subset=["mapping_id"],
                keep="last"
            )
            .sort_values(
                "mapping_id"
            )
        )

        result_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"Saved {len(result_df)} mappings."
        )

    except Exception as e:

        print("\nERROR:")
        print(e)

        print(
            "\nAlready completed mappings were saved."
        )

        raise

    # Stay comfortably below the free request limit.
    time.sleep(SLEEP_SECONDS)


# ============================================================
# FINAL SUMMARY
# ============================================================

final_df = pd.read_csv(
    OUTPUT_FILE
)

print("\n========================================")
print("FINAL ASPECT NORMALIZATION COMPLETE")
print("========================================")

print(
    f"Input combinations: "
    f"{len(df):,}"
)

print(
    f"Mappings created: "
    f"{len(final_df):,}"
)

print(
    f"Unique canonical parents: "
    f"{final_df['canonical_parent'].nunique():,}"
)

print(
    f"Unique canonical aspects: "
    f"{final_df['canonical_aspect'].nunique():,}"
)

print("\nFINAL TAXONOMY:\n")

taxonomy = (
    final_df[
        [
            "canonical_parent",
            "canonical_aspect"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "canonical_parent",
            "canonical_aspect"
        ]
    )
)

print(
    taxonomy.to_string(
        index=False
    )
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)