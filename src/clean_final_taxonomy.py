import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\final_aspect_mapping.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\clean_final_taxonomy.csv"
)

SUMMARY_FILE = (
    r"C:\FoodReviewIQ\data\processed\clean_taxonomy_summary.csv"
)


# ============================================================
# CONTEXT TAGS
# ============================================================

CONTEXT_TAGS = {
    "breakfast",
    "lunch",
    "brunch",
    "birthday party",
    "events & group dining",
    "group size",
    "indian food",
    "thai cuisine",
    "vegetarian food",
    "casual dining",
    "franchise",
    "casino",
    "takeout",
    "outdoor seating",
    "private dining",
}


# ============================================================
# SPECIFIC FOOD ITEMS
# ============================================================

SPECIFIC_FOOD_ITEMS = {
    "pizza",
    "burger",
    "wrap",
    "sushi",
    "salad",
    "sandwich",
    "caesar salad",
    "cajun pasta",
    "cuban bread",
    "buffalo chicken sandwich",
    "chicken chimichanga",
    "chicken pieces",
    "pappardelle",
    "penne with vodka sauce",
    "shrimp fried rice",
    "sourdough bread",
    "spaghetti",
    "veal chop",
    "acai bowl",
    "shrimp tempura",
    "spicy cucumber",
    "nashville hot cauliflower",
    "waffle cone",
    "alligator",
    "seafood",
    "meat",
    "dessert",
    "desserts",
    "chips",
    "steamer",
    "raw sushi",
    "fried chicken livers",
    "house chips",
    "lobster risotto",
    "naan",
    "oxtail",
    "roast duck",
    "souvlaki",
    "shrimp fried rice",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_parent(parent):

    parent = str(parent).strip()

    # Flatten accidental nested parent categories.
    if parent.startswith("Food /"):
        return "Food"

    if parent.startswith("Service /"):
        return "Service"

    if parent.startswith("Food/"):
        return "Food"

    if parent.startswith("Service/"):
        return "Service"

    return parent


def normalize_aspect(parent, aspect):

    parent = normalize_parent(parent)

    aspect = str(aspect).strip()

    lower = aspect.lower()

    # --------------------------------------------------------
    # CONTEXT TAGS
    # --------------------------------------------------------

    if lower in CONTEXT_TAGS:

        return {
            "canonical_parent": "Context",
            "canonical_aspect": lower.title(),
            "aspect_type": "context_tag",
            "aspect_value": lower,
        }

    # --------------------------------------------------------
    # FOOD
    # --------------------------------------------------------

    if lower in {
        "food",
        "food quality",
        "food_quality_disappointment",
    }:
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Food Quality",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "taste",
        "taste & flavor",
        "food flavor",
        "flavor",
    }:
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Taste & Flavor",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "portion",
        "portion size",
        "nice sized portion",
    }:
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Portion Size",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "freshness",
        "ingredient freshness",
        "ingredient use",
        "crisp lettuce",
    }:
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Ingredient Quality / Freshness",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "temperature":
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Food Temperature",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "side",
        "side dish",
        "side dishes",
    }:
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Side Dishes",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "sweetener",
        "sugar",
    }:
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Sweeteners",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    # Specific food items
    if lower in SPECIFIC_FOOD_ITEMS:
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Specific Food Item",
            "aspect_type": "specific_food_item",
            "aspect_value": lower,
        }

    # Things that clearly look like specific food/cuisine items.
    if (
        lower.endswith(" food")
        or lower.endswith(" cuisine")
        or "sushi" in lower
        or "pasta" in lower
        or "bread" in lower
        or "sandwich" in lower
        or "chicken" in lower
        or "shrimp" in lower
        or "salad" in lower
        or "burger" in lower
    ):
        return {
            "canonical_parent": "Food",
            "canonical_aspect": "Specific Food Item",
            "aspect_type": "specific_food_item",
            "aspect_value": lower,
        }

    # --------------------------------------------------------
    # MENU
    # --------------------------------------------------------

    if lower in {
        "menu",
        "menu / drink menu",
    }:
        return {
            "canonical_parent": "Menu",
            "canonical_aspect": "Menu",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "menu variety":
        return {
            "canonical_parent": "Menu",
            "canonical_aspect": "Menu Variety",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "specific food item":
        return {
            "canonical_parent": "Menu",
            "canonical_aspect": "Specific Food Item",
            "aspect_type": "specific_food_item",
            "aspect_value": "",
        }

    # --------------------------------------------------------
    # BEVERAGES
    # --------------------------------------------------------

    if lower in {
        "beer",
        "cocktails",
        "milkshake",
        "lassi",
        "beverages",
        "specific beverage",
        "specific beverage item",
        "beverages / specific beverage",
        "vodka",
        "sangria",
    }:
        return {
            "canonical_parent": "Beverages",
            "canonical_aspect": "Specific Beverage",
            "aspect_type": "specific_beverage",
            "aspect_value": lower,
        }

    # --------------------------------------------------------
    # SERVICE
    # --------------------------------------------------------

    if lower in {
        "service",
        "service issues",
        "service quality",
        "service performance",
        "overall service experience",
        "issues",
    }:
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Service Quality",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "service staff",
        "service_staff",
        "staff",
        "employee",
        "busser",
    }:
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Service Staff",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "staff attitude",
        "staff friendliness",
        "friendly staff",
        "eye contact",
        "warm welcome",
    }:
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Staff Attitude",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "staff availability",
        "staff absence",
    }:
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Staff Availability",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "waiting time",
        "waiting",
        "service timing",
        "timely service",
        "food delivery time",
    }:
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Waiting Time",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "order process":
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Order Process",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "tableware",
        "silverware",
    }:
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Tableware",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "takeout",
        "grab_and_go",
    }:
        return {
            "canonical_parent": "Service",
            "canonical_aspect": "Takeout",
            "aspect_type": "context_tag",
            "aspect_value": "takeout",
        }

    # --------------------------------------------------------
    # DELIVERY
    # --------------------------------------------------------

    if lower == "delivery speed":
        return {
            "canonical_parent": "Delivery",
            "canonical_aspect": "Delivery Speed",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    if lower in {
        "price",
        "pricing",
        "price/value",
    }:
        return {
            "canonical_parent": "Price & Value",
            "canonical_aspect": "Pricing",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    # --------------------------------------------------------
    # AMBIENCE
    # --------------------------------------------------------

    if lower in {
        "ambience",
        "atmosphere",
        "overall ambience",
    }:
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "Overall Ambience",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "music":
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "Music",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "noise level":
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "Noise Level",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "crowding":
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "Crowding",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "room",
        "small space",
        "space size",
    }:
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "Space",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "view":
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "View",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "historic/old building",
        "beautiful building",
        "old house",
        "charming",
    }:
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "Decor / Building Character",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower == "sports viewing":
        return {
            "canonical_parent": "Ambience",
            "canonical_aspect": "Sports Viewing",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    if lower in {
        "location",
        "location preference",
        "preferred location",
    }:
        return {
            "canonical_parent": "Location & Access",
            "canonical_aspect": "Location",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    # --------------------------------------------------------
    # CLEANLINESS
    # --------------------------------------------------------

    if lower == "minor maintenance issues":
        return {
            "canonical_parent": "Cleanliness & Maintenance",
            "canonical_aspect": "Maintenance",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    # --------------------------------------------------------
    # DINING EXPERIENCE
    # --------------------------------------------------------

    if lower in {
        "seating",
        "seating arrangement",
        "outdoor seating",
        "private dining",
    }:
        return {
            "canonical_parent": "Dining Experience",
            "canonical_aspect": "Seating",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    if lower in {
        "overall experience",
        "negative experience",
        "dining",
    }:
        return {
            "canonical_parent": "Dining Experience",
            "canonical_aspect": "Overall Experience",
            "aspect_type": "core_aspect",
            "aspect_value": "",
        }

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    if lower in {
        "birthday party",
        "events & group dining",
    }:
        return {
            "canonical_parent": "Events & Group Dining",
            "canonical_aspect": "Events & Group Dining",
            "aspect_type": "context_tag",
            "aspect_value": lower,
        }

    # --------------------------------------------------------
    # RESTAURANT TYPE
    # --------------------------------------------------------

    if lower in {
        "restaurant type",
        "casual dining",
        "franchise",
        "casino",
    }:
        return {
            "canonical_parent": "Restaurant Type",
            "canonical_aspect": "Restaurant Type",
            "aspect_type": "context_tag",
            "aspect_value": lower,
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {
        "canonical_parent": parent,
        "canonical_aspect": aspect,
        "aspect_type": "other",
        "aspect_value": "",
    }


# ============================================================
# LOAD DATA
# ============================================================

print("Loading final aspect mapping...")

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "mapping_id",
    "canonical_parent",
    "canonical_aspect",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print(
    f"Mappings loaded: {len(df):,}"
)


# ============================================================
# CLEAN INPUT COLUMNS
# ============================================================

df["canonical_parent"] = (
    df["canonical_parent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["canonical_aspect"] = (
    df["canonical_aspect"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["mapping_id"] = (
    df["mapping_id"]
    .astype(int)
)


# ============================================================
# APPLY NORMALIZATION
# ============================================================

print("\nCleaning taxonomy...")

normalized_rows = []

for _, row in df.iterrows():

    result = normalize_aspect(
        row["canonical_parent"],
        row["canonical_aspect"]
    )

    normalized_rows.append({
        "mapping_id": int(row["mapping_id"]),
        "source_parent": row["canonical_parent"],
        "source_aspect": row["canonical_aspect"],
        "canonical_parent": result["canonical_parent"],
        "canonical_aspect": result["canonical_aspect"],
        "aspect_type": result["aspect_type"],
        "aspect_value": result["aspect_value"],
    })


clean_df = pd.DataFrame(
    normalized_rows
)


# ============================================================
# SAVE CLEAN MAPPING
# ============================================================

clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    clean_df
    .groupby(
        [
            "canonical_parent",
            "canonical_aspect",
            "aspect_type"
        ]
    )
    .agg(
        source_count=(
            "mapping_id",
            "count"
        )
    )
    .reset_index()
    .sort_values(
        [
            "canonical_parent",
            "canonical_aspect"
        ]
    )
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("TAXONOMY CLEANING COMPLETE")
print("========================================")

print(
    f"Source mappings:      {len(df):,}"
)

print(
    f"Clean mappings:       {len(clean_df):,}"
)

print(
    f"Canonical parents:    "
    f"{clean_df['canonical_parent'].nunique():,}"
)

print(
    f"Canonical aspects:    "
    f"{clean_df['canonical_aspect'].nunique():,}"
)

print(
    f"Core aspects:         "
    f"{(clean_df['aspect_type'] == 'core_aspect').sum():,}"
)

print(
    f"Context tags:         "
    f"{(clean_df['aspect_type'] == 'context_tag').sum():,}"
)

print(
    f"Specific food items:  "
    f"{(clean_df['aspect_type'] == 'specific_food_item').sum():,}"
)


# ============================================================
# FINAL CORE HIERARCHY
# ============================================================

print("\n========================================")
print("FINAL CORE ASPECT HIERARCHY")
print("========================================")

core = clean_df[
    clean_df["aspect_type"] == "core_aspect"
]

for parent in sorted(
    core["canonical_parent"].unique()
):

    print(f"\n### {parent}")

    aspects = (
        core[
            core["canonical_parent"] == parent
        ]["canonical_aspect"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for aspect in aspects:
        print(
            f"  - {aspect}"
        )


# ============================================================
# CONTEXT TAGS
# ============================================================

print("\n========================================")
print("CONTEXT TAGS")
print("========================================")

context = clean_df[
    clean_df["aspect_type"] == "context_tag"
]

for tag in sorted(
    context["canonical_aspect"].unique()
):

    print(
        f"  - {tag}"
    )


# ============================================================
# SPECIFIC FOOD ITEMS
# ============================================================

print("\n========================================")
print("SPECIFIC FOOD ITEMS")
print("========================================")

food_items = clean_df[
    clean_df["aspect_type"]
    == "specific_food_item"
]

for item in sorted(
    food_items["aspect_value"]
    .dropna()
    .unique()
):

    if item:
        print(
            f"  - {item}"
        )


print(
    f"\nSaved:\n{OUTPUT_FILE}"
)

print(
    f"\nSummary:\n{SUMMARY_FILE}"
)