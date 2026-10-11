import pandas as pd


INPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\direct_cluster_aspect_mapping.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\final_cluster_aspect_mapping.csv"
)


df = pd.read_csv(INPUT_FILE)

df["canonical_aspect"] = (
    df["canonical_aspect"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["canonical_parent"] = (
    df["canonical_parent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["aspect_type"] = (
    df["aspect_type"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["aspect_value"] = (
    df["aspect_value"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# NORMALIZE REMAINING LABELS
# ============================================================

ASPECT_REMAP = {

    # ---------- FOOD ----------
    "food items": "Specific Food Item",
    "meal": "Food Quality",
    "food combination": "Food Quality",

    "wing seasoning": "Taste & Flavor",
    "spicy broth": "Taste & Flavor",
    "ketchup": "Taste & Flavor",
    "sugar": "Taste & Flavor",

    "crust": "Food Quality",
    "frozen_section_food": "Food Quality",
    "homemade ice cream": "Specific Food Item",
    "crispy lettuce": "Food Quality",
    "crisp lettuce": "Food Quality",

    # ---------- SERVICE ----------
    "friendly_staff": "Staff Attitude",
    "staff_absence": "Staff Availability",
    "timely_service": "Waiting Time",
    "grab_and_go": "Takeout",
    "bar service": "Service Quality",

    "negative_experience": "Overall Experience",

    # ---------- AMBIENCE ----------
    "outdoor patio": "Seating",
    "private dining room": "Seating",
    "long table": "Seating",

    "crowdedness": "Crowding",
    "beautiful building": "Decor / Building Character",
    "old house": "Decor / Building Character",
    "charming": "Overall Ambience",

    # ---------- MENU ----------
    "menu_item": "Specific Food Item",
    "drink menu": "Menu",

    # ---------- OTHER ----------
    "separate plate": "Plating / Presentation",
    "separate container": "Food Packaging",
    "minor maintenance issues": "Maintenance",
}


df["canonical_aspect"] = (
    df["canonical_aspect"]
    .replace(ASPECT_REMAP)
)


# ============================================================
# NORMALIZE PARENT CATEGORIES
# ============================================================

PARENT_REMAP = {
    "food": "Food",
    "Food": "Food",

    "service": "Service",
    "Service": "Service",

    "ambience": "Ambience",
    "Ambience": "Ambience",

    "location": "Location & Access",
    "Location & Access": "Location & Access",

    "menu": "Menu",
    "Menu": "Menu",

    "price": "Price & Value",
    "Price & Value": "Price & Value",

    "delivery": "Delivery",
    "Delivery": "Delivery",

    "dining": "Dining Experience",
    "Dining Experience": "Dining Experience",

    "food items": "Food",

    "bar service": "Service",
}

df["canonical_parent"] = (
    df["canonical_parent"]
    .replace(PARENT_REMAP)
)


# ============================================================
# FIX SPECIFIC FOOD ITEMS
# ============================================================

specific_items = {
    "pizza",
    "sushi",
    "sushi rolls",
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
    "naan",
    "souvlaki",
    "roast duck",
    "lobster risotto",
    "oxtail",
    "fried chicken livers",
    "house chips",
    "greasy burger",
    "spanish fries",
    "hush puppy, pork rind",
    "specialty roll",
    "egg tart",
}


for item in specific_items:

    mask = (
        df["canonical_aspect"]
        .str.lower()
        == item.lower()
    )

    df.loc[
        mask,
        "canonical_parent"
    ] = "Food"

    df.loc[
        mask,
        "aspect_type"
    ] = "specific_food_item"

    df.loc[
        mask,
        "aspect_value"
    ] = item


# ============================================================
# REMOVE DUPLICATE CANDIDATE ASSIGNMENTS
# ============================================================

df = df.drop_duplicates(
    subset=["aspect_candidate"],
    keep="first"
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

print("\n========================================")
print("FINAL DIRECT MAPPING COMPLETE")
print("========================================")

print(
    f"Candidate mappings: "
    f"{len(df):,}"
)

print(
    f"Unique clusters: "
    f"{df['cluster_id'].nunique():,}"
)

print(
    f"Unique canonical aspects: "
    f"{df['canonical_aspect'].nunique():,}"
)

print("\nCanonical aspect distribution:\n")

print(
    df[
        "canonical_aspect"
    ]
    .value_counts()
    .to_string()
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)