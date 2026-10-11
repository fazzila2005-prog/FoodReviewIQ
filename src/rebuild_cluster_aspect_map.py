import pandas as pd


# ============================================================
# FILES
# ============================================================

CLUSTERS_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_context_clusters.csv"
)

LABELS_FILE = (
    r"C:\FoodReviewIQ\data\processed\aspect_cluster_labels.csv"
)

OUTPUT_FILE = (
    r"C:\FoodReviewIQ\data\processed\direct_cluster_aspect_mapping.csv"
)


# ============================================================
# SIMPLE CANONICALIZATION
# ============================================================

def canonicalize(parent, aspect):

    parent = str(parent).strip()
    aspect = str(aspect).strip()

    p = parent.lower()
    a = aspect.lower()

    # Flatten accidental nested parents
    if p.startswith("food /"):
        p = "food"

    if p.startswith("service /"):
        p = "service"

    # -------------------------
    # FOOD
    # -------------------------

    if a in {
        "food",
        "food quality",
        "food_quality_disappointment",
    }:
        return (
            "Food",
            "Food Quality",
            "core_aspect",
            ""
        )

    if a in {
        "taste",
        "taste & flavor",
        "food flavor",
        "flavor",
    }:
        return (
            "Food",
            "Taste & Flavor",
            "core_aspect",
            ""
        )

    if a in {
        "portion",
        "portion size",
        "nice sized portion",
    }:
        return (
            "Food",
            "Portion Size",
            "core_aspect",
            ""
        )

    if a in {
        "freshness",
        "ingredient freshness",
        "ingredient use",
    }:
        return (
            "Food",
            "Ingredient Quality / Freshness",
            "core_aspect",
            ""
        )

    if a == "temperature":
        return (
            "Food",
            "Food Temperature",
            "core_aspect",
            ""
        )

    if a in {
        "side",
        "side dish",
        "side dishes",
    }:
        return (
            "Food",
            "Side Dishes",
            "core_aspect",
            ""
        )

    # Specific food items
    specific_food = {
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
    }

    if a in specific_food:
        return (
            "Food",
            "Specific Food Item",
            "specific_food_item",
            a
        )

    # -------------------------
    # MENU
    # -------------------------

    if a in {
        "menu",
        "menu / drink menu",
    }:
        return (
            "Menu",
            "Menu",
            "core_aspect",
            ""
        )

    if a == "menu variety":
        return (
            "Menu",
            "Menu Variety",
            "core_aspect",
            ""
        )

    # -------------------------
    # BEVERAGES
    # -------------------------

    beverage_items = {
        "beer",
        "cocktails",
        "milkshake",
        "lassi",
        "sangria",
        "vodka",
        "beverages",
        "specific beverage",
        "specific beverage item",
        "beverages / specific beverage",
    }

    if a in beverage_items:
        return (
            "Beverages",
            "Specific Beverage",
            "specific_beverage",
            a
        )

    # -------------------------
    # SERVICE
    # -------------------------

    if a in {
        "service",
        "service issues",
        "service quality",
        "service performance",
        "overall service experience",
        "issues",
    }:
        return (
            "Service",
            "Service Quality",
            "core_aspect",
            ""
        )

    if a in {
        "service staff",
        "service_staff",
        "staff",
        "employee",
        "busser",
    }:
        return (
            "Service",
            "Service Staff",
            "core_aspect",
            ""
        )

    if a in {
        "staff attitude",
        "staff friendliness",
        "friendly staff",
        "eye contact",
        "warm welcome",
    }:
        return (
            "Service",
            "Staff Attitude",
            "core_aspect",
            ""
        )

    if a in {
        "staff availability",
        "staff absence",
    }:
        return (
            "Service",
            "Staff Availability",
            "core_aspect",
            ""
        )

    if a in {
        "waiting",
        "waiting time",
        "service timing",
        "timely service",
        "food delivery time",
    }:
        return (
            "Service",
            "Waiting Time",
            "core_aspect",
            ""
        )

    if a == "order process":
        return (
            "Service",
            "Order Process",
            "core_aspect",
            ""
        )

    if a in {
        "tableware",
        "silverware",
    }:
        return (
            "Service",
            "Tableware",
            "core_aspect",
            ""
        )

    # -------------------------
    # DELIVERY
    # -------------------------

    if a == "delivery speed":
        return (
            "Delivery",
            "Delivery Speed",
            "core_aspect",
            ""
        )

    # -------------------------
    # PRICE
    # -------------------------

    if a in {
        "price",
        "pricing",
        "price/value",
    }:
        return (
            "Price & Value",
            "Pricing",
            "core_aspect",
            ""
        )

    # -------------------------
    # AMBIENCE
    # -------------------------

    if a in {
        "ambience",
        "atmosphere",
        "overall ambience",
    }:
        return (
            "Ambience",
            "Overall Ambience",
            "core_aspect",
            ""
        )

    if a == "music":
        return (
            "Ambience",
            "Music",
            "core_aspect",
            ""
        )

    if a == "noise level":
        return (
            "Ambience",
            "Noise Level",
            "core_aspect",
            ""
        )

    if a == "crowding":
        return (
            "Ambience",
            "Crowding",
            "core_aspect",
            ""
        )

    if a in {
        "room",
        "small space",
        "space size",
    }:
        return (
            "Ambience",
            "Space",
            "core_aspect",
            ""
        )

    if a == "view":
        return (
            "Ambience",
            "View",
            "core_aspect",
            ""
        )

    if a == "sports viewing":
        return (
            "Ambience",
            "Sports Viewing",
            "core_aspect",
            ""
        )

    # -------------------------
    # LOCATION
    # -------------------------

    if a in {
        "location",
        "location preference",
        "preferred location",
    }:
        return (
            "Location & Access",
            "Location",
            "core_aspect",
            ""
        )

    # -------------------------
    # DINING
    # -------------------------

    if a in {
        "seating",
        "seating arrangement",
        "outdoor seating",
        "private dining",
    }:
        return (
            "Dining Experience",
            "Seating",
            "core_aspect",
            ""
        )

    if a in {
        "overall experience",
        "negative experience",
        "dining",
    }:
        return (
            "Dining Experience",
            "Overall Experience",
            "core_aspect",
            ""
        )

    # -------------------------
    # CONTEXT
    # -------------------------

    context_tags = {
        "breakfast",
        "lunch",
        "brunch",
        "birthday party",
        "casino",
        "casual dining",
        "franchise",
        "indian food",
        "thai cuisine",
        "vegetarian food",
        "takeout",
        "events & group dining",
        "group size",
    }

    if a in context_tags:
        return (
            "Context",
            a.title(),
            "context_tag",
            a
        )

    # -------------------------
    # FALLBACK
    # -------------------------

    return (
        parent,
        aspect,
        "other",
        ""
    )


# ============================================================
# LOAD
# ============================================================

print("Loading cluster data...")

clusters = pd.read_csv(
    CLUSTERS_FILE,
    usecols=[
        "aspect_candidate",
        "cluster_id"
    ]
)

clusters["aspect_candidate"] = (
    clusters["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

clusters["cluster_id"] = (
    clusters["cluster_id"]
    .astype(int)
)


print(
    f"Candidate mappings: "
    f"{len(clusters):,}"
)


print("\nLoading Groq cluster labels...")

labels = pd.read_csv(
    LABELS_FILE
)

labels["cluster_id"] = (
    labels["cluster_id"]
    .astype(int)
)

labels["keep"] = (
    labels["keep"]
    .astype(str)
    .str.lower()
    .eq("true")
)


print(
    f"Cluster labels: "
    f"{len(labels):,}"
)


# ============================================================
# KEEP USEFUL CLUSTERS
# ============================================================

labels = labels[
    labels["keep"]
].copy()


# ============================================================
# JOIN DIRECTLY BY CLUSTER ID
# ============================================================

mapping = clusters.merge(
    labels[
        [
            "cluster_id",
            "parent_aspect",
            "normalized_aspect"
        ]
    ],
    on="cluster_id",
    how="left"
)


# Keep only clusters that Groq marked useful.
mapping = mapping.dropna(
    subset=[
        "normalized_aspect"
    ]
)


# ============================================================
# CANONICALIZE
# ============================================================

print(
    "\nCanonicalizing labels..."
)

canonical_rows = []

for _, row in mapping.iterrows():

    (
        parent,
        aspect,
        aspect_type,
        aspect_value
    ) = canonicalize(
        row["parent_aspect"],
        row["normalized_aspect"]
    )

    canonical_rows.append({
        "aspect_candidate":
            row["aspect_candidate"],

        "cluster_id":
            int(row["cluster_id"]),

        "source_parent":
            row["parent_aspect"],

        "source_aspect":
            row["normalized_aspect"],

        "canonical_parent":
            parent,

        "canonical_aspect":
            aspect,

        "aspect_type":
            aspect_type,

        "aspect_value":
            aspect_value,
    })


result = pd.DataFrame(
    canonical_rows
)


# ============================================================
# REMOVE DUPLICATE CANDIDATE MAPPINGS
# ============================================================

result = result.drop_duplicates(
    subset=[
        "aspect_candidate"
    ]
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n========================================")
print("DIRECT ASPECT MAPPING COMPLETE")
print("========================================")

print(
    f"Candidate mappings: "
    f"{len(result):,}"
)

print(
    f"Unique clusters: "
    f"{result['cluster_id'].nunique():,}"
)

print(
    f"Unique canonical aspects: "
    f"{result['canonical_aspect'].nunique():,}"
)

print("\nCanonical aspect distribution:\n")

print(
    result[
        "canonical_aspect"
    ]
    .value_counts()
    .to_string()
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)