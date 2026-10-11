import csv
from pathlib import Path

from setfit import AbsaModel


# =========================================================
# 1. FILE LOCATIONS
# =========================================================

INPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\evaluation_reviews.csv"
)

OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\absa_auto_labels.csv"
)


# =========================================================
# 2. LOAD PRETRAINED RESTAURANT ABSA MODEL
# =========================================================

print("Loading pretrained ABSA model...")

model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
    spacy_model="en_core_web_sm"
)

print("Model loaded.")


# =========================================================
# 3. CATEGORY VOCABULARY
# =========================================================

CATEGORY_TERMS = {

    "food": {
        "food", "dish", "dishes", "meal", "meals",
        "pizza", "pasta", "burger", "burgers",
        "chicken", "beef", "steak", "fish",
        "sushi", "curry", "dessert", "desserts",
        "cake", "bread", "rice", "soup", "salad",
        "drink", "drinks", "coffee", "tea",
        "beer", "wine", "sandwich", "naan",
        "buffet", "fries", "burger", "sauce",
        "cheese", "meat"
    },

    "service": {
        "service", "staff", "waiter", "waitress",
        "waitstaff", "server", "bartender",
        "host", "hostess", "cashier",
        "manager", "worker", "workers",
        "delivery", "customer service"
    },

    "ambience": {
        "ambience", "ambiance", "atmosphere",
        "decor", "decoration", "vibe",
        "environment", "interior", "music",
        "seating", "noise", "patio",
        "rooftop", "restaurant"
    },

    "price": {
        "price", "prices", "cost", "expensive",
        "cheap", "value", "worth", "bill",
        "pricing", "money"
    },

    "waiting_time": {
        "wait", "waiting", "waited",
        "queue", "line", "delay", "delayed",
        "wait time", "delivery time"
    },

    "cleanliness": {
        "clean", "cleanliness", "dirty",
        "dirt", "hygiene", "bathroom",
        "restroom", "toilet"
    },

    "location": {
        "location", "parking", "neighborhood",
        "neighbourhood", "downtown", "area"
    },

    "portion": {
        "portion", "portions", "serving",
        "servings", "quantity", "size"
    }
}


# =========================================================
# 4. MAP MODEL ASPECT TO OUR PROJECT CATEGORY
# =========================================================

def map_category(aspect_text):

    aspect_lower = aspect_text.lower().strip()

    # Exact / phrase matching first
    for category, terms in CATEGORY_TERMS.items():

        if aspect_lower in terms:
            return category

    # Partial phrase matching
    for category, terms in CATEGORY_TERMS.items():

        for term in terms:

            if term in aspect_lower:
                return category

    return "other"


# =========================================================
# 5. LOAD EVALUATION REVIEWS
# =========================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    reviews = list(reader)


print(
    f"Reviews to annotate: {len(reviews)}"
)


# =========================================================
# 6. RUN ABSA MODEL
# =========================================================

texts = [
    review["text"]
    for review in reviews
]

print()
print("Running ABSA predictions...")
print("This can take a little while.")

predictions = model.predict(texts)

print("Predictions complete.")


# =========================================================
# 7. CREATE OUTPUT
# =========================================================

output_fields = [
    "review_id",
    "business_id",
    "restaurant_name",
    "city",
    "state",
    "review_stars",
    "text",
    "annotations"
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=output_fields
    )

    writer.writeheader()


    # =====================================================
    # 8. PROCESS EACH REVIEW
    # =====================================================

    for review, result in zip(
        reviews,
        predictions
    ):

        annotations = []

        seen = set()


        # -------------------------------------------------
        # Model predictions for this review
        # -------------------------------------------------

        for prediction in result:

            aspect = prediction.get(
                "span",
                ""
            ).strip()

            sentiment = prediction.get(
                "polarity",
                ""
            ).strip().lower()


            # Ignore incomplete predictions
            if not aspect or not sentiment:
                continue


            category = map_category(
                aspect
            )


            # Avoid duplicate annotations
            key = (
                aspect.lower(),
                category,
                sentiment
            )


            if key in seen:
                continue


            seen.add(key)


            annotations.append(
                f"{aspect}|"
                f"{category}|"
                f"{sentiment}"
            )


        # -------------------------------------------------
        # Save annotation string
        # -------------------------------------------------

        annotation_text = ";".join(
            annotations
        )


        writer.writerow({
            "review_id": review["review_id"],
            "business_id": review["business_id"],
            "restaurant_name": review["restaurant_name"],
            "city": review["city"],
            "state": review["state"],
            "review_stars": review["review_stars"],
            "text": review["text"],
            "annotations": annotation_text
        })


# =========================================================
# 9. FINISHED
# =========================================================

print()
print("=" * 70)
print("AUTOMATIC ANNOTATION COMPLETE")
print("=" * 70)

print(
    f"Reviews annotated: {len(reviews)}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)