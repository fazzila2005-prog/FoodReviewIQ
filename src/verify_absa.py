import csv
import random
from pathlib import Path


# =========================================================
# FILE
# =========================================================

INPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\absa_auto_labels.csv"
)


# =========================================================
# SETTINGS
# =========================================================

# 4 reviews from each star rating = 20 reviews total
REVIEWS_PER_STAR = 4

# Keeps the same 20 reviews every time we run the script
RANDOM_SEED = 42


# =========================================================
# LOAD REVIEWS
# =========================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    all_reviews = list(reader)


# =========================================================
# GROUP REVIEWS BY STAR RATING
# =========================================================

by_star = {
    "1.0": [],
    "2.0": [],
    "3.0": [],
    "4.0": [],
    "5.0": []
}


for review in all_reviews:

    star = review["review_stars"]

    if star in by_star:
        by_star[star].append(review)


# =========================================================
# SELECT 4 FROM EACH STAR RATING
# =========================================================

random.seed(RANDOM_SEED)

selected_reviews = []


for star in [
    "1.0",
    "2.0",
    "3.0",
    "4.0",
    "5.0"
]:

    candidates = by_star[star].copy()

    random.shuffle(candidates)

    selected_reviews.extend(
        candidates[:REVIEWS_PER_STAR]
    )


# =========================================================
# PRINT ALL 20 REVIEWS
# =========================================================

print()
print("=" * 100)
print("FOODREVIEWIQ - 20 REVIEW ABSA VERIFICATION")
print("=" * 100)

print()
print(
    "There are 20 reviews below: "
    "4 from each star rating."
)

print()
print(
    "COPY THE ENTIRE OUTPUT AND PASTE IT INTO CHAT."
)

print()
print(
    "I will review the model annotations and give you "
    "the corrected annotations."
)

print()
print("=" * 100)


# =========================================================
# DISPLAY REVIEWS
# =========================================================

for number, review in enumerate(
    selected_reviews,
    start=1
):

    print()
    print()
    print("#" * 100)

    print(
        f"REVIEW {number}/20"
    )

    print("#" * 100)

    print()

    print(
        f"Restaurant: {review['restaurant_name']}"
    )

    print(
        f"Rating: {review['review_stars']} stars"
    )

    print(
        f"Location: {review['city']}, {review['state']}"
    )

    print()

    print("REVIEW TEXT")
    print("-" * 100)

    print(
        review["text"]
    )

    print()

    print("MODEL ANNOTATIONS")
    print("-" * 100)

    annotations = review["annotations"].strip()

    if annotations:

        print(
            annotations
        )

    else:

        print(
            "NONE DETECTED"
        )

    print()
    print("#" * 100)


# =========================================================
# END
# =========================================================

print()
print()
print("=" * 100)
print("END OF 20-REVIEW BATCH")
print("=" * 100)