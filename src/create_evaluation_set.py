import csv
import random
from pathlib import Path
from collections import defaultdict


# =========================================================
# FILES
# =========================================================

INPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\restaurant_reviews.csv"
)

OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\evaluation_reviews.csv"
)


# =========================================================
# SETTINGS
# =========================================================

# 20 reviews from each rating
REVIEWS_PER_STAR = 20

# Fixed seed = reproducible sampling
RANDOM_SEED = 42


# =========================================================
# LOAD REVIEWS
# =========================================================

reviews_by_star = defaultdict(list)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        star = row["review_stars"]

        if star in {"1.0", "2.0", "3.0", "4.0", "5.0"}:
            reviews_by_star[star].append(row)


# =========================================================
# SELECT REVIEWS
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

    available = reviews_by_star[star]

    # Shuffle deterministically
    random.shuffle(available)

    # Try to select different restaurants
    selected_for_star = []
    used_businesses = set()

    for review in available:

        business_id = review["business_id"]

        if business_id not in used_businesses:

            selected_for_star.append(review)
            used_businesses.add(business_id)

        if len(selected_for_star) >= REVIEWS_PER_STAR:
            break

    selected_reviews.extend(selected_for_star)


# =========================================================
# SAVE EVALUATION SET
# =========================================================

fields = [
    "review_id",
    "business_id",
    "restaurant_name",
    "city",
    "state",
    "review_stars",
    "date",
    "text"
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fields
    )

    writer.writeheader()

    for review in selected_reviews:

        writer.writerow({
            field: review[field]
            for field in fields
        })


# =========================================================
# RESULT
# =========================================================

print("=" * 60)
print("EVALUATION SET CREATED")
print("=" * 60)

print(
    f"Total reviews: {len(selected_reviews)}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print()

for star in [
    "1.0",
    "2.0",
    "3.0",
    "4.0",
    "5.0"
]:

    count = sum(
        1
        for review in selected_reviews
        if review["review_stars"] == star
    )

    print(
        f"{star} stars: {count}"
    )