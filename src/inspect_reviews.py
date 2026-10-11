import csv
from collections import Counter


REVIEW_FILE = r"C:\FoodReviewIQ\data\processed\restaurant_reviews.csv"


total_reviews = 0
restaurant_ids = set()
star_counts = Counter()

text_lengths = []
missing_text = 0


with open(REVIEW_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total_reviews += 1

        # Unique restaurants
        restaurant_ids.add(row["business_id"])

        # Star distribution
        star_counts[row["review_stars"]] += 1

        # Review length
        text = row["text"].strip()

        if not text:
            missing_text += 1
        else:
            text_lengths.append(len(text))


average_length = (
    sum(text_lengths) / len(text_lengths)
    if text_lengths
    else 0
)


print("=" * 50)
print("FOODREVIEWIQ DATASET INSPECTION")
print("=" * 50)

print(f"Total reviews: {total_reviews:,}")
print(f"Unique restaurants: {len(restaurant_ids):,}")
print(f"Missing review text: {missing_text:,}")

print()
print("STAR DISTRIBUTION")
print("-" * 30)

for star in sorted(star_counts, key=float):
    print(f"{star} stars: {star_counts[star]:,}")

print()
print(f"Average review length: {average_length:.0f} characters")