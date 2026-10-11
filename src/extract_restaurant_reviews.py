import json
import csv
from pathlib import Path


# ---------------------------------------------------
# File locations
# ---------------------------------------------------

BUSINESS_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\restaurant_businesses.csv"
)

REVIEW_FILE = Path(
    r"C:\FoodReviewIQ\data\raw\Yelp-JSON\Yelp JSON\yelp_academic_dataset_review.json"
)

OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\restaurant_reviews.csv"
)

# Number of reviews we want for our first RAG dataset
MAX_REVIEWS = 50_000


# ---------------------------------------------------
# Step 1: Load restaurant business information
# ---------------------------------------------------

restaurant_businesses = {}

with open(BUSINESS_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        restaurant_businesses[row["business_id"]] = {
            "restaurant_name": row["name"],
            "city": row["city"],
            "state": row["state"],
            "business_stars": row["stars"],
            "business_review_count": row["review_count"],
            "categories": row["categories"]
        }


print(f"Restaurant businesses loaded: {len(restaurant_businesses)}")


# ---------------------------------------------------
# Step 2: Read Yelp reviews ONE LINE AT A TIME
# ---------------------------------------------------

review_count = 0
total_reviews_checked = 0

with open(REVIEW_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    fieldnames = [
        "review_id",
        "business_id",
        "restaurant_name",
        "city",
        "state",
        "business_stars",
        "review_stars",
        "date",
        "text",
        "useful",
        "funny",
        "cool",
        "categories"
    ]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for line in infile:

        total_reviews_checked += 1

        review = json.loads(line)

        business_id = review.get("business_id")

        # Keep only reviews belonging to restaurants
        if business_id in restaurant_businesses:

            business = restaurant_businesses[business_id]

            writer.writerow({
                "review_id": review.get("review_id"),
                "business_id": business_id,
                "restaurant_name": business["restaurant_name"],
                "city": business["city"],
                "state": business["state"],
                "business_stars": business["business_stars"],
                "review_stars": review.get("stars"),
                "date": review.get("date"),
                "text": review.get("text"),
                "useful": review.get("useful"),
                "funny": review.get("funny"),
                "cool": review.get("cool"),
                "categories": business["categories"]
            })

            review_count += 1

            # Show progress every 100,000 reviews checked
            if total_reviews_checked % 100_000 == 0:
                print(
                    f"Checked: {total_reviews_checked:,} | "
                    f"Restaurant reviews found: {review_count:,}"
                )

            # Stop once we have enough reviews
            if review_count >= MAX_REVIEWS:
                break


print()
print(f"Reviews checked: {total_reviews_checked:,}")
print(f"Restaurant reviews saved: {review_count:,}")
print(f"Saved to: {OUTPUT_FILE}")