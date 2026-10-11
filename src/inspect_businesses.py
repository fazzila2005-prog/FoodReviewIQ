import json
import csv
from pathlib import Path


# Yelp business dataset
BUSINESS_FILE = Path(
    r"C:\FoodReviewIQ\data\raw\Yelp-JSON\Yelp JSON\yelp_academic_dataset_business.json"
)

# Our smaller output file
OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\restaurant_businesses.csv"
)


restaurant_count = 0


# Open the Yelp business file
with open(BUSINESS_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    fieldnames = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "categories"
    ]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    # Read one business at a time
    for line in infile:
        business = json.loads(line)

        categories = business.get("categories") or ""

        # Keep businesses that are restaurants
        if "Restaurants" in categories:
            writer.writerow({
                "business_id": business.get("business_id"),
                "name": business.get("name"),
                "city": business.get("city"),
                "state": business.get("state"),
                "stars": business.get("stars"),
                "review_count": business.get("review_count"),
                "categories": categories
            })

            restaurant_count += 1


print(f"Restaurant businesses found: {restaurant_count}")
print(f"Saved to: {OUTPUT_FILE}")