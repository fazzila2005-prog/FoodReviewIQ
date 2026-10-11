import csv
from pathlib import Path


INPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\evaluation_reviews.csv"
)

OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\absa_gold.csv"
)


# Maximum aspects we allow per review.
# Most reviews will need only 1–3.
MAX_ASPECTS = 5


input_fields = [
    "review_id",
    "business_id",
    "restaurant_name",
    "city",
    "state",
    "review_stars",
    "text"
]


# Create columns for manual annotation
output_fields = input_fields.copy()

for i in range(1, MAX_ASPECTS + 1):

    output_fields.extend([
        f"aspect_{i}_span",
        f"aspect_{i}_category",
        f"aspect_{i}_sentiment"
    ])


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as infile:

    reader = csv.DictReader(infile)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=output_fields
        )

        writer.writeheader()

        for row in reader:

            new_row = {
                field: row[field]
                for field in input_fields
            }

            # Empty annotation fields
            for i in range(1, MAX_ASPECTS + 1):

                new_row[f"aspect_{i}_span"] = ""
                new_row[f"aspect_{i}_category"] = ""
                new_row[f"aspect_{i}_sentiment"] = ""

            writer.writerow(new_row)


print("Annotation template created.")
print(f"Saved to: {OUTPUT_FILE}")