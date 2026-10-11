import csv
from pathlib import Path


# =========================================================
# FILES
# =========================================================

INPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\evaluation_reviews.csv"
)

OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\absa_gold.csv"
)


# =========================================================
# ALLOWED LABELS
# =========================================================

CATEGORIES = {
    "food",
    "service",
    "ambience",
    "price",
    "waiting_time",
    "cleanliness",
    "location",
    "portion",
    "other"
}

SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
    "conflict"
}


# =========================================================
# LOAD THE 100 EVALUATION REVIEWS
# =========================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    reviews = list(reader)


# =========================================================
# LOAD ONLY ACTUALLY COMPLETED REVIEWS
# =========================================================

completed_ids = set()

if OUTPUT_FILE.exists():

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            annotations = row.get(
                "annotations",
                ""
            ).strip()

            # Only treat the review as completed when
            # an actual annotation exists.
            if annotations not in {"", "SKIPPED"}:

                completed_ids.add(
                    row["review_id"]
                )


# =========================================================
# CREATE OUTPUT FILE IF IT DOES NOT EXIST
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


if not OUTPUT_FILE.exists():

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


# =========================================================
# INSTRUCTIONS
# =========================================================

print()
print("=" * 75)
print("FOODREVIEWIQ - GOLD ABSA ANNOTATION")
print("=" * 75)

print()
print("For each review, identify the aspects and sentiment.")

print()
print("FORMAT:")
print("aspect words|category|sentiment")

print()
print("MULTIPLE ASPECTS:")
print("food|food|positive;service|service|negative")

print()
print("NO CLEAR ASPECT:")
print("none")

print()
print("COMMANDS:")
print("q = quit and save")
print("s = skip this review")

print()
print("CATEGORIES:")
print(", ".join(sorted(CATEGORIES)))

print()
print("SENTIMENTS:")
print(", ".join(sorted(SENTIMENTS)))

print()
print("=" * 75)


# =========================================================
# ANNOTATION
# =========================================================

for number, review in enumerate(
    reviews,
    start=1
):

    review_id = review["review_id"]

    # Skip reviews already annotated
    if review_id in completed_ids:
        continue


    print()
    print()
    print("#" * 75)

    print(
        f"REVIEW {number} OF {len(reviews)}"
    )

    print("#" * 75)

    print()
    print(
        f"Restaurant : {review['restaurant_name']}"
    )

    print(
        f"Rating     : {review['review_stars']} stars"
    )

    print(
        f"Location   : "
        f"{review['city']}, {review['state']}"
    )

    print()
    print("REVIEW:")
    print("-" * 75)

    print(review["text"])

    print("-" * 75)


    # =====================================================
    # ASK FOR ANNOTATION
    # =====================================================

    while True:

        answer = input(
            "\nYour annotation: "
        ).strip()


        # -------------------------------------------------
        # QUIT
        # -------------------------------------------------

        if answer.lower() == "q":

            print()
            print(
                "Stopping. Your annotations have been saved."
            )

            print(
                f"File: {OUTPUT_FILE}"
            )

            raise SystemExit


        # -------------------------------------------------
        # SKIP
        # -------------------------------------------------

        if answer.lower() == "s":

            annotations = "SKIPPED"

            break


        # -------------------------------------------------
        # NO CLEAR ASPECT
        # -------------------------------------------------

        if answer.lower() == "none":

            annotations = "NONE"

            break


        # -------------------------------------------------
        # PARSE ANNOTATIONS
        # -------------------------------------------------

        parts = answer.split(";")

        parsed = []

        valid = True


        for part in parts:

            fields = [
                x.strip()
                for x in part.split("|")
            ]


            if len(fields) != 3:

                print()
                print(
                    "ERROR: Each aspect needs:"
                )

                print(
                    "aspect|category|sentiment"
                )

                print(
                    "Example:"
                )

                print(
                    "food|food|positive"
                )

                valid = False

                break


            span = fields[0].lower()
            category = fields[1].lower()
            sentiment = fields[2].lower()


            # Validate category
            if category not in CATEGORIES:

                print()
                print(
                    f"ERROR: Unknown category "
                    f"'{category}'"
                )

                print(
                    "Choose from:"
                )

                print(
                    ", ".join(
                        sorted(CATEGORIES)
                    )
                )

                valid = False

                break


            # Validate sentiment
            if sentiment not in SENTIMENTS:

                print()
                print(
                    f"ERROR: Unknown sentiment "
                    f"'{sentiment}'"
                )

                print(
                    "Choose from:"
                )

                print(
                    ", ".join(
                        sorted(SENTIMENTS)
                    )
                )

                valid = False

                break


            if not span:

                print(
                    "ERROR: Aspect words cannot be empty."
                )

                valid = False

                break


            parsed.append(
                f"{span}|"
                f"{category}|"
                f"{sentiment}"
            )


        # -------------------------------------------------
        # VALID ANNOTATION
        # -------------------------------------------------

        if valid:

            annotations = ";".join(parsed)

            break


    # =====================================================
    # SAVE IMMEDIATELY
    # =====================================================

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=output_fields
        )

        writer.writerow({
            "review_id": review["review_id"],
            "business_id": review["business_id"],
            "restaurant_name": review["restaurant_name"],
            "city": review["city"],
            "state": review["state"],
            "review_stars": review["review_stars"],
            "text": review["text"],
            "annotations": annotations
        })


    print()
    print("Saved ✅")


# =========================================================
# FINISHED
# =========================================================

print()
print("=" * 75)
print("ALL AVAILABLE REVIEWS ANNOTATED")
print("=" * 75)

print(
    f"Saved to: {OUTPUT_FILE}"
)