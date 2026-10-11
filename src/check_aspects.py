import csv


FILE = r"C:\FoodReviewIQ\data\processed\aspect_sentiment_reviews.csv"


with open(
    FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    shown = 0

    for row in reader:

        if row["detected_aspects"]:

            print()
            print("=" * 70)

            print("ORIGINAL REVIEW:")
            print(row["text"])

            print()
            print("ASPECTS:")
            print(row["detected_aspects"])

            print()
            print("ASPECT SENTIMENT:")
            print(row["aspect_sentiments"])

            print()
            print("EVIDENCE:")
            print(row["aspect_evidence"])

            shown += 1

        if shown >= 5:
            break