import csv

FILE = r"C:\FoodReviewIQ\data\processed\processed_reviews.csv"

with open(FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for i, row in enumerate(reader):

        print("\n" + "=" * 70)
        print(f"REVIEW {i + 1}")
        print("=" * 70)

        print("ORIGINAL:")
        print(row["text"])

        print("\nCLEANED:")
        print(row["cleaned_text"])

        print("\nWITHOUT STOPWORDS:")
        print(row["without_stopwords"])

        print("\nSTEMMED:")
        print(row["stemmed_text"])

        print("\nLEMMATIZED:")
        print(row["lemmatized_text"])

        print("\nPOS TAGS:")
        print(row["pos_tags"])

        if i == 2:
            break