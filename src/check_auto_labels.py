import csv

FILE = r"C:\FoodReviewIQ\data\processed\absa_auto_labels.csv"


with open(FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    shown = 0

    for row in reader:

        print()
        print("=" * 80)

        print("REVIEW:")
        print(row["text"])

        print()
        print("AUTOMATIC ANNOTATION:")

        if row["annotations"]:
            print(row["annotations"])
        else:
            print("NONE DETECTED")

        shown += 1

        if shown >= 10:
            break