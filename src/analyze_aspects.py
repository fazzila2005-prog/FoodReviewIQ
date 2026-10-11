import csv
import re
from pathlib import Path

import nltk
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer


# =========================================================
# 1. NLTK DATA LOCATION
# =========================================================

NLTK_DATA_DIR = r"C:\FoodReviewIQ\nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)


# =========================================================
# 2. FILE LOCATIONS
# =========================================================

INPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\processed_reviews.csv"
)

OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\aspect_sentiment_reviews.csv"
)


# =========================================================
# 3. TEST MODE
# =========================================================

MAX_ROWS = 100


# =========================================================
# 4. SENTIMENT ANALYZER
# =========================================================

sia = SentimentIntensityAnalyzer()


# =========================================================
# 5. RESTAURANT ASPECT LEXICON
# =========================================================

ASPECT_TERMS = {

    "food": {
        "food",
        "dish",
        "dishes",
        "meal",
        "meals",
        "menu",
        "taste",
        "flavor",
        "flavour",
        "pizza",
        "pasta",
        "burger",
        "chicken",
        "beef",
        "steak",
        "fish",
        "sushi",
        "curry",
        "dessert",
        "desserts",
        "cake",
        "bread",
        "rice",
        "soup",
        "salad",
        "drink",
        "drinks",
        "coffee",
        "tea",
        "beer",
        "wine",
        "sandwich",
        "naan",
        "buffet"
    },

    "service": {
        "service",
        "staff",
        "waiter",
        "waitress",
        "waitstaff",
        "server",
        "bartender",
        "host",
        "hostess",
        "cashier",
        "manager"
    },

    "ambience": {
        "ambience",
        "ambiance",
        "atmosphere",
        "decor",
        "decoration",
        "vibe",
        "environment",
        "interior",
        "music",
        "seating",
        "noise",
        "patio",
        "rooftop"
    },

    "price": {
        "price",
        "prices",
        "cost",
        "expensive",
        "cheap",
        "value",
        "worth",
        "bill",
        "pricing",
        "money"
    },

    "waiting_time": {
        "wait",
        "waiting",
        "waited",
        "queue",
        "line",
        "delay",
        "delayed"
    },

    "cleanliness": {
        "clean",
        "cleanliness",
        "dirty",
        "dirt",
        "hygiene",
        "bathroom",
        "restroom",
        "toilet"
    },

    "location": {
        "location",
        "parking",
        "neighborhood",
        "neighbourhood",
        "downtown"
    },

    "portion": {
        "portion",
        "portions",
        "serving",
        "servings",
        "quantity",
        "size"
    }
}


# =========================================================
# 6. RESTAURANT-SPECIFIC SENTIMENT WORDS
# =========================================================

POSITIVE_WORDS = {
    "good",
    "great",
    "amazing",
    "excellent",
    "delicious",
    "fantastic",
    "wonderful",
    "awesome",
    "tasty",
    "fresh",
    "friendly",
    "pleasant",
    "attentive",
    "perfect",
    "love",
    "loved",
    "favorite",
    "favourite",
    "clean",
    "generous"
}


NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "horrible",
    "awful",
    "slow",
    "rude",
    "dirty",
    "poor",
    "worst",
    "disappointing",
    "disappointed",
    "bland",
    "cold",
    "overpriced",
    "expensive",
    "unfriendly",
    "unpleasant",
    "fishy",
    "waited"
}


# =========================================================
# 7. NEGATIVE PHRASES
# =========================================================

NEGATIVE_PHRASES = {
    "too long",
    "very long",
    "long wait",
    "long waiting",
    "long time",
    "took forever",
    "never came",
    "couldn't eat",
    "not good",
    "not fresh",
    "not worth",
    "would not go back",
    "wouldn't go back",
    "never again"
}


# =========================================================
# 8. FIND ASPECTS
# =========================================================

def find_aspects(sentence):

    words = word_tokenize(sentence.lower())

    found_aspects = set()

    for aspect, terms in ASPECT_TERMS.items():

        for word in words:

            clean_word = re.sub(
                r"[^\w]",
                "",
                word
            )

            if clean_word in terms:
                found_aspects.add(aspect)
                break

    return list(found_aspects)


# =========================================================
# 9. SENTIMENT FUNCTION
# =========================================================

def get_sentiment(text):

    text_lower = text.lower()

    # -----------------------------------------------------
    # Check explicit negative phrases first
    # -----------------------------------------------------

    for phrase in NEGATIVE_PHRASES:

        if phrase in text_lower:
            return "negative", -0.8


    # -----------------------------------------------------
    # Check individual sentiment words
    # -----------------------------------------------------

    positive_count = 0
    negative_count = 0

    words = set(
        word_tokenize(text_lower)
    )

    for word in POSITIVE_WORDS:

        if word in words:
            positive_count += 1

    for word in NEGATIVE_WORDS:

        if word in words:
            negative_count += 1


    if negative_count > positive_count:
        return "negative", -0.6

    if positive_count > negative_count:
        return "positive", 0.6


    # -----------------------------------------------------
    # Fall back to VADER
    # -----------------------------------------------------

    scores = sia.polarity_scores(text)

    compound = scores["compound"]


    if compound >= 0.05:
        return "positive", compound

    if compound <= -0.05:
        return "negative", compound

    return "neutral", compound


# =========================================================
# 10. GET ASPECT CONTEXT
# =========================================================

def get_aspect_context(sentence, aspect):

    words = word_tokenize(sentence)

    lower_words = [
        word.lower()
        for word in words
    ]

    aspect_words = ASPECT_TERMS[aspect]

    positions = []

    for index, word in enumerate(lower_words):

        clean_word = re.sub(
            r"[^\w]",
            "",
            word
        )

        if clean_word in aspect_words:
            positions.append(index)


    if not positions:
        return sentence


    # Use the first occurrence
    position = positions[0]


    # Context around the aspect
    start = max(0, position - 6)
    end = min(len(words), position + 7)


    return " ".join(
        words[start:end]
    )


# =========================================================
# 11. ANALYZE ONE REVIEW
# =========================================================

def analyze_review(text):

    sentences = nltk.sent_tokenize(text)

    results = []

    for sentence in sentences:

        aspects = find_aspects(sentence)

        if not aspects:
            continue


        for aspect in aspects:

            context = get_aspect_context(
                sentence,
                aspect
            )


            sentiment, score = get_sentiment(
                context
            )


            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "score": round(score, 4),
                "context": context
            })


    return results


# =========================================================
# 12. MAIN PROCESSING
# =========================================================

processed_count = 0


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as infile:

    reader = csv.DictReader(infile)


    output_fields = reader.fieldnames + [
        "detected_aspects",
        "aspect_sentiments",
        "aspect_evidence"
    ]


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

            results = analyze_review(
                row["text"]
            )


            # -------------------------------------------------
            # Store one result per aspect
            # -------------------------------------------------

            aspect_results = {}


            for result in results:

                aspect = result["aspect"]

                # Keep the first result for each aspect
                if aspect not in aspect_results:

                    aspect_results[aspect] = result


            # -------------------------------------------------
            # Create output strings
            # -------------------------------------------------

            aspects = list(
                aspect_results.keys()
            )


            sentiment_results = []

            evidence = []


            for aspect in aspects:

                result = aspect_results[aspect]

                sentiment_results.append(
                    f"{aspect}: {result['sentiment']}"
                )


                evidence.append(
                    f"{aspect} → "
                    f"{result['sentiment']} "
                    f"→ {result['context']}"
                )


            row["detected_aspects"] = ", ".join(
                aspects
            )


            row["aspect_sentiments"] = " | ".join(
                sentiment_results
            )


            row["aspect_evidence"] = " || ".join(
                evidence
            )


            writer.writerow(row)


            processed_count += 1


            if processed_count % 10 == 0:

                print(
                    f"Processed {processed_count} reviews..."
                )


            if (
                MAX_ROWS is not None
                and processed_count >= MAX_ROWS
            ):
                break


print()
print("=" * 55)
print("ASPECT-BASED SENTIMENT ANALYSIS COMPLETE")
print("=" * 55)

print(
    f"Reviews processed: {processed_count}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)