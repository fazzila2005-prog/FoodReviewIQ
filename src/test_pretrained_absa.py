from setfit import AbsaModel


# =========================================================
# Load pretrained restaurant ABSA model
# =========================================================

model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
    spacy_model="en_core_web_sm"
)


# =========================================================
# Test reviews
# =========================================================

reviews = [
    "The food was amazing but the service was terrible.",

    "The waiter was friendly and the food was delicious.",

    "We waited for an hour and the service was very slow.",

    "The restaurant was beautiful but the prices were too high.",

    "The pizza was excellent but the staff were rude."
]


# =========================================================
# Run ABSA
# =========================================================

results = model.predict(reviews)


# =========================================================
# Display results
# =========================================================

for i, (review, result) in enumerate(
    zip(reviews, results),
    start=1
):

    print()
    print("=" * 70)
    print(f"REVIEW {i}")
    print("=" * 70)

    print("TEXT:")
    print(review)

    print()
    print("ABSA RESULT:")

    for prediction in result:
        print(
            f"Aspect: {prediction['span']} "
            f"| Sentiment: {prediction['polarity']}"
        )