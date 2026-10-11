from setfit import AbsaModel
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. LOAD PRETRAINED RESTAURANT ABSA MODEL
# =========================================================

model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
    spacy_model="en_core_web_sm"
)


# =========================================================
# 2. LOAD SEMEVAL RESTAURANT TEST DATA
# =========================================================

test_dataset = load_dataset(
    "tomaarsen/setfit-absa-semeval-restaurants",
    split="test"
)


print("=" * 60)
print("SEMEVAL RESTAURANT ABSA EVALUATION")
print("=" * 60)

print(f"Total test examples: {len(test_dataset)}")


# =========================================================
# 3. KEEP GOLD LABELS FROM THE ORIGINAL DATASET
# =========================================================

gold_labels = test_dataset["label"]


# =========================================================
# 4. RUN THE ABSA MODEL
# =========================================================

predictions = model.predict(test_dataset)


# =========================================================
# 5. COLLECT VALID PREDICTIONS
# =========================================================

true_labels = []
predicted_labels = []

skipped = 0


for index, row in enumerate(predictions):

    prediction = row["pred_polarity"]

    # Some aspect spans cannot be aligned by spaCy.
    if prediction is None:

        skipped += 1
        continue

    gold = gold_labels[index]

    # Make sure the gold label is a string
    gold = str(gold).lower().strip()

    prediction = str(prediction).lower().strip()

    true_labels.append(gold)
    predicted_labels.append(prediction)


# =========================================================
# 6. BASIC COUNTS
# =========================================================

evaluated = len(true_labels)


print()
print(f"Successfully evaluated: {evaluated}")
print(f"Skipped: {skipped}")


# =========================================================
# 7. SHOW A FEW EXAMPLES
# =========================================================

print()
print("FIRST 10 PREDICTIONS")
print("-" * 60)

for i in range(min(10, evaluated)):

    print(
        f"Gold: {true_labels[i]:10s} | "
        f"Predicted: {predicted_labels[i]}"
    )


# =========================================================
# 8. ACCURACY
# =========================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Polarity Accuracy: {accuracy:.4f}"
)

print(
    f"Polarity Accuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

print()
print("CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

print()
print("CONFUSION MATRIX")
print("-" * 60)

labels = sorted(
    set(true_labels) | set(predicted_labels)
)


matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=labels
)


print("Labels:")
print(labels)

print()
print(matrix)