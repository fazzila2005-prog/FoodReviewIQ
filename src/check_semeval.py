from datasets import load_dataset


# Load SemEval restaurant test set
dataset = load_dataset(
    "tomaarsen/setfit-absa-semeval-restaurants",
    split="test"
)


print("=" * 60)
print("DATASET FEATURES")
print("=" * 60)

print(dataset.features)


print()
print("=" * 60)
print("FIRST 10 ROWS")
print("=" * 60)

for i in range(10):

    row = dataset[i]

    print()
    print(f"ROW {i}")
    print("text     :", row["text"])
    print("span     :", row["span"])
    print("label    :", repr(row["label"]))
    print("type     :", type(row["label"]))
    print("ordinal  :", row["ordinal"])


print()
print("=" * 60)
print("UNIQUE LABEL VALUES")
print("=" * 60)

labels = dataset["label"]

print(set(labels))