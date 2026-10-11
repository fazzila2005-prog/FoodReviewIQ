import nltk
from pathlib import Path


# Keep NLTK's downloaded resources inside our project
NLTK_DATA_DIR = Path(r"C:\FoodReviewIQ\nltk_data")
NLTK_DATA_DIR.mkdir(exist_ok=True)


resources = [
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng",
    "vader_lexicon"
]


for resource in resources:
    print(f"Downloading: {resource}")
    nltk.download(resource, download_dir=str(NLTK_DATA_DIR))


print()
print("NLTK setup complete.")
print(f"Resources saved to: {NLTK_DATA_DIR}")