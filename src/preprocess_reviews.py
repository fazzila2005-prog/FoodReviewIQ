import csv
import re
from pathlib import Path

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# =========================================================
# 1. NLTK DATA LOCATION
# =========================================================

NLTK_DATA_DIR = r"C:\FoodReviewIQ\nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)


# =========================================================
# 2. FILE LOCATIONS
# =========================================================

INPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\restaurant_reviews.csv"
)

OUTPUT_FILE = Path(
    r"C:\FoodReviewIQ\data\processed\processed_reviews.csv"
)


# =========================================================
# 3. TEST MODE
# =========================================================
# For now, process only 100 reviews.
#
# Later, after we verify everything is correct,
# change:
#
#     MAX_ROWS = 100
#
# to:
#
#     MAX_ROWS = None
#
# That will process all 50,000 reviews.

MAX_ROWS = None


# =========================================================
# 4. NLP TOOLS
# =========================================================

# Get NLTK's English stopwords
stop_words = set(stopwords.words("english"))


# Important words that we do NOT want to remove
# because they can change the meaning/sentiment.
NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "cannot",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "shouldn't",
    "can't",
    "haven't",
    "hasn't",
    "hadn't"
}

# Remove negation words from the stopword list
stop_words = stop_words - NEGATION_WORDS


# Porter Stemmer
stemmer = PorterStemmer()


# WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()


# =========================================================
# 5. EXPAND CONTRACTIONS
# =========================================================

def expand_contractions(text):
    """
    Convert contractions into full words.

    Example:
        don't  -> do not
        wasn't -> was not
        you'll -> you will
    """

    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "can not",
        "cannot": "can not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "mightn't": "might not",
        "mustn't": "must not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",

        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",

        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",

        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will"
    }

    # Lowercase first
    text = text.lower()

    # Replace contractions
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    return text


# =========================================================
# 6. CONVERT PENN TREEBANK POS TAGS
#    TO WORDNET POS TAGS
# =========================================================

def get_wordnet_pos(treebank_tag):
    """
    Convert Penn Treebank POS tags into the POS format
    required by WordNetLemmatizer.
    """

    if treebank_tag.startswith("J"):
        return "a"      # adjective

    if treebank_tag.startswith("V"):
        return "v"      # verb

    if treebank_tag.startswith("N"):
        return "n"      # noun

    if treebank_tag.startswith("R"):
        return "r"      # adverb

    return "n"          # default = noun


# =========================================================
# 7. CLEAN TEXT
# =========================================================

def clean_text(text):
    """
    Clean a review while keeping the meaning useful
    for sentiment analysis.
    """

    # Expand contractions first
    text = expand_contractions(text)

    # Remove punctuation and special symbols
    # Keep letters, numbers and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at beginning/end
    text = text.strip()

    return text


# =========================================================
# 8. PROCESS ONE REVIEW
# =========================================================

def process_review(text):

    # -----------------------------------------------------
    # Step 1: Cleaning
    # -----------------------------------------------------

    cleaned = clean_text(text)


    # -----------------------------------------------------
    # Step 2: Tokenization
    # -----------------------------------------------------

    tokens = word_tokenize(cleaned)


    # -----------------------------------------------------
    # Step 3: POS Tagging
    # -----------------------------------------------------

    tagged_tokens = pos_tag(tokens)


    # -----------------------------------------------------
    # Step 4: Stopword Removal
    # -----------------------------------------------------

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]


    # -----------------------------------------------------
    # Step 5: Porter Stemming
    # -----------------------------------------------------

    stemmed_tokens = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]


    # -----------------------------------------------------
    # Step 6: POS-Aware Lemmatization
    # -----------------------------------------------------

    lemmatized_tokens = []

    for word, tag in tagged_tokens:

        # Don't remove important negation words
        if word in stop_words:
            continue

        # Convert Treebank tag to WordNet tag
        wordnet_pos = get_wordnet_pos(tag)

        # Lemmatize
        lemma = lemmatizer.lemmatize(
            word,
            wordnet_pos
        )

        lemmatized_tokens.append(lemma)


    # -----------------------------------------------------
    # Step 7: Store POS tags
    # -----------------------------------------------------

    pos_text = " ".join(
        f"{word}/{tag}"
        for word, tag in tagged_tokens
    )


    # -----------------------------------------------------
    # Return all processed versions
    # -----------------------------------------------------

    return {
        "cleaned_text": cleaned,
        "tokens": " ".join(tokens),
        "without_stopwords": " ".join(filtered_tokens),
        "stemmed_text": " ".join(stemmed_tokens),
        "lemmatized_text": " ".join(lemmatized_tokens),
        "pos_tags": pos_text
    }


# =========================================================
# 9. MAIN PROGRAM
# =========================================================

processed_count = 0


# Open original restaurant reviews
with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as infile:

    reader = csv.DictReader(infile)

    # Keep all original columns
    # and add our NLP columns
    output_fields = reader.fieldnames + [
        "cleaned_text",
        "tokens",
        "without_stopwords",
        "stemmed_text",
        "lemmatized_text",
        "pos_tags"
    ]


    # Create processed CSV
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

        # Write column names
        writer.writeheader()


        # Process each review
        for row in reader:

            processed = process_review(
                row["text"]
            )

            # Add NLP results to the row
            row.update(processed)

            # Save row
            writer.writerow(row)

            processed_count += 1


            # Show progress every 10 reviews
            if processed_count % 10 == 0:
                print(
                    f"Processed {processed_count} reviews..."
                )


            # Stop during test mode
            if (
                MAX_ROWS is not None
                and processed_count >= MAX_ROWS
            ):
                break


# =========================================================
# 10. FINISHED
# =========================================================

print()
print("=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)

print(f"Reviews processed: {processed_count}")

print(f"Output file: {OUTPUT_FILE}")