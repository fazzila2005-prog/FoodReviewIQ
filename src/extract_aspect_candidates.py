import os
import re
from collections import Counter, defaultdict

import pandas as pd
import spacy
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"C:\FoodReviewIQ\data\processed\processed_reviews.csv"

OUTPUT_OCCURRENCES = r"C:\FoodReviewIQ\data\processed\aspect_candidates.csv"
OUTPUT_COUNTS = r"C:\FoodReviewIQ\data\processed\aspect_candidate_counts.csv"


# ============================================================
# SETTINGS
# ============================================================

MIN_TOKENS = 1
MAX_TOKENS = 6
MIN_FREQUENCY = 5


# Words that are usually people rather than restaurant aspects.
GENERIC_NON_ASPECT_WORDS = {
    "customer",
    "customers",
    "person",
    "people",
    "guy",
    "guys",
    "man",
    "woman",
    "boy",
    "girl",
    "boyfriend",
    "girlfriend",
    "husband",
    "wife",
    "friend",
    "friends",
    "family",
    "mom",
    "dad",
    "mother",
    "father",
    "son",
    "daughter",
    "child",
    "children",
    "kid",
    "kids",
    "reviewer",
}


# Entities that are unlikely to be useful restaurant aspects.
IGNORED_ENTITY_TYPES = {
    "PERSON",
    "GPE",
    "LOC",
    "ORG",
    "DATE",
    "TIME",
    "MONEY",
    "CARDINAL",
    "ORDINAL",
    "PERCENT",
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed reviews...")

df = pd.read_csv(INPUT_FILE)

print(f"Reviews loaded: {len(df):,}")


# We want the original review text because it gives us
# natural phrases such as "waiter", "pasta sauce", etc.
if "text" in df.columns:
    TEXT_COLUMN = "text"
else:
    raise ValueError("Could not find the original review 'text' column.")


if "review_id" not in df.columns:
    raise ValueError("Could not find 'review_id' column.")


# ============================================================
# LOAD SPACY
# ============================================================

print("Loading spaCy model...")

try:
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["ner"]
    )
except Exception:
    print("Could not load en_core_web_sm.")
    print("Run:")
    print("python -m spacy download en_core_web_sm")
    raise


# We need noun_chunks, which depend on the parser.
# So reload with parser enabled and NER disabled.
nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner"]
)

print("spaCy loaded.")


# ============================================================
# EXTRACTION HELPERS
# ============================================================

def clean_candidate(span):
    """
    Convert a noun phrase into a cleaner normalized phrase.
    """

    words = []

    for token in span:

        # Skip punctuation and spaces
        if token.is_punct or token.is_space:
            continue

        # Ignore pronouns
        if token.pos_ == "PRON":
            continue

        # Keep nouns, proper nouns, adjectives and compounds
        if token.pos_ not in {"NOUN", "PROPN", "ADJ"}:
            continue

        word = token.lemma_.lower().strip()

        # Only keep meaningful alphabetic tokens
        if not word:
            continue

        if not any(char.isalpha() for char in word):
            continue

        words.append(word)

    if not words:
        return None

    phrase = " ".join(words)

    # Remove repeated spaces
    phrase = re.sub(r"\s+", " ", phrase).strip()

    return phrase


def is_valid_candidate(span, phrase):
    """
    Decide whether a candidate phrase is worth keeping.
    """

    tokens = phrase.split()

    if len(tokens) < MIN_TOKENS or len(tokens) > MAX_TOKENS:
        return False

    # Reject phrases containing obvious people-related words.
    if any(word in GENERIC_NON_ASPECT_WORDS for word in tokens):
        return False

    # Original span should contain at least one noun/proper noun.
    if not any(token.pos_ in {"NOUN", "PROPN"} for token in span):
        return False

    # Avoid URL-like / email-like strings.
    if "http" in phrase or "@" in phrase:
        return False

    return True


# ============================================================
# PROCESS REVIEWS
# ============================================================

candidate_rows = []

candidate_frequency = Counter()
candidate_reviews = defaultdict(set)

print("\nExtracting candidate aspects...")

texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
review_ids = df["review_id"].astype(str).tolist()

for review_id, doc in tqdm(
    zip(
        review_ids,
        nlp.pipe(
            texts,
            batch_size=200
        )
    ),
    total=len(texts),
    desc="Processing reviews"
):

    seen_in_review = set()

    # --------------------------------------------------------
    # Main candidate extraction:
    # noun phrases
    # --------------------------------------------------------

    for chunk in doc.noun_chunks:

        phrase = clean_candidate(chunk)

        if phrase is None:
            continue

        if not is_valid_candidate(chunk, phrase):
            continue

        # Avoid counting the same phrase twice in one review.
        if phrase in seen_in_review:
            continue

        seen_in_review.add(phrase)

        surface_phrase = chunk.text.strip()

        candidate_rows.append({
            "review_id": review_id,
            "surface_phrase": surface_phrase,
            "aspect_candidate": phrase
        })

        candidate_frequency[phrase] += 1
        candidate_reviews[phrase].add(review_id)


# ============================================================
# SAVE ALL OCCURRENCES
# ============================================================

occurrence_df = pd.DataFrame(candidate_rows)

occurrence_df.to_csv(
    OUTPUT_OCCURRENCES,
    index=False
)

print(
    f"\nCandidate occurrences saved: "
    f"{OUTPUT_OCCURRENCES}"
)


# ============================================================
# BUILD FREQUENCY TABLE
# ============================================================

count_rows = []

for phrase, frequency in candidate_frequency.items():

    count_rows.append({
        "aspect_candidate": phrase,
        "frequency": frequency,
        "unique_reviews": len(candidate_reviews[phrase])
    })


count_df = pd.DataFrame(count_rows)

count_df = count_df.sort_values(
    by=["frequency", "unique_reviews"],
    ascending=False
)

# Keep reasonably frequent candidates for the next stages.
count_df_filtered = count_df[
    count_df["frequency"] >= MIN_FREQUENCY
].copy()

count_df_filtered.to_csv(
    OUTPUT_COUNTS,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("ASPECT EXTRACTION COMPLETE")
print("========================================")

print(f"Reviews processed: {len(df):,}")
print(f"Unique candidate phrases: {len(count_df):,}")
print(
    f"Candidates with frequency >= {MIN_FREQUENCY}: "
    f"{len(count_df_filtered):,}"
)

print("\nTop 50 candidate aspects:\n")

print(
    count_df_filtered.head(50).to_string(index=False)
)