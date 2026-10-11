import pandas as pd
import spacy


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"C:\FoodReviewIQ\data\processed\filtered_aspect_candidates.csv"

OUTPUT_FILE = r"C:\FoodReviewIQ\data\processed\refined_aspect_candidates.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading candidate aspects...")

df = pd.read_csv(INPUT_FILE)

df["aspect_candidate"] = (
    df["aspect_candidate"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["aspect_candidate"] != ""].copy()

print(f"Candidates loaded: {len(df):,}")


# ============================================================
# LOAD SPACY WITH NER
# ============================================================

print("\nLoading spaCy...")

nlp = spacy.load("en_core_web_sm")

print("spaCy loaded.")


# ============================================================
# ENTITY TYPES TO REMOVE
# ============================================================

BAD_ENTITY_TYPES = {
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
# GENERIC WORDS
# ============================================================

GENERIC_WORDS = {
    "thing",
    "things",
    "way",
    "reason",
    "fact",
    "point",
    "kind",
    "sort",
    "one",
    "someone",
    "something",
    "anything",
    "everything",
    "people",
    "person",
    "couple",
    "group",
    "guy",
    "guys",
    "man",
    "woman",
    "boy",
    "girl",
    "home",
    "life",
    "year",
    "day",
    "night",
    "name",
    "time",
    "rest",
    "other",
    "part",
    "lot",
    "plenty",
    "top",
}


# ============================================================
# CHECK WHETHER CANDIDATE SHOULD BE KEPT
# ============================================================

def should_keep(doc):

    phrase = doc.text.strip().lower()

    if not phrase:
        return False

    # --------------------------------------------------------
    # Remove named entities / locations / people
    # --------------------------------------------------------

    for ent in doc.ents:
        if ent.label_ in BAD_ENTITY_TYPES:
            return False

    # --------------------------------------------------------
    # Remove generic single words
    # --------------------------------------------------------

    words = phrase.split()

    if len(words) == 1 and words[0] in GENERIC_WORDS:
        return False

    # --------------------------------------------------------
    # Keep only phrases containing a noun
    # --------------------------------------------------------

    if not any(token.pos_ in {"NOUN", "PROPN"} for token in doc):
        return False

    return True


# ============================================================
# PROCESS CANDIDATES
# ============================================================

print("\nChecking candidate quality...")

documents = nlp.pipe(
    df["aspect_candidate"].tolist(),
    batch_size=500
)

keep_flags = []

for doc in documents:
    keep_flags.append(should_keep(doc))


df["keep"] = keep_flags

refined_df = df[df["keep"]].copy()

refined_df.drop(columns=["keep"], inplace=True)


# ============================================================
# SORT
# ============================================================

refined_df = refined_df.sort_values(
    by=["frequency", "unique_reviews"],
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

refined_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("CANDIDATE REFINEMENT COMPLETE")
print("========================================")

print(f"Before refinement: {len(df):,}")
print(f"After refinement:  {len(refined_df):,}")
print(f"Removed:           {len(df) - len(refined_df):,}")

print("\nTop 100 refined candidates:\n")

print(
    refined_df.head(100).to_string(index=False)
)

print(f"\nSaved to:")
print(OUTPUT_FILE)