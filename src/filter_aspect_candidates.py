import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"C:\FoodReviewIQ\data\processed\aspect_candidate_counts.csv"

OUTPUT_FILE = r"C:\FoodReviewIQ\data\processed\filtered_aspect_candidates.csv"


# ============================================================
# SETTINGS
# ============================================================

MIN_FREQUENCY = 5


# These are very common words, but usually do not represent
# a useful restaurant aspect by themselves.

GENERIC_SINGLE_WORDS = {
    "place",
    "restaurant",
    "time",
    "lot",
    "star",
    "way",
    "thing",
    "one",
    "day",
    "town",
    "review",
    "experience",
    "area",
    "first",
    "person",
    "people",
    "customer",
    "customers",
    "guy",
    "guys",
    "man",
    "woman",
    "boy",
    "girl",
    "friend",
    "friends",
    "family",
}


# Words that are mostly review-writing/meta language.
GENERIC_PHRASES = {
    "first time",
    "review",
    "first review",
    "last time",
    "other thing",
}


# ============================================================
# LOAD
# ============================================================

print("Loading candidate aspects...")

df = pd.read_csv(INPUT_FILE)

print(f"Candidates before filtering: {len(df):,}")


# ============================================================
# FREQUENCY FILTER
# ============================================================

df = df[df["frequency"] >= MIN_FREQUENCY].copy()

print(
    f"Candidates after frequency filter: "
    f"{len(df):,}"
)


# ============================================================
# CLEAN
# ============================================================

def should_keep(phrase):

    # Handle empty / missing values
    if pd.isna(phrase):
        return False

    phrase = str(phrase).strip().lower()

    # Reject empty phrases
    if not phrase:
        return False

    # Reject known useless phrases.
    if phrase in GENERIC_PHRASES:
        return False

    # For single words, remove generic words.
    if len(phrase.split()) == 1:
        if phrase in GENERIC_SINGLE_WORDS:
            return False

    return True

df["keep"] = df["aspect_candidate"].apply(should_keep)

filtered_df = df[df["keep"]].copy()

filtered_df.drop(columns=["keep"], inplace=True)


# ============================================================
# SORT
# ============================================================

filtered_df = filtered_df.sort_values(
    by=["frequency", "unique_reviews"],
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

filtered_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("CANDIDATE FILTERING COMPLETE")
print("========================================")

print(
    f"Before filtering: "
    f"{len(df):,}"
)

print(
    f"After filtering:  "
    f"{len(filtered_df):,}"
)

print(
    f"Removed:           "
    f"{len(df) - len(filtered_df):,}"
)

print("\nTop 100 remaining candidates:\n")

print(
    filtered_df.head(100).to_string(index=False)
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)