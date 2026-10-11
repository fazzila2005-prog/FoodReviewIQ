import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# FOODREVIEWIQ
# STEP 15 - COMPLETE EVALUATION
# NO GROQ CALLS
# ============================================================

print("=" * 80)
print("FoodReviewIQ - Step 15 Evaluation")
print("Retrieval + Aspect + Quantitative + Evidence Evaluation")
print("=" * 80)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\FoodReviewIQ"

CHROMA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "chroma_db"
)

COLLECTION_NAME = "foodreviewiq_reviews"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5

QUANTITATIVE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "rating_log_odds.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "evaluation_results.csv"
)


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_CASES = [

    {
        "id": "Q1",
        "type": "RAG",
        "question": "What do customers think about food quality?",
        "expected_aspects": [
            "Food Quality"
        ]
    },

    {
        "id": "Q2",
        "type": "RAG",
        "question": "What do customers say about waiting times?",
        "expected_aspects": [
            "Waiting Time"
        ]
    },

    {
        "id": "Q3",
        "type": "RAG",
        "question": "What do customers think about service?",
        "expected_aspects": [
            "Service Quality",
            "Service Staff",
            "Staff Attitude"
        ]
    },

    {
        "id": "Q4",
        "type": "QUANTITATIVE",
        "question": "Which aspects are associated with low ratings?",
        "expected_aspects": [
            "Waiting Time",
            "Service Quality",
            "Service Staff"
        ]
    },

    {
        "id": "Q5",
        "type": "QUANTITATIVE",
        "question": "Which aspects are associated with high ratings?",
        "expected_aspects": [
            "Overall Ambience",
            "Ingredient Quality / Freshness",
            "Menu"
        ]
    },

    {
        "id": "Q6",
        "type": "COMBINED",
        "question": "Why is waiting time associated with bad ratings?",
        "expected_aspects": [
            "Waiting Time"
        ]
    }
]


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# LOAD CHROMADB
# ============================================================

print("\nLoading ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Documents in ChromaDB: "
    f"{collection.count():,}"
)


# ============================================================
# LOAD QUANTITATIVE DATA
# ============================================================

print("\nLoading quantitative analysis...")

quant_df = pd.read_csv(
    QUANTITATIVE_PATH
)

print(
    f"Quantitative aspect groups: "
    f"{len(quant_df)}"
)


# ============================================================
# RETRIEVE REVIEWS
# ============================================================

def retrieve_reviews(question):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# ============================================================
# CHECK ASPECT MATCH
# ============================================================

def aspect_matches(
    metadata,
    expected_aspects
):

    aspect_summary = str(
        metadata.get(
            "aspect_summary",
            ""
        )
    ).lower()

    for aspect in expected_aspects:

        if aspect.lower() in aspect_summary:

            return True

    return False


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    question,
    expected_aspects
):

    results = retrieve_reviews(
        question
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    relevant_count = 0

    evidence = []

    for i in range(len(documents)):

        metadata = metadatas[i]

        relevant = aspect_matches(
            metadata,
            expected_aspects
        )

        if relevant:

            relevant_count += 1

        evidence.append(
            {
                "rank": i + 1,

                "review_id": metadata.get(
                    "review_id",
                    ""
                ),

                "restaurant": metadata.get(
                    "restaurant_name",
                    ""
                ),

                "stars": metadata.get(
                    "review_stars",
                    ""
                ),

                "aspects": metadata.get(
                    "aspect_summary",
                    ""
                ),

                "document": documents[i],

                "distance": distances[i],

                "relevant": relevant
            }
        )

    precision = (
        relevant_count / TOP_K
    )

    return (
        precision,
        relevant_count,
        evidence
    )


# ============================================================
# QUANTITATIVE TOP-N EVALUATION
# ============================================================

def get_quantitative_ranking(
    low_rating=True
):

    filtered = quant_df[
        quant_df["total_reviews"] >= 100
    ].copy()

    if low_rating:

        filtered = filtered.sort_values(
            "log_odds_low_vs_high",
            ascending=False
        )

    else:

        filtered = filtered.sort_values(
            "log_odds_low_vs_high",
            ascending=True
        )

    return filtered.head(10)


# ============================================================
# CHECK EXPECTED ASPECTS IN TOP 10
# ============================================================

def evaluate_quantitative_aspects(
    expected_aspects,
    low_rating
):

    top_results = get_quantitative_ranking(
        low_rating=low_rating
    )

    actual = (
        top_results[
            "analysis_aspect"
        ]
        .astype(str)
        .tolist()
    )

    matched = 0

    matched_names = []

    for expected in expected_aspects:

        found = None

        for actual_name in actual:

            if (
                expected.lower()
                in actual_name.lower()
                or
                actual_name.lower()
                in expected.lower()
            ):

                found = actual_name
                break

        if found:

            matched += 1

            matched_names.append(
                found
            )

    coverage = (
        matched /
        len(expected_aspects)
    )

    return (
        coverage,
        matched_names,
        top_results
    )


# ============================================================
# WAITING TIME NUMERICAL VALIDATION
# ============================================================

def check_waiting_time():

    rows = quant_df[
        quant_df["analysis_aspect"]
        .astype(str)
        .str.lower()
        == "waiting time"
    ]

    if rows.empty:

        return None

    row = rows.iloc[0]

    return {
        "low_reviews": int(
            row["low_rating_reviews"]
        ),

        "high_reviews": int(
            row["high_rating_reviews"]
        ),

        "total_reviews": int(
            row["total_reviews"]
        ),

        "low_rate": float(
            row["low_rating_rate"]
        ),

        "high_rate": float(
            row["high_rating_rate"]
        ),

        "log_odds": float(
            row["log_odds_low_vs_high"]
        )
    }


# ============================================================
# RUN EVALUATION
# ============================================================

all_results = []

print("\n")
print("=" * 80)
print("RUNNING EVALUATION")
print("=" * 80)


for test in TEST_CASES:

    question_id = test["id"]

    question = test["question"]

    question_type = test["type"]

    expected_aspects = test[
        "expected_aspects"
    ]

    print("\n")
    print("=" * 80)

    print(
        f"{question_id}: "
        f"{question}"
    )

    print("=" * 80)

    print(
        f"Type: {question_type}"
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if question_type == "RAG":

        (
            precision,
            relevant_count,
            evidence
        ) = evaluate_retrieval(
            question,
            expected_aspects
        )

        print(
            f"\nRelevant reviews: "
            f"{relevant_count}/{TOP_K}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{precision:.2%}"
        )

        print("\nRetrieved evidence:")

        for item in evidence:

            print(
                f"  Rank {item['rank']} | "
                f"{item['restaurant']} | "
                f"{item['stars']} stars | "
                f"Relevant={item['relevant']}"
            )

        all_results.append(
            {
                "question_id": question_id,
                "question": question,
                "type": question_type,
                "retrieval_precision_at_5":
                    precision,
                "relevant_reviews":
                    relevant_count,
                "quantitative_coverage":
                    None
            }
        )

    # --------------------------------------------------------
    # QUANTITATIVE
    # --------------------------------------------------------

    elif question_type == "QUANTITATIVE":

        low_rating = (
            "low ratings"
            in question.lower()
        )

        (
            coverage,
            matched_names,
            top_results
        ) = evaluate_quantitative_aspects(
            expected_aspects,
            low_rating
        )

        print(
            f"\nExpected aspect coverage: "
            f"{coverage:.2%}"
        )

        print(
            "Matched aspects:",
            ", ".join(matched_names)
            if matched_names
            else "None"
        )

        print("\nTop 10 quantitative results:")

        for _, row in top_results.iterrows():

            print(
                f"  {row['analysis_aspect']} | "
                f"log-odds="
                f"{row['log_odds_low_vs_high']:.3f} | "
                f"total="
                f"{int(row['total_reviews']):,}"
            )

        all_results.append(
            {
                "question_id": question_id,
                "question": question,
                "type": question_type,
                "retrieval_precision_at_5":
                    None,
                "relevant_reviews":
                    None,
                "quantitative_coverage":
                    coverage
            }
        )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    elif question_type == "COMBINED":

        (
            precision,
            relevant_count,
            evidence
        ) = evaluate_retrieval(
            question,
            expected_aspects
        )

        waiting = check_waiting_time()

        print(
            f"\nEvidence retrieval:"
        )

        print(
            f"Relevant reviews: "
            f"{relevant_count}/{TOP_K}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{precision:.2%}"
        )

        print(
            "\nWaiting Time quantitative evidence:"
        )

        if waiting:

            print(
                f"  Low-rating reviews: "
                f"{waiting['low_reviews']:,}"
            )

            print(
                f"  High-rating reviews: "
                f"{waiting['high_reviews']:,}"
            )

            print(
                f"  Total reviews: "
                f"{waiting['total_reviews']:,}"
            )

            print(
                f"  Low-rating rate: "
                f"{waiting['low_rate']:.2%}"
            )

            print(
                f"  High-rating rate: "
                f"{waiting['high_rate']:.2%}"
            )

            print(
                f"  Log-odds: "
                f"{waiting['log_odds']:.4f}"
            )

        else:

            print(
                "  Waiting Time data not found."
            )

        print(
            "\nRetrieved evidence:"
        )

        for item in evidence:

            print(
                f"  Rank {item['rank']} | "
                f"{item['restaurant']} | "
                f"{item['stars']} stars | "
                f"Relevant={item['relevant']}"
            )

        all_results.append(
            {
                "question_id": question_id,
                "question": question,
                "type": question_type,
                "retrieval_precision_at_5":
                    precision,
                "relevant_reviews":
                    relevant_count,
                "quantitative_coverage":
                    None
            }
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# AVERAGE RETRIEVAL PRECISION
# ============================================================

retrieval_scores = (
    results_df[
        "retrieval_precision_at_5"
    ]
    .dropna()
)

if len(retrieval_scores) > 0:

    average_retrieval = (
        retrieval_scores.mean()
    )

else:

    average_retrieval = 0


# ============================================================
# AVERAGE QUANTITATIVE COVERAGE
# ============================================================

quantitative_scores = (
    results_df[
        "quantitative_coverage"
    ]
    .dropna()
)

if len(quantitative_scores) > 0:

    average_quantitative = (
        quantitative_scores.mean()
    )

else:

    average_quantitative = 0


# ============================================================
# COMBINED EVIDENCE
# ============================================================

combined_rows = results_df[
    results_df["type"] == "COMBINED"
]

if not combined_rows.empty:

    combined_precision = float(
        combined_rows[
            "retrieval_precision_at_5"
        ].iloc[0]
    )

else:

    combined_precision = 0


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL EVALUATION SUMMARY")
print("=" * 80)

print(
    f"\nRAG Average Precision@5: "
    f"{average_retrieval:.2%}"
)

print(
    f"Quantitative Aspect Coverage: "
    f"{average_quantitative:.2%}"
)

print(
    f"Combined Evidence Precision@5: "
    f"{combined_precision:.2%}"
)


# ============================================================
# WAITING TIME VALIDATION
# ============================================================

waiting = check_waiting_time()

print("\n")
print("=" * 80)
print("WAITING TIME NUMERICAL VALIDATION")
print("=" * 80)

if waiting:

    print(
        f"Low-rating reviews: "
        f"{waiting['low_reviews']:,}"
    )

    print(
        f"High-rating reviews: "
        f"{waiting['high_reviews']:,}"
    )

    print(
        f"Total reviews: "
        f"{waiting['total_reviews']:,}"
    )

    print(
        f"Low-rating rate: "
        f"{waiting['low_rate']:.2%}"
    )

    print(
        f"High-rating rate: "
        f"{waiting['high_rate']:.2%}"
    )

    print(
        f"Log-odds: "
        f"{waiting['log_odds']:.4f}"
    )


# ============================================================
# OVERALL SCORE
# ============================================================

scores = [
    average_retrieval,
    average_quantitative,
    combined_precision
]

overall_score = (
    sum(scores) /
    len(scores)
)


print("\n")
print("=" * 80)
print("OVERALL SYSTEM EVALUATION")
print("=" * 80)

print(
    f"\nOverall measurable system score: "
    f"{overall_score:.2%}"
)


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print(
    "\nDetailed results saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\nNo Groq API calls were used."
)

print(
    "Step 15 evaluation is complete."
)