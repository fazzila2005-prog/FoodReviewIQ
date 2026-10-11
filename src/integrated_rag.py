import os
import re

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

from question_router import detect_question_type
from combined_analysis import (
    build_quantitative_context,
    get_specific_aspect_context
)


# ============================================================
# SETTINGS
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

GROQ_MODEL = "openai/gpt-oss-20b"

CHROMA_PATH = r"C:\FoodReviewIQ\data\chroma_db"
COLLECTION_NAME = "foodreviewiq_reviews"

TOP_K = 12
INITIAL_RETRIEVAL_K = 30


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)

print("Connecting to Groq...")

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("System ready.")


# ============================================================
# ASPECT KEYWORDS
# ============================================================

ASPECT_KEYWORDS = {

    "Waiting Time": [
        "waiting time",
        "wait time",
        "waiting",
        "wait",
        "long wait",
        "slow service"
    ],

    "Service Quality": [
        "service quality",
        "service",
        "poor service",
        "good service"
    ],

    "Service Staff": [
        "service staff",
        "staff",
        "server",
        "servers",
        "waiter",
        "waitress"
    ],

    "Staff Attitude": [
        "staff attitude",
        "attitude",
        "rude staff",
        "friendly staff"
    ],

    "Food Quality": [
        "food quality",
        "food"
    ],

    "Taste & Flavor": [
        "taste",
        "flavor",
        "flavour"
    ],

    "Ingredient Quality / Freshness": [
        "freshness",
        "fresh",
        "ingredients",
        "ingredient quality"
    ],

    "Portion Size": [
        "portion",
        "portion size",
        "serving size"
    ],

    "Pricing": [
        "price",
        "pricing",
        "cost",
        "expensive",
        "cheap",
        "value"
    ],

    "Menu": [
        "menu"
    ],

    "Menu Variety": [
        "menu variety",
        "variety"
    ],

    "Seating": [
        "seating",
        "seat",
        "seats"
    ],

    "Space": [
        "space",
        "room",
        "dining area"
    ],

    "Overall Ambience": [
        "ambience",
        "ambiance",
        "atmosphere"
    ],

    "Noise Level": [
        "noise",
        "noisy",
        "loud"
    ],

    "Location": [
        "location",
        "located",
        "parking"
    ],

    "Delivery Speed": [
        "delivery",
        "delivery speed"
    ],

    "Order Process": [
        "order",
        "ordering"
    ],

    "Tableware": [
        "tableware",
        "plates",
        "cutlery",
        "utensils"
    ]
}


# ============================================================
# RATING FILTER DETECTION
# ============================================================

def detect_rating_filter(question):

    q = question.lower()

    low_words = [
        "1 star",
        "2 star",
        "1-star",
        "2-star",
        "low-rated",
        "low rated",
        "bad reviews",
        "bad ratings",
        "negative reviews",
        "complaints",
        "complain",
        "unhappy customers"
    ]

    high_words = [
        "4 star",
        "5 star",
        "4-star",
        "5-star",
        "high-rated",
        "high rated",
        "good reviews",
        "good ratings",
        "positive reviews",
        "happy customers",
        "what do customers like",
        "what do customers love"
    ]

    middle_words = [
        "3 star",
        "3-star",
        "average-rated",
        "average rated",
        "mixed reviews"
    ]

    if any(word in q for word in low_words):
        return "low"

    if any(word in q for word in high_words):
        return "high"

    if any(word in q for word in middle_words):
        return "middle"

    return None


# ============================================================
# ASPECT DETECTION
# ============================================================

def detect_aspects(question):

    q = question.lower()

    detected = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in q:
                detected.append(aspect)
                break

    return detected


# ============================================================
# ASPECT MATCHING
# ============================================================

def review_has_aspect(aspect_summary, aspect):

    if not aspect_summary:
        return False

    return aspect.lower() in aspect_summary.lower()


def get_aspect_sentiment(aspect_summary, aspect):

    if not aspect_summary:
        return None

    pattern = (
        re.escape(aspect)
        + r"\s*\((positive|negative|neutral)\)"
    )

    match = re.search(
        pattern,
        aspect_summary,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    return None


# ============================================================
# RETRIEVE REVIEWS
# ============================================================

def retrieve_reviews(question):

    rating_filter = detect_rating_filter(question)
    detected_aspects = detect_aspects(question)

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    where_filter = None

    if rating_filter == "low":

        where_filter = {
            "rating_group": "low"
        }

    elif rating_filter == "high":

        where_filter = {
            "rating_group": "high"
        }

    elif rating_filter == "middle":

        where_filter = {
            "rating_group": "middle"
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=INITIAL_RETRIEVAL_K,
        where=where_filter
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    candidates = []

    for doc, metadata, distance, review_id in zip(
        documents,
        metadatas,
        distances,
        ids
    ):

        aspect_match = False

        for aspect in detected_aspects:

            if review_has_aspect(
                metadata.get("aspect_summary", ""),
                aspect
            ):

                aspect_match = True

                distance -= 0.20

        candidates.append({
            "review_id": review_id,
            "document": doc,
            "metadata": metadata,
            "distance": distance,
            "aspect_match": aspect_match
        })

    candidates.sort(
        key=lambda x: x["distance"]
    )

    return (
        candidates[:TOP_K],
        detected_aspects,
        rating_filter
    )


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(reviews, question):

    detected_aspects = detect_aspects(question)

    context = []

    for i, review in enumerate(
        reviews,
        start=1
    ):

        metadata = review["metadata"]

        aspect_summary = metadata.get(
            "aspect_summary",
            ""
        )

        relevant_sentiments = []

        for aspect in detected_aspects:

            sentiment = get_aspect_sentiment(
                aspect_summary,
                aspect
            )

            if sentiment:

                relevant_sentiments.append(
                    f"{aspect}: {sentiment}"
                )

        context.append(
            f"""
EVIDENCE {i}

Review ID:
{review["review_id"]}

Restaurant:
{metadata.get("restaurant_name", "")}

Location:
{metadata.get("city", "")}, {metadata.get("state", "")}

Review rating:
{metadata.get("review_stars", "")}

Business rating:
{metadata.get("business_stars", "")}

Rating group:
{metadata.get("rating_group", "")}

Overall review sentiment:
{metadata.get("review_sentiment", "")}

Question-relevant aspect sentiment:
{", ".join(relevant_sentiments) if relevant_sentiments else "Not directly detected"}

Detected aspects:
{aspect_summary}

Review:
{review["document"]}
"""
        )

    return "\n".join(context)


# ============================================================
# RAG ANSWER
# ============================================================

def generate_rag_answer(question, reviews):

    context = build_context(
        reviews,
        question
    )

    prompt = f"""
You are FoodReviewIQ, an evidence-based restaurant
review intelligence assistant.

Answer the user's question using ONLY the review
evidence provided below.

USER QUESTION:

{question}

REVIEW EVIDENCE:

{context}

Instructions:

1. Give a concise but useful answer.
2. Base claims on the evidence.
3. Look specifically at the aspect relevant to the question.
4. Do not confuse overall review rating with aspect sentiment.
5. Mention both positive and negative evidence when relevant.
6. Do not invent information.
7. Do not write actual Review IDs in the answer.
8. Cite evidence using [EVIDENCE 1], [EVIDENCE 2], etc.
9. If evidence is mixed, clearly say that it is mixed.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ============================================================
# REPLACE EVIDENCE CITATIONS
# ============================================================

def replace_evidence_citations(answer, reviews):

    def replace(match):

        number = int(match.group(1))

        if 1 <= number <= len(reviews):

            review_id = reviews[
                number - 1
            ]["review_id"]

            return f"[Review ID: {review_id}]"

        return match.group(0)

    return re.sub(
        r"\[EVIDENCE\s+(\d+)\]",
        replace,
        answer
    )


# ============================================================
# QUANTITATIVE ANSWER
# ============================================================

def generate_quantitative_answer(question):

    quantitative_context = build_quantitative_context()

    prompt = f"""
You are FoodReviewIQ, an evidence-based restaurant
review intelligence assistant.

The user asked:

{question}

Use the following quantitative analysis:

{quantitative_context}

Important interpretation:

- Low-rating reviews are defined as 1–2 star reviews.
- High-rating reviews are defined as 4–5 star reviews.
- 3-star reviews are excluded from the low-vs-high comparison.
- Positive log-odds mean an aspect is relatively more
  associated with low ratings.
- Negative log-odds mean an aspect is relatively more
  associated with high ratings.
- These results show association, NOT causation.
- Do NOT call an association statistically significant
  unless an explicit statistical significance test is provided.
- Prefer aspects with larger review counts.
- Do not overstate rare aspects.
- Do not describe an aspect as a "driver", "cause",
  "culprit", or something that "makes" ratings lower.
- Use wording such as "associated with", "more common in",
  or "appears more often in".

Answer clearly for a non-technical user.

Use cautious scientific language.

Do not make causal claims.

Do not invent statistical tests or significance values.

Explain what the strongest patterns are and include
the relevant percentages and log-odds when useful.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ============================================================
# COMBINED ANSWER
# ============================================================

def generate_combined_answer(question, reviews):

    detected_aspects = detect_aspects(question)

    quantitative_parts = []

    for aspect in detected_aspects:

        result = get_specific_aspect_context(
            aspect
        )

        quantitative_parts.append(result)

    if quantitative_parts:

        quantitative_context = "\n\n".join(
            quantitative_parts
        )

    else:

        quantitative_context = (
            build_quantitative_context()
        )

    review_context = build_context(
        reviews,
        question
    )

    prompt = f"""
You are FoodReviewIQ, an evidence-based restaurant
review intelligence assistant.

The user asked:

{question}

QUANTITATIVE ANALYSIS:

{quantitative_context}

REVIEW EVIDENCE:

{review_context}

Your task is to combine the quantitative analysis
with the actual review evidence.

Instructions:

1. Explain the statistical association.
2. Then use the review evidence to explain what
   customers are actually saying.
3. Distinguish association from causation.
4. Low-rating reviews mean 1–2 stars.
5. High-rating reviews mean 4–5 stars.
6. Three-star reviews are excluded from this comparison.
7. Do not call results statistically significant unless
   a formal significance test is provided.
8. Do not use words such as "cause", "culprit",
   "driver", or "makes ratings lower" as factual conclusions.
9. Do not claim that the quantitative result proves
   that an aspect causes low ratings.
10. Use actual review evidence to illustrate the pattern.
11. Do not invent information.
12. Do not write actual Review IDs.
13. Cite review evidence using [EVIDENCE 1],
    [EVIDENCE 2], etc.
14. Keep the explanation understandable for a
    non-technical user.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)
    print("FoodReviewIQ")
    print("=" * 70)

    question = input(
        "\nAsk a restaurant review question: "
    ).strip()

    if not question:

        print("Please enter a question.")

        raise SystemExit

    question_type = detect_question_type(
        question
    )

    print(
        f"\nQuestion type: "
        f"{question_type.upper()}"
    )


    # ========================================================
    # QUANTITATIVE
    # ========================================================

    if question_type == "quantitative":

        print(
            "\nRunning quantitative analysis..."
        )

        answer = generate_quantitative_answer(
            question
        )

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(answer)


    # ========================================================
    # RAG
    # ========================================================

    elif question_type == "rag":

        print(
            "\nRetrieving review evidence..."
        )

        reviews, aspects, rating_filter = (
            retrieve_reviews(question)
        )

        print(
            f"Rating filter: "
            f"{rating_filter}"
        )

        print(
            f"Detected aspects: "
            f"{', '.join(aspects) if aspects else 'None'}"
        )

        print(
            f"Retrieved reviews: "
            f"{len(reviews)}"
        )

        aspect_matches = sum(
            r["aspect_match"]
            for r in reviews
        )

        print(
            f"Aspect matches: "
            f"{aspect_matches}/{len(reviews)}"
        )

        answer = generate_rag_answer(
            question,
            reviews
        )

        answer = replace_evidence_citations(
            answer,
            reviews
        )

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(answer)

        print("\n" + "=" * 70)
        print("EVIDENCE")
        print("=" * 70)

        for i, review in enumerate(
            reviews,
            start=1
        ):

            metadata = review["metadata"]

            print(
                f"\n[{i}] "
                f"{metadata.get('restaurant_name', '')}"
            )

            print(
                f"Review ID: "
                f"{review['review_id']}"
            )

            print(
                f"Rating: "
                f"{metadata.get('review_stars', '')}"
            )


    # ========================================================
    # COMBINED
    # ========================================================

    elif question_type == "combined":

        print(
            "\nRunning quantitative + RAG analysis..."
        )

        reviews, aspects, rating_filter = (
            retrieve_reviews(question)
        )

        print(
            f"Detected aspects: "
            f"{', '.join(aspects) if aspects else 'None'}"
        )

        print(
            f"Retrieved reviews: "
            f"{len(reviews)}"
        )

        answer = generate_combined_answer(
            question,
            reviews
        )

        answer = replace_evidence_citations(
            answer,
            reviews
        )

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(answer)

        print("\n" + "=" * 70)
        print("EVIDENCE")
        print("=" * 70)

        for i, review in enumerate(
            reviews,
            start=1
        ):

            metadata = review["metadata"]

            print(
                f"\n[{i}] "
                f"{metadata.get('restaurant_name', '')}"
            )

            print(
                f"Review ID: "
                f"{review['review_id']}"
            )

            print(
                f"Rating: "
                f"{metadata.get('review_stars', '')}"
            )