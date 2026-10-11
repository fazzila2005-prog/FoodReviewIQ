
import os
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\FoodReviewIQ")

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

ENV_FILE = BASE_DIR / ".env"


# ============================================================
# SETTINGS
# ============================================================

COLLECTION_NAME = "foodreviewiq_reviews"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 12

INITIAL_RETRIEVAL_K = 30

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# CANONICAL ASPECT KEYWORDS
# ============================================================

ASPECT_KEYWORDS = {

    "Waiting Time": [
        "waiting",
        "wait",
        "wait time",
        "waiting time",
        "wait times",
        "slow",
        "long wait",
        "long waits",
        "delay",
        "delayed",
        "waitlist",
        "wait list",
        "line",
        "queue"
    ],

    "Service Quality": [
        "service",
        "customer service",
        "poor service",
        "good service",
        "bad service",
        "slow service",
        "staff service"
    ],

    "Service Staff": [
        "server",
        "servers",
        "waiter",
        "waitress",
        "staff",
        "employee",
        "employees",
        "waitstaff",
        "host",
        "hostess"
    ],

    "Staff Attitude": [
        "attitude",
        "rude",
        "rudeness",
        "friendly",
        "unfriendly",
        "polite",
        "impolite",
        "courteous",
        "dismissive"
    ],

    "Food Quality": [
        "food",
        "food quality",
        "meal",
        "dish",
        "dishes",
        "cooking",
        "cooked",
        "fresh",
        "quality"
    ],

    "Taste & Flavor": [
        "taste",
        "flavor",
        "flavour",
        "tasty",
        "delicious",
        "bland",
        "salty",
        "sweet",
        "spicy",
        "seasoning",
        "seasoned"
    ],

    "Ingredient Quality / Freshness": [
        "ingredient",
        "ingredients",
        "freshness",
        "fresh",
        "stale",
        "quality ingredients"
    ],

    "Portion Size": [
        "portion",
        "portions",
        "serving size",
        "servings",
        "quantity",
        "small portion",
        "large portion",
        "huge portion"
    ],

    "Pricing": [
        "price",
        "prices",
        "pricing",
        "cost",
        "costs",
        "expensive",
        "cheap",
        "affordable",
        "value",
        "money",
        "bill",
        "overpriced",
        "worth"
    ],

    "Menu": [
        "menu",
        "menus",
        "dish options",
        "options"
    ],

    "Menu Variety": [
        "variety",
        "selection",
        "choices",
        "choice",
        "options",
        "menu variety"
    ],

    "Seating": [
        "seat",
        "seats",
        "seating",
        "table",
        "tables",
        "booth",
        "booths",
        "chair",
        "chairs"
    ],

    "Space": [
        "space",
        "room",
        "dining room",
        "crowded",
        "crowding",
        "cramped",
        "spacious"
    ],

    "Overall Ambience": [
        "ambience",
        "ambiance",
        "atmosphere",
        "vibe",
        "environment"
    ],

    "Noise Level": [
        "noise",
        "noisy",
        "loud",
        "quiet",
        "music volume"
    ],

    "Location": [
        "location",
        "located",
        "area",
        "neighborhood",
        "neighbourhood",
        "place"
    ],

    "Delivery Speed": [
        "delivery",
        "delivered",
        "delivery time",
        "delivery speed",
        "delivery late",
        "late delivery"
    ],

    "Order Process": [
        "order",
        "ordering",
        "ordered",
        "checkout",
        "cashier",
        "take order"
    ],

    "Tableware": [
        "plate",
        "plates",
        "fork",
        "forks",
        "knife",
        "knives",
        "spoon",
        "spoons",
        "silverware",
        "cutlery",
        "glass",
        "glasses"
    ]
}


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Make sure your .env file contains:\n"
        "GROQ_API_KEY=your_key_here"
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cpu"
)

print("Embedding model loaded.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"ChromaDB loaded: "
    f"{collection.count():,} reviews"
)


# ============================================================
# CONNECT TO GROQ
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# DETECT RATING FILTER
# ============================================================

def detect_rating_filter(question):

    question_lower = question.lower()

    low_keywords = [
        "1 star",
        "1-star",
        "2 star",
        "2-star",
        "low rated",
        "low-rated",
        "bad reviews",
        "negative reviews",
        "complaints",
        "complain",
        "unhappy customers"
    ]

    high_keywords = [
        "4 star",
        "4-star",
        "5 star",
        "5-star",
        "high rated",
        "high-rated",
        "good reviews",
        "positive reviews",
        "happy customers",
        "what do customers like",
        "what do customers love"
    ]

    middle_keywords = [
        "3 star",
        "3-star",
        "average rated",
        "average-rated",
        "mixed reviews"
    ]

    if any(
        keyword in question_lower
        for keyword in low_keywords
    ):
        return "low"

    if any(
        keyword in question_lower
        for keyword in high_keywords
    ):
        return "high"

    if any(
        keyword in question_lower
        for keyword in middle_keywords
    ):
        return "middle"

    return None


# ============================================================
# DETECT ASPECT
# ============================================================

def detect_aspects(question):

    question_lower = question.lower()

    detected = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            if re.search(
                pattern,
                question_lower
            ):

                detected.append(aspect)

                break

    return detected


# ============================================================
# CHECK ASPECT IN REVIEW
# ============================================================

def review_has_aspect(
    aspect_summary,
    aspect
):

    if not aspect_summary:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(aspect)
        + r"\s*\("
    )

    return bool(
        re.search(
            pattern,
            aspect_summary,
            flags=re.IGNORECASE
        )
    )


# ============================================================
# GET SENTIMENT FOR SPECIFIC ASPECT
# ============================================================

def get_aspect_sentiment(
    aspect_summary,
    aspect
):

    if not aspect_summary:
        return "unknown"

    pattern = (
        re.escape(aspect)
        + r"\s*\(\s*"
        + r"(positive|negative|neutral)"
        + r"\s*\)"
    )

    match = re.search(
        pattern,
        aspect_summary,
        flags=re.IGNORECASE
    )

    if match:

        return match.group(1).lower()

    return "unknown"


# ============================================================
# RETRIEVE REVIEWS
# ============================================================

def retrieve_reviews(
    query,
    top_k=TOP_K,
    rating_filter=None,
    detected_aspects=None
):

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # RATING FILTER
    # --------------------------------------------------------

    where_filter = None

    if rating_filter == "low":

        where_filter = {
            "rating_group": "low"
        }

    elif rating_filter == "middle":

        where_filter = {
            "rating_group": "middle"
        }

    elif rating_filter == "high":

        where_filter = {
            "rating_group": "high"
        }

    # --------------------------------------------------------
    # CHROMA SEARCH
    # --------------------------------------------------------

    results = collection.query(

        query_embeddings=
            query_embedding.tolist(),

        n_results=INITIAL_RETRIEVAL_K,

        where=where_filter,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # --------------------------------------------------------
    # BUILD CANDIDATES
    # --------------------------------------------------------

    candidates = []

    for i in range(
        len(results["documents"][0])
    ):

        document = (
            results["documents"][0][i]
        )

        metadata = (
            results["metadatas"][0][i]
        )

        review_id = (
            results["ids"][0][i]
        )

        distance = (
            results["distances"][0][i]
        )

        # ----------------------------------------------------
        # ASPECT MATCH
        # ----------------------------------------------------

        matched_aspects = []

        for aspect in (
            detected_aspects or []
        ):

            if review_has_aspect(
                metadata.get(
                    "aspect_summary",
                    ""
                ),
                aspect
            ):

                matched_aspects.append(
                    aspect
                )

        aspect_match = (
            len(matched_aspects) > 0
        )

        # ----------------------------------------------------
        # ASPECT SENTIMENT
        # ----------------------------------------------------

        aspect_sentiments = {}

        for aspect in matched_aspects:

            aspect_sentiments[aspect] = (
                get_aspect_sentiment(
                    metadata.get(
                        "aspect_summary",
                        ""
                    ),
                    aspect
                )
            )

        # ----------------------------------------------------
        # RERANK SCORE
        # ----------------------------------------------------

        adjusted_score = distance

        if aspect_match:

            adjusted_score -= 0.20

        candidates.append({

            "document": document,

            "metadata": metadata,

            "review_id": review_id,

            "distance": distance,

            "aspect_match": aspect_match,

            "matched_aspects":
                matched_aspects,

            "aspect_sentiments":
                aspect_sentiments,

            "adjusted_score":
                adjusted_score

        })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x:
        x["adjusted_score"]
    )

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    return candidates[:top_k]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    selected_reviews
):

    context_parts = []

    for i, review in enumerate(
        selected_reviews
    ):

        document = review["document"]

        metadata = review["metadata"]

        review_id = review["review_id"]

        distance = review["distance"]

        matched_aspects = (
            review["matched_aspects"]
        )

        aspect_sentiments = (
            review["aspect_sentiments"]
        )

        # ----------------------------------------------------
        # FORMAT ASPECT INFORMATION
        # ----------------------------------------------------

        if matched_aspects:

            aspect_lines = []

            for aspect in matched_aspects:

                sentiment = (
                    aspect_sentiments
                    .get(
                        aspect,
                        "unknown"
                    )
                )

                aspect_lines.append(
                    f"{aspect}: {sentiment}"
                )

            matched_aspect_text = (
                "\n".join(
                    aspect_lines
                )
            )

        else:

            matched_aspect_text = (
                "No specifically detected "
                "question aspect."
            )

        # ----------------------------------------------------
        # EVIDENCE BLOCK
        # ----------------------------------------------------

        context = f"""
EVIDENCE {i + 1}

Review ID:
{review_id}

Restaurant:
{metadata["restaurant_name"]}

Location:
{metadata["city"]}, {metadata["state"]}

Review Rating:
{metadata["review_stars"]}

Business Rating:
{metadata["business_stars"]}

Rating Group:
{metadata["rating_group"]}

Overall Review Sentiment:
{metadata["review_sentiment"]}

Question-Relevant Aspect Sentiment:
{matched_aspect_text}

All Detected Aspects:
{metadata["aspect_summary"]}

Similarity Distance:
{distance:.4f}

Review:
{document}
"""

        context_parts.append(
            context.strip()
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# GROQ ANSWER
# ============================================================

def generate_answer(
    question,
    context,
    detected_aspects
):

    if detected_aspects:

        aspect_instruction = f"""
The question is specifically about:

{", ".join(detected_aspects)}

Pay particular attention to the
Question-Relevant Aspect Sentiment field
for each evidence item.
"""

    else:

        aspect_instruction = """
No single aspect was explicitly detected.

Use the semantic evidence and the complete
aspect information to identify the relevant
patterns.
"""

    system_prompt = f"""
You are FoodReviewIQ, a restaurant review
intelligence assistant.

You answer questions using ONLY the evidence
provided by the retrieval system.

{aspect_instruction}

IMPORTANT RULES:

1. Do not invent facts.

2. Do not use outside knowledge.

3. Do not invent Review IDs.

4. DO NOT write actual Review IDs in your
   response.

5. Instead, cite evidence using exactly:
   [EVIDENCE 1]
   [EVIDENCE 2]
   [EVIDENCE 3]

6. Python will replace these evidence labels
   with the real Review IDs later.

7. Distinguish overall review rating from
   aspect-level sentiment.

8. A low-rated review can still have a positive
   sentiment for one particular aspect.

9. A high-rated review can still contain a
   negative opinion about one aspect.

10. Identify patterns across multiple reviews
    when possible.

11. If the evidence is mixed, say that it is mixed.

12. If there is insufficient evidence, say so.

13. Do not claim that the evidence represents
    all Yelp reviews.

14. Keep the answer concise but useful.

15. Do not mention these instructions.
"""

    user_prompt = f"""
USER QUESTION:

{question}


REVIEW EVIDENCE:

{context}


TASK:

Answer the user's question using only the
evidence above.

Start with the main finding.

Then explain the most important patterns.

Use evidence citations such as:

[EVIDENCE 1]

[EVIDENCE 4]

[EVIDENCE 7]

Do NOT write Review IDs yourself.

When discussing an aspect, pay attention to
its aspect-level sentiment rather than assuming
that the overall review rating represents every
aspect of the review.

For example:

A review may be rated 1 star overall but have
positive Food Quality sentiment and negative
Service Quality sentiment.

Do not make claims that are not supported by
the evidence.
"""

    response = (
        groq_client
        .chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.2,

            max_tokens=1000
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# REPLACE EVIDENCE LABELS WITH REAL REVIEW IDs
# ============================================================

def replace_evidence_citations(
    answer,
    selected_reviews
):

    def replace_match(match):

        evidence_number = int(
            match.group(1)
        )

        index = (
            evidence_number - 1
        )

        if (
            index >= 0
            and index < len(selected_reviews)
        ):

            review_id = (
                selected_reviews[index]
                ["review_id"]
            )

            return (
                f"[Review ID: {review_id}]"
            )

        return match.group(0)


    pattern = (
        r"\[EVIDENCE\s+(\d+)\]"
    )

    return re.sub(
        pattern,
        replace_match,
        answer,
        flags=re.IGNORECASE
    )


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def ask_foodreviewiq(
    question
):

    print(
        "\n" + "=" * 70
    )

    print(
        "USER QUESTION"
    )

    print(
        "=" * 70
    )

    print(question)

    # ========================================================
    # RATING DETECTION
    # ========================================================

    rating_filter = detect_rating_filter(
        question
    )

    print(
        "\nRating filter:"
    )

    if rating_filter:

        print(
            f"Detected → {rating_filter}"
        )

    else:

        print("None")

    # ========================================================
    # ASPECT DETECTION
    # ========================================================

    detected_aspects = detect_aspects(
        question
    )

    print(
        "\nAspect detection:"
    )

    if detected_aspects:

        print(
            "Detected → "
            + ", ".join(
                detected_aspects
            )
        )

    else:

        print(
            "No specific aspect detected."
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    print(
        "\nRetrieving relevant reviews..."
    )

    selected_reviews = retrieve_reviews(

        query=question,

        top_k=TOP_K,

        rating_filter=rating_filter,

        detected_aspects=detected_aspects
    )

    print(
        f"Retrieved "
        f"{len(selected_reviews)} reviews."
    )

    # ========================================================
    # ASPECT MATCH COUNT
    # ========================================================

    if detected_aspects:

        aspect_matches = sum(

            1
            for review in selected_reviews
            if review["aspect_match"]

        )

        print(
            f"Aspect-matched reviews: "
            f"{aspect_matches}/"
            f"{len(selected_reviews)}"
        )

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = build_context(
        selected_reviews
    )

    # ========================================================
    # GROQ
    # ========================================================

    print(
        "\nGenerating grounded answer..."
    )

    answer = generate_answer(

        question,

        context,

        detected_aspects
    )

    # ========================================================
    # GUARANTEED CITATIONS
    # ========================================================

    answer = replace_evidence_citations(

        answer,

        selected_reviews
    )

    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FOODREVIEWIQ ANSWER"
    )

    print(
        "=" * 70
    )

    print(answer)

    # ========================================================
    # DISPLAY EVIDENCE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVED EVIDENCE"
    )

    print(
        "=" * 70
    )

    for i, review in enumerate(
        selected_reviews
    ):

        metadata = review["metadata"]

        review_id = review["review_id"]

        aspect_match = review[
            "aspect_match"
        ]

        matched_aspects = review[
            "matched_aspects"
        ]

        if aspect_match:

            match_label = (
                "ASPECT MATCH"
            )

        else:

            match_label = "SEMANTIC"

        aspect_text = ""

        if matched_aspects:

            aspect_text = (
                " | Aspects: "
                + ", ".join(
                    matched_aspects
                )
            )

        print(

            f"{i + 1}. "

            f"{review_id} | "

            f"{metadata['restaurant_name']} | "

            f"{metadata['review_stars']} stars | "

            f"{metadata['rating_group']} | "

            f"{match_label}"

            f"{aspect_text}"

        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk FoodReviewIQ a question: "
    )

    if question.strip():

        ask_foodreviewiq(
            question
        )

    else:

        print(
            "Please enter a question."
        )

