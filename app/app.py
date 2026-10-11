import os
import sys
import pandas as pd
import streamlit as st
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = r"C:\FoodReviewIQ"

SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

CHROMA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "chroma_db"
)

QUANTITATIVE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "rating_log_odds.csv"
)

STATISTICS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "statistical_significance_results.csv"
)

ENV_PATH = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(ENV_PATH)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FoodReviewIQ",
    page_icon="🍽️",
    layout="wide"
)

GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 8


# ============================================================
# LOAD MODELS / DATABASE
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_chroma():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name="foodreviewiq_reviews"
    )

    return collection


@st.cache_resource
def load_groq():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


@st.cache_data
def load_quantitative_data():

    return pd.read_csv(
        QUANTITATIVE_PATH
    )


@st.cache_data
def load_statistical_data():

    return pd.read_csv(
        STATISTICS_PATH
    )


embedding_model = load_embedding_model()
collection = load_chroma()
groq_client = load_groq()

quant_df = load_quantitative_data()
stats_df = load_statistical_data()


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(question):

    q = question.lower()

    combined_keywords = [
        "why is",
        "why are",
        "why does",
        "why do",
        "explain why",
        "what explains",
        "explain the association",
        "why associated",
        "why linked",
        "what causes",
        "what might explain"
    ]

    quantitative_keywords = [
        "which aspects",
        "what aspects",
        "most associated",
        "associated with",
        "low ratings",
        "low-rated",
        "low rated",
        "bad ratings",
        "high ratings",
        "high-rated",
        "high rated",
        "good ratings",
        "rating patterns",
        "log odds",
        "log-odds",
        "statistically",
        "strongest association",
        "strongest associations",
        "top aspects",
        "main drivers",
        "main factors",
        "factors behind",
        "compare aspects",
        "compare aspect"
    ]

    has_combined = any(
        keyword in q
        for keyword in combined_keywords
    )

    has_quantitative = any(
        keyword in q
        for keyword in quantitative_keywords
    )

    if has_combined and has_quantitative:
        return "COMBINED"

    if has_quantitative:
        return "QUANTITATIVE"

    return "RAG"


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_reviews(question, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    evidence = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for i in range(len(documents)):

        metadata = metadatas[i]

        evidence.append({
            "rank": i + 1,
            "review_id": ids[i],
            "document": documents[i],
            "metadata": metadata,
            "distance": distances[i]
        })

    return evidence


# ============================================================
# QUANTITATIVE ANALYSIS
# ============================================================

def get_low_rating_aspects():

    data = quant_df[
        quant_df["total_reviews"] >= 100
    ].copy()

    return data.sort_values(
        "log_odds_low_vs_high",
        ascending=False
    ).head(10)


def get_high_rating_aspects():

    data = quant_df[
        quant_df["total_reviews"] >= 100
    ].copy()

    return data.sort_values(
        "log_odds_low_vs_high",
        ascending=True
    ).head(10)


def get_waiting_time():

    rows = quant_df[
        quant_df["analysis_aspect"]
        .astype(str)
        .str.lower()
        == "waiting time"
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


# ============================================================
# BUILD QUANTITATIVE CONTEXT
# ============================================================

def build_quantitative_context():

    low = get_low_rating_aspects()
    high = get_high_rating_aspects()

    context = []

    context.append(
        "ASPECTS ASSOCIATED WITH LOW RATINGS:"
    )

    for _, row in low.iterrows():

        context.append(
            f"- {row['analysis_aspect']}: "
            f"log-odds={row['log_odds_low_vs_high']:.3f}, "
            f"low-rating rate={row['low_rating_rate']:.3f}, "
            f"high-rating rate={row['high_rating_rate']:.3f}, "
            f"total reviews={int(row['total_reviews'])}"
        )

    context.append("")

    context.append(
        "ASPECTS ASSOCIATED WITH HIGH RATINGS:"
    )

    for _, row in high.iterrows():

        context.append(
            f"- {row['analysis_aspect']}: "
            f"log-odds={row['log_odds_low_vs_high']:.3f}, "
            f"low-rating rate={row['low_rating_rate']:.3f}, "
            f"high-rating rate={row['high_rating_rate']:.3f}, "
            f"total reviews={int(row['total_reviews'])}"
        )

    return "\n".join(context)


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    question_type,
    evidence=None,
    quantitative_context=None
):

    if groq_client is None:

        return (
            "Groq API key was not found. "
            "Please check your .env file."
        )

    evidence_text = ""

    if evidence:

        for i, item in enumerate(evidence):

            metadata = item["metadata"]

            evidence_text += (
                f"\n[EVIDENCE {i + 1}]\n"
                f"Review ID: {item.get('review_id', '')}\n"
                f"Restaurant: {metadata.get('restaurant_name', '')}\n"
                f"Review rating: {metadata.get('review_stars', '')}\n"
                f"Detected aspects: {metadata.get('aspect_summary', '')}\n"
                f"Review:\n{item['document']}\n"
            )

    prompt = f"""
You are FoodReviewIQ, an evidence-based restaurant review
intelligence assistant.

Question:
{question}

Question type:
{question_type}

Quantitative analysis:
{quantitative_context if quantitative_context else "Not required."}

Retrieved review evidence:
{evidence_text if evidence_text else "Not required."}

Instructions:

1. Answer the user's question directly.
2. Use only the supplied evidence and quantitative data.
3. Do not invent facts.
4. For RAG questions, summarize what customers say.
5. For quantitative questions, report the actual numbers.
6. For combined questions, connect the numerical pattern
   with examples from the retrieved reviews.
7. Distinguish association from causation.
8. Do not claim statistical significance unless the supplied
   statistical analysis supports it.
9. When citing a review, use:
   [Review ID: XXXXX]
10. Keep the answer clear and concise.
11. If the evidence is mixed, say so.
12. Do not mention internal implementation details unless useful.

Answer:
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
# SESSION STATE
# ============================================================

if "answer" not in st.session_state:
    st.session_state.answer = None

if "evidence" not in st.session_state:
    st.session_state.evidence = []

if "question_type" not in st.session_state:
    st.session_state.question_type = None


# ============================================================
# HEADER
# ============================================================

st.title("🍽️ FoodReviewIQ")

st.markdown(
    """
### Aspect-Aware Restaurant Review Intelligence

Ask questions about restaurant reviews and get
**evidence-based answers using RAG, dynamic aspect discovery,
and quantitative rating analysis.**
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 System")

    st.metric(
        "Reviews in RAG",
        f"{collection.count():,}"
    )

    st.metric(
        "Quantitative Aspects",
        f"{len(quant_df):,}"
    )

    significant_count = int(
        stats_df[
            stats_df["significant_after_bh"] == True
        ].shape[0]
    )

    st.metric(
        "Significant Aspects",
        f"{significant_count}"
    )

    st.divider()

    st.markdown(
        """
**Pipeline**

50,000 Yelp Reviews  
↓  
NLP Processing  
↓  
Dynamic Aspect Discovery  
↓  
Aspect Sentiment  
↓  
Statistical Analysis  
↓  
BGE Embeddings  
↓  
ChromaDB  
↓  
RAG / Quantitative Analysis  
↓  
Groq LLM
"""
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 Ask FoodReviewIQ",
        "📈 Rating Insights",
        "ℹ️ About the Project"
    ]
)


# ============================================================
# TAB 1 — ASK
# ============================================================

with tab1:

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question:",
        placeholder=(
            "e.g. What do customers think about food quality?"
        )
    )

    example_questions = [
        "What do customers think about food quality?",
        "What do customers say about waiting times?",
        "What do customers think about service?",
        "Which aspects are associated with low ratings?",
        "Which aspects are associated with high ratings?",
        "Why is waiting time associated with bad ratings?"
    ]

    st.caption("Try one of these:")

    selected_example = st.selectbox(
        "Example questions",
        ["-- Select --"] + example_questions
    )

    if selected_example != "-- Select --":

        question = selected_example

    ask_button = st.button(
        "🔍 Analyze Reviews",
        type="primary",
        use_container_width=True
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            question_type = detect_question_type(
                question
            )

            st.session_state.question_type = question_type

            with st.spinner(
                "Analyzing restaurant reviews..."
            ):

                evidence = []

                if question_type in [
                    "RAG",
                    "COMBINED"
                ]:

                    evidence = retrieve_reviews(
                        question
                    )

                quantitative_context = None

                if question_type in [
                    "QUANTITATIVE",
                    "COMBINED"
                ]:

                    quantitative_context = (
                        build_quantitative_context()
                    )

                try:

                    answer = generate_answer(
                        question,
                        question_type,
                        evidence,
                        quantitative_context
                    )

                    st.session_state.answer = answer
                    st.session_state.evidence = evidence

                except Exception as e:

                    st.session_state.answer = None

                    st.error(
                        "The answer could not be generated. "
                        "Your Groq API may currently be rate-limited "
                        "or unavailable."
                    )

                    st.caption(
                        f"Technical message: {str(e)}"
                    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    if st.session_state.answer:

        st.divider()

        st.subheader("💡 FoodReviewIQ Answer")

        if st.session_state.question_type:

            st.caption(
                f"Analysis type: "
                f"**{st.session_state.question_type}**"
            )

        st.markdown(
            st.session_state.answer
        )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    if st.session_state.evidence:

        st.divider()

        st.subheader(
            "📚 Retrieved Evidence"
        )

        for item in st.session_state.evidence:

            metadata = item["metadata"]

            with st.expander(
                f"#{item['rank']} — "
                f"{metadata.get('restaurant_name', 'Unknown')} "
                f"⭐ {metadata.get('review_stars', '')}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f"**Review ID:** "
                        f"{item.get('review_id', '')}"
                    )

                    st.write(
                        f"**Restaurant:** "
                        f"{metadata.get('restaurant_name', '')}"
                    )

                with col2:

                    st.write(
                        f"**Rating:** "
                        f"{metadata.get('review_stars', '')} ⭐"
                    )

                    st.write(
                        f"**Detected aspects:** "
                        f"{metadata.get('aspect_summary', '')}"
                    )

                st.write(
                    metadata.get(
                        "review_sentiment",
                        ""
                    )
                )

                st.write(
                    item["document"]
                )


# ============================================================
# TAB 2 — RATING INSIGHTS
# ============================================================

with tab2:

    st.subheader(
        "📈 What is associated with ratings?"
    )

    st.markdown(
        """
This section shows patterns found by comparing
**1–2 star reviews** with **4–5 star reviews**.

3-star reviews were excluded from this analysis.
"""
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LOW RATINGS
    # --------------------------------------------------------

    with col1:

        st.markdown(
            "### 🔻 Associated with Low Ratings"
        )

        low = get_low_rating_aspects().copy()

        low_display = low[
            [
                "analysis_aspect",
                "log_odds_low_vs_high",
                "low_rating_rate",
                "high_rating_rate",
                "total_reviews"
            ]
        ].copy()

        low_display.columns = [
            "Aspect",
            "Log-Odds",
            "Low Rate",
            "High Rate",
            "Reviews"
        ]

        low_display["Low Rate"] = (
            low_display["Low Rate"] * 100
        ).round(2).astype(str) + "%"

        low_display["High Rate"] = (
            low_display["High Rate"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            low_display,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # HIGH RATINGS
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### 🔺 Associated with High Ratings"
        )

        high = get_high_rating_aspects().copy()

        high_display = high[
            [
                "analysis_aspect",
                "log_odds_low_vs_high",
                "low_rating_rate",
                "high_rating_rate",
                "total_reviews"
            ]
        ].copy()

        high_display.columns = [
            "Aspect",
            "Log-Odds",
            "Low Rate",
            "High Rate",
            "Reviews"
        ]

        high_display["Low Rate"] = (
            high_display["Low Rate"] * 100
        ).round(2).astype(str) + "%"

        high_display["High Rate"] = (
            high_display["High Rate"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            high_display,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # WAITING TIME
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "⏱️ Waiting Time — Statistical Evidence"
    )

    waiting = get_waiting_time()

    if waiting is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Low-rating rate",
                f"{waiting['low_rating_rate']:.2%}"
            )

        with col2:

            st.metric(
                "High-rating rate",
                f"{waiting['high_rating_rate']:.2%}"
            )

        with col3:

            waiting_stats = stats_df[
                stats_df["analysis_aspect"]
                .astype(str)
                .str.lower()
                == "waiting time"
            ]

            if not waiting_stats.empty:

                or_value = float(
                    waiting_stats.iloc[0]["odds_ratio"]
                )

                st.metric(
                    "Odds Ratio",
                    f"{or_value:.2f}"
                )

        with col4:

            if not waiting_stats.empty:

                significant = bool(
                    waiting_stats.iloc[0]
                    ["significant_after_bh"]
                )

                st.metric(
                    "Significant",
                    "YES" if significant else "NO"
                )

        st.info(
            "Waiting Time is significantly associated "
            "with low ratings. This is an association, "
            "not proof of causation."
        )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with tab3:

    st.subheader(
        "ℹ️ About FoodReviewIQ"
    )

    st.markdown(
        """
### What is FoodReviewIQ?

FoodReviewIQ is an **Aspect-Aware RAG system** for
evidence-based restaurant review intelligence.

Instead of simply summarizing reviews, the system:

1. Processes Yelp restaurant reviews using NLP.
2. Discovers restaurant-related aspects dynamically.
3. Groups semantically related aspect phrases.
4. Uses an LLM to label the discovered aspect groups.
5. Performs aspect-level sentiment analysis.
6. Compares aspects across low and high ratings.
7. Tests rating associations statistically.
8. Stores representative reviews as vector embeddings.
9. Retrieves relevant evidence using semantic search.
10. Generates grounded answers using an LLM.

### Dataset

**50,000 Yelp restaurant reviews**

### RAG database

**5,000 representative reviews**

### Embedding model

**BAAI/bge-small-en-v1.5**

### Vector database

**ChromaDB**

### LLM

**Groq — openai/gpt-oss-20b**

### Statistical analysis

- Log-odds
- Odds ratios
- 95% confidence intervals
- Chi-square testing
- Fisher's exact test when appropriate
- Benjamini-Hochberg multiple-testing correction

### Evaluation

- RAG Precision@5: **95%**
- Quantitative aspect coverage: **100%**
- Combined evidence Precision@5: **100%**
"""
    )

    st.success(
        "FoodReviewIQ analytical pipeline is complete."
    )