import json
import os
import re
from pathlib import Path

import altair as alt
import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS (works wherever the project folder lives)
# ============================================================


def find_root():
    here = Path(__file__).resolve().parent
    for c in (here, here.parent, Path(r"C:\FoodReviewIQ")):
        if (c / "data" / "chroma_db").exists():
            return c
    return here


ROOT = find_root()
CHROMA_PATH = str(ROOT / "data" / "chroma_db")
QUANT_PATH = ROOT / "data" / "processed" / "rating_log_odds.csv"
STATS_PATH = ROOT / "data" / "processed" / "statistical_significance_results.csv"
EVAL_PATH = ROOT / "data" / "processed" / "evaluation_results.json"  # optional
load_dotenv(ROOT / ".env")

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="FoodReviewIQ", page_icon="🍽️", layout="wide")

GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 8
MIN_REVIEWS = 100

# ============================================================
# LOADERS
# ============================================================


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name="foodreviewiq_reviews")


@st.cache_resource
def load_groq():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


@st.cache_data
def load_data():
    quant = pd.read_csv(QUANT_PATH)
    stats = pd.read_csv(STATS_PATH)
    # keep only columns from stats that quant doesn't already have
    extra = ["analysis_aspect"] + [
        c for c in stats.columns if c not in quant.columns
    ]
    merged = quant.merge(stats[extra], on="analysis_aspect", how="left")
    if "significant_after_bh" in merged:
        merged["significant_after_bh"] = (
            merged["significant_after_bh"].fillna(False).astype(bool)
        )
    else:
        merged["significant_after_bh"] = False
    return quant, stats, merged


@st.cache_data
def load_eval():
    if EVAL_PATH.exists():
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return None


embedding_model = load_embedding_model()
collection = load_chroma()
groq_client = load_groq()
quant_df, stats_df, merged_df = load_data()
eval_results = load_eval()

CI_COLS = [
    c for c in merged_df.columns
    if "ci" in c.lower() and c not in ("analysis_aspect",)
]

# ============================================================
# QUESTION ROUTING
# ============================================================

COMBINED_KW = [
    "why is", "why are", "why does", "why do", "explain why",
    "what explains", "explain the association", "why associated",
    "why linked", "what causes", "what might explain",
]
QUANT_KW = [
    "which aspects", "what aspects", "most associated", "associated with",
    "low ratings", "low-rated", "low rated", "bad ratings", "high ratings",
    "high-rated", "high rated", "good ratings", "rating patterns",
    "log odds", "log-odds", "statistically", "strongest association",
    "top aspects", "main drivers", "main factors", "factors behind",
    "compare aspect",
]


def detect_question_type(question):
    q = question.lower()
    c = [k for k in COMBINED_KW if k in q]
    n = [k for k in QUANT_KW if k in q]
    if c and n:
        return "COMBINED", f"matched {c[0]!r} + {n[0]!r}"
    if n:
        return "QUANTITATIVE", f"matched {n[0]!r}"
    return "RAG", "no rating-association keywords found"


# ============================================================
# RETRIEVAL
# ============================================================


def build_where(group, restaurant):
    conds = []
    if group != "All":
        conds.append({"rating_group": group})
    if restaurant.strip():
        conds.append({"restaurant_name": restaurant.strip()})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def retrieve_reviews(question, where=None, top_k=TOP_K):
    emb = embedding_model.encode(
        question, normalize_embeddings=True
    ).tolist()
    res = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    return [
        {
            "rank": i + 1,
            "review_id": ids[i],  # Chroma ID = Yelp review ID
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        }
        for i in range(len(ids))
    ]


# ============================================================
# QUANTITATIVE HELPERS
# ============================================================


def eligible():
    d = merged_df[merged_df["total_reviews"] >= MIN_REVIEWS]
    sig = d[d["significant_after_bh"]]
    return sig if len(sig) >= 5 else d


def top_low(n=10):
    return eligible().sort_values("log_odds_low_vs_high", ascending=False).head(n)


def top_high(n=10):
    return eligible().sort_values("log_odds_low_vs_high", ascending=True).head(n)


def get_mentioned_aspects(question):
    words = {w.rstrip("s") for w in re.findall(r"[a-z]+", question.lower())}
    hits = []
    for _, row in merged_df.iterrows():
        tokens = [t.rstrip("s") for t in re.findall(r"[a-z]+", str(row["analysis_aspect"]).lower())]
        if tokens and all(t in words for t in tokens):
            hits.append(row)
    return pd.DataFrame(hits)


def describe_row(row):
    """Plain-English sentence computed in code, so the LLM doesn't misread numbers."""
    low, high = row["low_rating_rate"], row["high_rating_rate"]
    ratio = low / high if high > 0 else float("inf")
    direction = (
        "MORE common in low-rated reviews" if row["log_odds_low_vs_high"] > 0
        else "MORE common in high-rated reviews"
    )
    sig = "statistically significant (BH-corrected)" if row["significant_after_bh"] else "NOT statistically significant after correction"
    odds = f", odds ratio={row['odds_ratio']:.2f}" if "odds_ratio" in row and pd.notna(row.get("odds_ratio")) else ""
    return (
        f"- {row['analysis_aspect']}: appears in {low:.1%} of low-rated vs "
        f"{high:.1%} of high-rated reviews ({ratio:.2f}x). {direction}. "
        f"log-odds={row['log_odds_low_vs_high']:.3f}{odds}, "
        f"{int(row['total_reviews'])} reviews, {sig}."
    )


def build_quantitative_context(question):
    parts = []
    focus = get_mentioned_aspects(question)
    if not focus.empty:
        parts.append("ASPECT(S) NAMED IN THE QUESTION:")
        parts += [describe_row(r) for _, r in focus.iterrows()]
        parts.append("")
    parts.append("TOP ASPECTS ASSOCIATED WITH LOW RATINGS:")
    parts += [describe_row(r) for _, r in top_low().iterrows()]
    parts.append("")
    parts.append("TOP ASPECTS ASSOCIATED WITH HIGH RATINGS:")
    parts += [describe_row(r) for _, r in top_high().iterrows()]
    return "\n".join(parts)


# ============================================================
# GENERATION
# ============================================================


def generate_answer(question, qtype, evidence, quant_ctx):
    if groq_client is None:
        return "Groq API key not found. Check your .env file."

    ev_text = ""
    for i, it in enumerate(evidence):
        m = it["metadata"]
        ev_text += (
            f"\n[EVIDENCE {i + 1}]\n"
            f"Review ID: {it['review_id']}\n"
            f"Restaurant: {m.get('restaurant_name', '')}\n"
            f"Review rating: {m.get('review_stars', '')}\n"
            f"Detected aspects: {m.get('aspect_summary', '')}\n"
            f"Review:\n{it['document']}\n"
        )

    prompt = f"""
You are FoodReviewIQ, an evidence-based restaurant review assistant.

Question:
{question}

Question type: {qtype}

Quantitative analysis (already interpreted; do not reinterpret the signs):
{quant_ctx if quant_ctx else "Not required."}

Retrieved review evidence:
{ev_text if ev_text else "Not required."}

Instructions:
1. Answer the question directly using only the supplied data.
2. Do not invent facts or numbers. If the data does not answer the
   question, say "The available reviews do not provide enough evidence."
3. Positive log-odds = more common in LOW-rated reviews; negative =
   more common in HIGH-rated reviews. Always report both rates.
4. Never call an aspect "rare" unless its rate is under 5%.
5. Only claim statistical significance if the data says so.
6. Association is not causation; say so for rating questions.
7. Cite reviews with their exact ID as [Review ID: <id>]. Never cite
   an ID that is not in the evidence. Quote only text that appears
   in the review.
8. If evidence is mixed, say so. Keep the answer concise.

Answer:
"""
    r = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return r.choices[0].message.content


def check_citations(answer, evidence):
    cited = set(re.findall(r"Review ID:\s*([A-Za-z0-9_\-]+)", answer or ""))
    valid = {e["review_id"] for e in evidence}
    return cited, cited - valid


# ============================================================
# UI HELPERS
# ============================================================

ICON = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


def stars(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return "?"


def aspect_tags(summary):
    out = []
    for part in str(summary).split(";"):
        part = part.strip()
        m = re.match(r"(.+)\((\w+)\)$", part)
        if m:
            out.append(f"{ICON.get(m.group(2).lower(), '⚪')} {m.group(1).strip()}")
    return "  ·  ".join(out)


def example_callback():
    if st.session_state.ex != "-- Select --":
        st.session_state.q = st.session_state.ex


for k, v in (("answer", None), ("evidence", []), ("qtype", None),
             ("reason", ""), ("q", "")):
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HEADER + SIDEBAR
# ============================================================

st.title("🍽️ FoodReviewIQ")
st.markdown(
    "### Aspect-Aware Restaurant Review Intelligence\n"
    "Evidence-based answers using RAG, dynamic aspect discovery and "
    "quantitative rating analysis."
)

with st.sidebar:
    st.header("📊 System")
    st.metric("Reviews in RAG", f"{collection.count():,}")
    st.metric("Quantitative aspects", f"{len(quant_df):,}")
    n_sig = int(merged_df["significant_after_bh"].sum())
    st.metric("Significant aspects", n_sig)
    st.caption(
        "Statistics come from all 50,000 reviews; RAG searches a "
        "5,000-review representative sample."
    )
    if groq_client is None:
        st.error("Groq API key missing (.env)")
    else:
        st.success("Groq connected")

    st.divider()
    st.subheader("🔧 Retrieval filters")
    group_filter = st.selectbox("Rating group", ["All", "low", "high"])
    restaurant_filter = st.text_input("Restaurant (exact name)", "")

    st.divider()
    st.markdown(
        "**Pipeline**  \n50,000 Yelp reviews  \n↓  \nNLP + aspect discovery  \n↓  \n"
        "Aspect sentiment  \n↓  \nStatistics  \n↓  \nBGE embeddings → ChromaDB  \n↓  \n"
        "RAG / quantitative analysis  \n↓  \nGroq LLM"
    )

tab1, tab2, tab3 = st.tabs(
    ["🔎 Ask FoodReviewIQ", "📈 Rating Insights", "ℹ️ About the Project"]
)

# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.subheader("Ask a question")
    examples = [
        "What do customers think about food quality?",
        "What do customers say about waiting times?",
        "What do customers think about service?",
        "Which aspects are associated with low ratings?",
        "Which aspects are associated with high ratings?",
        "Why is waiting time associated with bad ratings?",
        "Why is service quality associated with low ratings?",
    ]
    question = st.text_input(
        "Enter your question:", key="q",
        placeholder="e.g. What do customers think about food quality?",
    )
    st.selectbox(
        "Example questions", ["-- Select --"] + examples,
        key="ex", on_change=example_callback,
    )

    if st.button("🔍 Analyze Reviews", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            qtype, reason = detect_question_type(question)
            st.session_state.qtype, st.session_state.reason = qtype, reason
            with st.spinner("Analyzing restaurant reviews..."):
                evidence, quant_ctx = [], None
                try:
                    if qtype in ("RAG", "COMBINED"):
                        evidence = retrieve_reviews(
                            question,
                            where=build_where(group_filter, restaurant_filter),
                        )
                    if qtype in ("QUANTITATIVE", "COMBINED"):
                        quant_ctx = build_quantitative_context(question)
                    if qtype in ("RAG", "COMBINED") and not evidence:
                        st.session_state.answer = (
                            "No reviews matched your filters. "
                            "Try removing the restaurant or rating filter."
                        )
                    else:
                        st.session_state.answer = generate_answer(
                            question, qtype, evidence, quant_ctx
                        )
                    st.session_state.evidence = evidence
                except Exception as e:
                    st.session_state.answer = None
                    st.error(
                        "The answer could not be generated. The Groq API may "
                        "be rate-limited or unavailable, or a filter is invalid."
                    )
                    st.caption(f"Technical message: {e}")

    if st.session_state.answer:
        st.divider()
        st.subheader("💡 FoodReviewIQ Answer")
        st.caption(
            f"Analysis type: **{st.session_state.qtype}** "
            f"({st.session_state.reason})"
        )
        st.markdown(st.session_state.answer)

        if st.session_state.evidence:
            cited, bad = check_citations(
                st.session_state.answer, st.session_state.evidence
            )
            if bad:
                st.warning(
                    f"Faithfulness check: {len(bad)} cited ID(s) are not in the "
                    f"retrieved evidence: {', '.join(sorted(bad))}"
                )
            elif cited:
                st.caption(
                    f"✅ Faithfulness check: all {len(cited)} cited review IDs "
                    "exist in the retrieved evidence."
                )

    if st.session_state.evidence:
        st.divider()
        st.subheader("📚 Retrieved Evidence")
        for item in st.session_state.evidence:
            m = item["metadata"]
            label = (
                f"#{item['rank']} — {m.get('restaurant_name', 'Unknown')} "
                f"⭐ {stars(m.get('review_stars'))}"
            )
            with st.expander(label, expanded=item["rank"] <= 3):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Review ID:** `{item['review_id']}`")
                    st.write(f"**Restaurant:** {m.get('restaurant_name', '')}")
                    st.write(f"**Location:** {m.get('city', '')}, {m.get('state', '')}")
                with c2:
                    st.write(f"**Review rating:** {stars(m.get('review_stars'))} ⭐")
                    st.write(f"**Review sentiment:** {m.get('review_sentiment', '')}")
                    st.write(f"**Similarity distance:** {item['distance']:.3f}")
                tags = aspect_tags(m.get("aspect_summary", ""))
                if tags:
                    st.markdown(f"**Aspects:** {tags}")
                st.write(item["document"])

# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.subheader("📈 What is associated with ratings?")
    st.markdown(
        "Compares **1–2 star** reviews with **4–5 star** reviews "
        "(3-star reviews excluded). These are associations, not causes."
    )
    only_sig = st.checkbox("Show only statistically significant aspects", value=True)

    base = merged_df[merged_df["total_reviews"] >= MIN_REVIEWS].copy()
    if only_sig:
        base = base[base["significant_after_bh"]]

    def display_table(df):
        cols = ["analysis_aspect", "log_odds_low_vs_high", "low_rating_rate",
                "high_rating_rate", "total_reviews"]
        if "odds_ratio" in df:
            cols.append("odds_ratio")
        cols += [c for c in CI_COLS if c in df and c not in cols]
        cols.append("significant_after_bh")
        t = df[cols].copy()
        t["low_rating_rate"] = (t["low_rating_rate"] * 100).round(1).astype(str) + "%"
        t["high_rating_rate"] = (t["high_rating_rate"] * 100).round(1).astype(str) + "%"
        t["significant_after_bh"] = t["significant_after_bh"].map({True: "✅", False: "❌"})
        return t.rename(columns={
            "analysis_aspect": "Aspect", "log_odds_low_vs_high": "Log-Odds",
            "low_rating_rate": "Low Rate", "high_rating_rate": "High Rate",
            "total_reviews": "Reviews", "odds_ratio": "Odds Ratio",
            "significant_after_bh": "Sig.",
        })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔻 Associated with Low Ratings")
        st.dataframe(
            display_table(base.sort_values("log_odds_low_vs_high", ascending=False).head(10)),
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.markdown("### 🔺 Associated with High Ratings")
        st.dataframe(
            display_table(base.sort_values("log_odds_low_vs_high").head(10)),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### Log-odds by aspect")
    st.caption("Right of zero = more common in low-rated reviews. Left = more common in high-rated.")
    chart_df = base.copy()
    chart_df["Significant"] = chart_df["significant_after_bh"].map(
        {True: "Significant", False: "Not significant"}
    )
    if not chart_df.empty:
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("log_odds_low_vs_high:Q", title="Log-odds (low vs high rating)"),
                y=alt.Y("analysis_aspect:N", sort="-x", title=None),
                color=alt.Color(
                    "Significant:N",
                    scale=alt.Scale(
                        domain=["Significant", "Not significant"],
                        range=["#d62728", "#bbbbbb"],
                    ),
                ),
                tooltip=["analysis_aspect", "log_odds_low_vs_high",
                         "low_rating_rate", "high_rating_rate", "total_reviews"],
            )
            .properties(height=max(200, 22 * len(chart_df)))
        )
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("🔍 Aspect explorer")
    aspect = st.selectbox(
        "Pick an aspect", sorted(merged_df["analysis_aspect"].astype(str).unique()),
        index=0,
    )
    row = merged_df[merged_df["analysis_aspect"].astype(str) == aspect].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Low-rating rate", f"{row['low_rating_rate']:.2%}")
    m2.metric("High-rating rate", f"{row['high_rating_rate']:.2%}")
    if "odds_ratio" in row and pd.notna(row["odds_ratio"]):
        m3.metric("Odds ratio", f"{row['odds_ratio']:.2f}")
    m4.metric("Significant", "YES" if row["significant_after_bh"] else "NO")

    direction = "low" if row["log_odds_low_vs_high"] > 0 else "high"
    if row["significant_after_bh"]:
        st.info(
            f"{aspect} is significantly associated with **{direction}** ratings "
            "(Benjamini-Hochberg corrected). This is an association, not proof of causation."
        )
    else:
        st.warning(
            f"{aspect} is **not** statistically significant after multiple-testing "
            "correction, so treat its apparent direction with caution."
        )

# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.subheader("ℹ️ About FoodReviewIQ")
    st.markdown(
        """
### What is FoodReviewIQ?
An **aspect-aware RAG system** for evidence-based restaurant review intelligence.

1. Processes Yelp restaurant reviews using NLP.
2. Discovers restaurant aspects dynamically and groups related phrases.
3. Uses an LLM to label the aspect groups.
4. Performs aspect-level sentiment analysis.
5. Compares aspects across low and high ratings and tests associations statistically.
6. Stores representative reviews as embeddings in ChromaDB.
7. Retrieves evidence by semantic search (with optional metadata filters).
8. Generates grounded answers with an LLM and checks that cited review IDs exist.

### Setup
- **Dataset:** 50,000 Yelp restaurant reviews
- **RAG database:** 5,000 representative reviews
- **Embedding model:** BAAI/bge-small-en-v1.5
- **Vector database:** ChromaDB
- **LLM:** Groq, openai/gpt-oss-20b

### Statistical analysis
Log-odds, odds ratios, 95% confidence intervals, chi-square, Fisher's exact test
(when appropriate) and Benjamini-Hochberg correction.
"""
    )
    st.markdown("### Evaluation")
    if eval_results:
        st.json(eval_results)
    else:
        st.caption(
            "No evaluation file found. Save your measured results to "
            "data/processed/evaluation_results.json to display them here."
        )