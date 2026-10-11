
import re


# ============================================================
# QUANTITATIVE QUESTION KEYWORDS
# ============================================================

QUANTITATIVE_KEYWORDS = [

    "which aspects",
    "what aspects",
    "most associated",
    "associated with",
    "associated with bad",
    "associated with low",
    "associated with high",
    "low ratings",
    "low-rated",
    "low rated",
    "high ratings",
    "high-rated",
    "high rated",
    "bad ratings",
    "good ratings",
    "rating patterns",
    "rating pattern",
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


# ============================================================
# COMBINED QUESTION KEYWORDS
# ============================================================

COMBINED_KEYWORDS = [

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


# ============================================================
# DETECT QUANTITATIVE INTENT
# ============================================================

def detect_question_type(question):

    question_lower = question.lower().strip()

    # --------------------------------------------------------
    # Combined questions first
    # --------------------------------------------------------

    has_combined_keyword = any(
        keyword in question_lower
        for keyword in COMBINED_KEYWORDS
    )

    has_quantitative_keyword = any(
        keyword in question_lower
        for keyword in QUANTITATIVE_KEYWORDS
    )

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    if (
        has_combined_keyword
        and has_quantitative_keyword
    ):

        return "combined"

    # --------------------------------------------------------
    # Quantitative
    # --------------------------------------------------------

    if has_quantitative_keyword:

        return "quantitative"

    # --------------------------------------------------------
    # Normal RAG
    # --------------------------------------------------------

    return "rag"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [

        "What do customers say about waiting times?",

        "What do customers think about food quality?",

        "Which aspects are associated with bad ratings?",

        "What aspects are associated with low ratings?",

        "Which aspects are associated with high ratings?",

        "Why is waiting time associated with bad ratings?",

        "Explain why service quality is associated with low ratings."

    ]

    for question in test_questions:

        result = detect_question_type(
            question
        )

        print(
            f"{result.upper():12} | {question}"
        )

