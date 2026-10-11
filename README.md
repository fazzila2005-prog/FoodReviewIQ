# FoodReviewIQ: Aspect-Aware Restaurant Review Intelligence

**FoodReviewIQ** is an NLP-powered restaurant review analysis system that combines aspect-based sentiment analysis, statistical analysis, and Retrieval-Augmented Generation (RAG) to help users understand customer opinions using real restaurant reviews.

Instead of simply summarizing reviews, FoodReviewIQ identifies specific restaurant aspects, analyses their association with customer ratings, and retrieves relevant reviews to provide evidence-based answers.

## Project Overview

Restaurant reviews contain valuable information about food quality, pricing, service, waiting time, ambience, and other aspects of the dining experience. However, manually analysing thousands of reviews to identify recurring complaints and positive experiences can be time-consuming.

FoodReviewIQ addresses this problem by processing restaurant reviews, discovering meaningful aspects, analysing their sentiment and relationship with ratings, and retrieving relevant customer feedback to answer natural-language questions.

## Key Features

* **Aspect Discovery:** Identifies and organises restaurant-related aspects, including food quality, waiting time, pricing, service, seating, and ambience.
* **Aspect-Level Sentiment Analysis:** Associates review mentions with restaurant aspects and classifies their sentiment as positive, neutral, or negative.
* **Statistical Rating Analysis:** Compares aspects mentioned in low-rated and high-rated reviews using log-odds and statistical significance testing.
* **Evidence-Based RAG:** Retrieves relevant customer reviews from a vector database and uses them as evidence when generating answers.
* **Combined Analysis:** Connects statistical findings with retrieved customer reviews to help explain observed rating patterns.
* **Interactive Streamlit Interface:** Provides a user interface for asking questions, exploring rating insights, and learning about the project.

## Technology Stack

| Component                   | Technology                                    |
| --------------------------- | --------------------------------------------- |
| Programming language        | Python                                        |
| Data processing             | Pandas, NumPy                                 |
| Natural Language Processing | NLTK                                          |
| Embeddings                  | BAAI/bge-small-en-v1.5                        |
| Vector database             | ChromaDB                                      |
| Large Language Model        | Groq API                                      |
| Statistical analysis        | Log-odds and statistical significance testing |
| Application interface       | Streamlit                                     |

## Dataset

FoodReviewIQ uses the official Yelp Open Dataset.

The project initially extracted restaurant businesses and selected 50,000 restaurant reviews for processing. A representative subset of 5,000 reviews was subsequently selected for the RAG retrieval pipeline.

The processed reviews cover multiple restaurants and aspects of the customer experience.

The raw dataset is not included in this repository. Users must obtain it through the official Yelp Open Dataset source and follow its applicable terms.

## Methodology

1. **Data extraction:** Identify restaurant businesses and extract a working sample of customer reviews.
2. **Text preprocessing:** Clean review text and perform tokenisation, stop-word removal, stemming, lemmatisation, and part-of-speech tagging.
3. **Aspect discovery:** Identify recurring aspect-related phrases, group contextual variations, and organise them into a structured aspect taxonomy.
4. **Aspect sentiment analysis:** Associate review mentions with discovered aspects and assign sentiment labels.
5. **Statistical analysis:** Compare aspect mention patterns across low-rated reviews (1–2 stars) and high-rated reviews (4–5 stars). Three-star reviews are excluded from this comparison.
6. **RAG preparation:** Prepare review documents, generate embeddings, and store the selected review subset in ChromaDB.
7. **Question answering:** Retrieve relevant reviews, combine the evidence with quantitative findings when appropriate, and generate natural-language responses.
8. **Evaluation:** Assess retrieval relevance, aspect coverage, and selected quantitative outputs using project-specific tests.

## Statistical Analysis

The quantitative component investigates which aspects are more strongly associated with low or high ratings.

It uses log-odds to compare aspect mention patterns between rating groups and statistical significance testing with multiple-testing correction.

These results indicate associations, not proof of causation. For example, an association between waiting time and low ratings does not establish that waiting time alone caused a customer's rating.

## Evaluation

Initial project-specific checks reported:

* **RAG Precision@5:** 95% average across three test queries.
* **Quantitative aspect coverage:** 100% on the implemented evaluation checks.
* **Combined-analysis evidence Precision@5:** 100% on the implemented evaluation checks.

These are preliminary results from a limited, project-specific evaluation set. They should not be interpreted as general model accuracy or as a guarantee of performance on unseen questions.

## Project Structure

```text
FoodReviewIQ/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma_db/
├── docs/
├── evaluation/
├── src/
│   ├── preprocess_reviews.py
│   ├── build_aspect_sentiment.py
│   ├── rating_log_odds.py
│   ├── build_rag_documents.py
│   ├── select_rag_reviews.py
│   ├── build_vector_db.py
│   ├── rag_query.py
│   ├── quantitative_analysis.py
│   ├── question_router.py
│   ├── combined_analysis.py
│   ├── integrated_rag.py
│   └── evaluation.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

*The structure above represents the intended project layout. Adjust filenames if your local project differs.*

## Setup and Installation

### Prerequisites

* Python 3.10
* Git
* A Groq API key
* The Yelp Open Dataset
* The project's required Python packages

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/FoodReviewIQ.git
cd FoodReviewIQ
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your real API key to GitHub.

### 5. Prepare the data

Download the Yelp Open Dataset from its official source. Place the relevant files in the expected raw-data directory, then run the project's preprocessing and analysis scripts in the appropriate order.

The processed data, taxonomy files, statistical outputs, and vector database must be generated or supplied separately if they are not included in the repository.

### 6. Launch the application

```bash
streamlit run app/app.py
```

Follow any additional configuration instructions required by the current version of the code.

## Limitations

* Results depend on the coverage and quality of the available Yelp reviews.
* Aspect discovery and sentiment classification may produce incorrect or incomplete labels.
* Retrieved evidence can be incomplete or imperfectly matched to a question.
* LLM-generated answers may contain errors and should be checked against the cited review evidence.
* Statistical associations do not establish causation.
* Evaluation results are based on a limited set of project-specific tests.

## Future Improvements

* Expand and strengthen the manually validated evaluation dataset.
* Improve aspect and sentiment classification using labelled examples.
* Add more rigorous retrieval and answer-quality evaluation.
* Improve source attribution and evidence verification.
* Support additional restaurant datasets and more advanced filtering.

## Acknowledgements

* **Yelp Open Dataset** for the restaurant review data.
* **NLTK** for natural-language processing tools.
* **ChromaDB** for vector storage and retrieval.
* **BAAI** for the BGE embedding model.
* **Groq** for LLM inference.
* **Streamlit** for the interactive application framework.

---

**FoodReviewIQ** — Turning restaurant reviews into structured insights and evidence-based answers.
