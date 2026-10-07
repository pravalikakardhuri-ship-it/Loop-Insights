# LOOP Insights – AI Customer Feedback Intelligence Platform
   Live demo: https://your-app-link.streamlit.app

**Listen. Understand. Improve.**

> Status: Phases 1-11 complete. See
> [Development Phases](#development-phases) for what's built and what's next.

## Project Overview

LOOP Insights is an end-to-end platform that turns raw customer feedback into
actionable business intelligence: upload feedback, clean and store it, run
sentiment/emotion/topic analysis, explore it on an interactive dashboard,
and generate AI summaries, recommendations, and answers via the Groq LLM.

## Problem Statement

Businesses collect large volumes of customer feedback but rarely have the
tooling to systematically turn it into insight -- most feedback is read
manually, if at all, and recurring problems go unnoticed until they show up
in churn or revenue.

## Proposed Solution

An AI-powered Streamlit application that automates the pipeline:
**Collect → Clean → Analyze → Store → Visualize → Summarize → Recommend.**

## Features

- CSV/Excel upload with column mapping and validation
- Manual single-feedback entry and analysis
- ML-based sentiment classification (TF-IDF + Logistic Regression)
- Emotion detection and topic/issue classification
- Recurring complaint detection with semantic grouping
- Interactive Plotly dashboard with filters
- Custom LOOP Customer Satisfaction Score
- Groq-powered AI summaries and prioritized recommendations
- LOOP Insight Assistant (data-grounded Q&A, optional RAG via FAISS)
- Product-level analysis and exportable reports (CSV/PDF)

## Technology Stack

| Layer          | Technology                                   |
|----------------|-----------------------------------------------|
| UI             | Streamlit                                      |
| Data           | Pandas, NumPy                                  |
| ML             | Scikit-learn                                   |
| NLP            | NLTK / spaCy                                   |
| Generative AI  | Groq API (LLM)                                 |
| Database       | MySQL (via SQLAlchemy)                         |
| Visualization  | Plotly                                         |
| Optional RAG   | Sentence Transformers + FAISS                  |

## Architecture

```text
User
 ↓
Upload CSV / Excel
 ↓
Preview Feedback
 ↓
Validate & Clean Data
 ↓
Store Data in MySQL
 ↓
NLP Processing (Sentiment / Emotion / Topic)
 ↓
Analytics
 ↓
Interactive Dashboard
 ↓
Groq LLM (Summary / Recommendations / Assistant)
```

## Project Structure

```text
project-loop/
├── app.py                  # Streamlit entry point, sidebar navigation, routing
├── config.py                # Central config loaded from .env
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── sample_feedback.csv   # (added in Phase 3)
│   └── processed/
├── models/
│   ├── sentiment_model.pkl   # (added in Phase 4)
│   └── tfidf_vectorizer.pkl
├── services/                 # preprocessing, sentiment, emotion, topic_model,
│                              # analytics, llm_service, assistant_context, report_service
├── database/                 # models.py, queries.py (Phase 2)
├── training/                 # train_sentiment.py, evaluate_model.py (Phase 4)
├── utils/                    # validators.py, helpers.py
└── tests/                    # pytest suite
```

## Installation

```bash
# 1. Clone / open the project folder
cd project-loop

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## MySQL Setup

> Full setup and table creation land in **Phase 2 - Database**. For now:

1. Install MySQL locally or use a hosted instance.
2. Create a database, e.g.:
   ```sql
   CREATE DATABASE project_loop;
   ```
3. Fill in the `MYSQL_*` values in your `.env` (see below).

## Environment Variables

Copy the example file and fill in your own values -- never commit `.env`:

```bash
cp .env.example .env
```

```text
GROQ_API_KEY=
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=project_loop
```

## LOOP Insight Assistant (Phase 8)

The chatbot (`ui/loop_assistant.py`) follows the spec's exact diagram: question
→ retrieve relevant data (`services/assistant_context.py`) → structured
context → Groq → answer. Every question gets overall KPIs, a per-product
breakdown, and the top complaints; questions with matching keywords also
pull relevant individual feedback snippets as supporting evidence.

**On the retrieval approach:** this uses simple keyword-overlap matching
to find relevant feedback snippets, not embedding-based search (the spec
marks FAISS/embeddings as optional, section 23). It's a reasonable
scope tradeoff, but it does mean a snippet about delivery that never uses
the literal word "delivery" (e.g. "arrived a week late") won't be pulled
in via the snippet search — though the topic-classified aggregate stats
(powered by the real ML topic model, not keyword matching) still catch
it correctly either way. Worth knowing, and an easy documented upgrade
path if you want to add real embedding search later.

## Reports (Phase 9)

The **Reports** page (`ui/reports.py`, logic in `services/report_service.py`)
offers:

- **CSV exports** - feedback with analysis, per-product ratings, and top
  recurring complaints (UTF-8 with BOM so Excel opens them correctly).
- **PDF report** - key metrics, sentiment chart, product ratings, top
  complaints, and optionally the Groq summary and recommendations.
- **Report history** - AI summary + recommendations can be saved to the
  `reports` table and re-downloaded as PDF later.

Product and sentiment filters apply to the exports, the PDF, and the
complaints table, so everything in one report covers the same rows.

Two things worth knowing:

- A saved report stores only the AI text, not a snapshot of the numbers. Its
  PDF therefore has no statistics sections (it would otherwise show today's
  data under an old date). Generate a fresh PDF for current numbers.
- As in Phase 7, the Groq calls have not been tested against the real API
  from the development environment. If Groq fails, the PDF is still built
  without the AI sections and a warning explains why.

## Groq AI (Phase 7)

**AI Summary and AI Recommendations** (`services/llm_service.py`, surfaced
on the **AI Insights** page and in **Product Analysis**) call the Groq API,
using the `openai/gpt-oss-120b` model by default (Groq deprecated
`llama-3.3-70b-versatile` in June 2026 -- override `GROQ_MODEL` in `.env`
if Groq's lineup changes again; check https://console.groq.com/docs/models).

**The core rule enforced throughout:** the LLM only ever receives numbers
that were already calculated by this app's own analytics code -- it
summarizes and prioritizes, it never estimates or invents a statistic.
Recommendation responses are requested as structured JSON and parsed
defensively; if Groq returns something that doesn't match the expected
shape, the app shows an error rather than displaying garbage.

**Important limitation to know about:** this development environment has
no network access to `api.groq.com`, so `services/llm_service.py` has
**not been tested against the real Groq API** -- only against a mocked
client (see `tests/test_llm_service.py`) that verifies the request shape
and response parsing logic. Test it for real with your own `GROQ_API_KEY`
once the app is running -- **Settings → Test Groq Connection** is the
fastest way to check it's working.

## Model Training

The sentiment model (TF-IDF + Logistic Regression) is trained on
`data/sample_feedback.csv`, with labels bootstrapped from star ratings
(4-5 → Positive, 3 → Neutral, 1-2 → Negative). A pretrained model is
already included at `models/sentiment_model.pkl` / `models/tfidf_vectorizer.pkl`,
so the app works out of the box -- retrain any time with:

```bash
python -m training.train_sentiment
```

This prints accuracy, precision, recall, F1, and a confusion matrix, and
writes them to `models/metrics.json`. To re-evaluate the currently saved
model without retraining:

```bash
python -m training.evaluate_model
```

**On the metrics:** because the dataset is built from a fixed set of
sentence templates, the train/test split groups by exact feedback text so
no identical sentence leaks between the two sets (see the comments in
`training/train_sentiment.py`). The same sentence template appears once per product, so the split also
holds out whole templates (the product name is stripped before grouping): every
test sentence uses wording the model never saw in training. On the current
dataset this gives ~74% sentiment accuracy and ~54% topic accuracy. These are
deliberately honest numbers -- an earlier split that only separated identical
sentences reported ~100%, which was template memorisation. Real, varied
customer writing would need more diverse training data (ideally real reviews). Swap in a larger,
more diverse (ideally real) dataset before treating this as production-grade.

## Emotion & Topic Detection (Phase 5)

**Emotion detection** (`services/emotion.py`) uses a documented keyword
lexicon rather than a trained classifier -- there's no ground-truth emotion
label available for feedback text, so a transparent, inspectable baseline
is the honest choice. It's built so a transformer-based model can be
swapped in later without touching any calling code (every caller just
uses `predict_emotion(text) -> (label, confidence)`).

**Topic/issue detection** (`services/topic_model.py`, trained via
`training/train_topic.py`) uses the TF-IDF + classification approach the
spec recommends. There's no ground-truth topic label either, so training
labels are bootstrapped from a keyword lexicon (see `weak_label_topic()`
-- deliberately named "weak" to flag that it's heuristic, not ground
truth). **Worth knowing before trusting the topic model's accuracy:**
because the training labels themselves come from keyword matching, and
TF-IDF is also keyword-based, the accuracy only measures how well the model
reproduces those keyword rules on unseen sentence templates (~54% on the
current dataset) -- a different (weaker) kind of validation than the
sentiment model's accuracy. This caveat is also shown directly in the Settings page next to
the topic model's metrics, not just buried here.

**Recurring complaint detection** (`database.queries.get_top_complaints`)
aggregates negative feedback by topic. The "group similarly-worded
complaints together" requirement from the spec is satisfied by the topic
*classifier* itself (it maps differently-phrased complaints about the same
issue to the same topic label) -- the aggregation query on top of that is
then a straightforward group-by.

Retrain the topic model any time with:

```bash
python -m training.train_topic
```

## Running the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with the sidebar navigation
(Home, Dashboard, Upload Feedback, Manual Feedback, Feedback Explorer,
AI Insights, LOOP Assistant, Product Analysis, Reports, Settings).

## Screenshots

> Added once the Dashboard (Phase 6) and Reports (Phase 9) pages are built.

## Sample Dataset

A 3,000-row synthetic dataset is included at `data/sample_feedback.csv`,
covering 26 products in 7 categories (electronics, accessories, wearables,
fashion, home & kitchen, beauty, health & fitness) and a realistic mix of
positive (~55%), negative (~25%), and neutral (~20%) feedback. The product
catalog lives in `config.py` (`PRODUCT_CATALOG`) and feeds both the data
generator and the Manual Feedback form, so adding a product is one line. Regenerate it anytime
with:

```bash
python data/generate_sample_data.py
```

## Future Enhancements

- Multilingual sentiment analysis
- Voice feedback
- Real-time review monitoring
- Social media feedback integration
- Email alerts for critical complaints
- Customer churn prediction
- Advanced RAG
- Automated business reports
- Real-time analytics
- Feedback trend forecasting

## Development Phases

| Phase | Scope                                   | Status        |
|-------|------------------------------------------|---------------|
| 1     | Project setup & Streamlit architecture   | ✅ Done       |
| 2     | Database (MySQL, tables, CRUD)           | ✅ Done       |
| 3     | Data pipeline (sample data, upload, clean)| ✅ Done       |
| 4     | Machine learning (sentiment model)       | ✅ Done       |
| 5     | NLP (emotion, topic, complaints)         | ✅ Done       |
| 6     | Dashboard (KPIs, charts, filters)        | ✅ Done       |
| 7     | Groq AI (summary, recommendations)       | ✅ Done       |
| 8     | LOOP Assistant (Q&A, optional RAG)       | ✅ Done       |
| 9     | Reports (PDF/CSV export)                 | ✅ Done       |
| 10    | Testing                                  | ✅ Done       |
| 11    | Final UI polish                          | ✅ Done       |
| 12    | Deployment prep                          | ⬜            |

## Testing (Phase 10)

Run the automated suite from the project folder:

```bash
python -m pytest -q
```

It covers preprocessing, sentiment/emotion/topic models, analytics (including
empty data, a single product, and all-negative feedback), validators, file
helpers, reports, and the assistant. The real Groq API and a click-through of every page were also checked manually.

## Final UI Polish (Phase 11)

- One shared empty state (`no_data_state`) and one shared database error
  (`load_error`) in `ui/theme.py`, so every page says the same thing and offers
  a button for the next step.
- Visible keyboard focus, reduced-motion support, and a mobile layout for the
  header, hero and cards.
- Home page explains what to do when MySQL has no data yet.
- Cleaned up `app.py` (removed the unused Phase 1 placeholder code).

## Time Periods and Comparison

New uploads are added to the existing data, so the database keeps the full
history. Choose what to look at with the period buttons at the top of
Dashboard, Product Analysis, AI Insights and Reports:
Last 7 days, Last 30 days, Last 90 days, Last 1 year, All time, or Custom dates.

- "Last N days" counts back from the **newest feedback date in the data**, not
  from today's date.
- On the Dashboard, each number shows its change against the period just before
  it (same length). For example, Last 7 days is compared with the 7 days before
  that. Negative feedback going up is shown in red.
- No comparison is shown for All time, or when the previous period has no data.
- Reports and PDFs use the selected period, and the PDF lists it under Scope.
- Feedback with no date is kept in the normal view but left out of comparisons.
- This measures customer feedback (ratings, sentiment, complaints), not sales.

## Admin Access to Settings

Clients see every page except **Settings**. The admin opens the **Admin** box at
the bottom of the sidebar, enters the password, and Settings appears. Click
**Lock admin** to hide it again.

- Set `ADMIN_PASSWORD` in `.env` (copy `.env.example`) or in the hosting
  platform's secrets. Use at least 10 characters. If it is empty, nobody can
  open Settings.
- After 5 wrong attempts the box locks for 60 seconds.
- The Settings page checks admin status again itself, so changing the page name
  in the browser does not open it.
- Setup notices on the Home page (missing keys, empty passwords) are shown only
  to the admin.

## What Clients See When Something Fails

The Groq key and database password live on the server (`.env` or the host's
secrets). Clients never set or test them. If Groq or MySQL has a problem,
clients see a short plain message ("AI features are unavailable right now.
Please try again later."). Technical reasons, file names and Settings hints are
shown only when the admin has unlocked admin mode. After deployment, test the
Groq connection in Settings once, and again only if you change the key.
