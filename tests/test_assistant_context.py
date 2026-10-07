"""
tests/test_assistant_context.py
----------------------------------
Tests for services/assistant_context.py -- the retrieval/context-building
step of the LOOP Insight Assistant (the part that can be tested without
a live Groq API call).
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database
from database.models import Base, init_db
from database import queries
from services.assistant_context import build_assistant_context, _retrieve_relevant_snippets


@pytest.fixture(autouse=True)
def sqlite_db():
    test_engine = create_engine("sqlite:///:memory:")
    database.engine = test_engine
    database.SessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    init_db()
    yield
    Base.metadata.drop_all(bind=test_engine)


def _seed(product, sentiment, topic, text, rating=3):
    fid = queries.insert_feedback(
        customer_id="C001", product=product, category="Electronics",
        rating=rating, feedback_text=text,
    )
    queries.insert_analysis(feedback_id=fid, sentiment=sentiment, topic=topic)


def test_build_context_returns_none_for_empty_data():
    empty_df = pd.DataFrame()
    assert build_assistant_context(empty_df, "What are the complaints?") is None


def test_build_context_includes_kpis():
    _seed("Laptop", "Negative", "Delivery", "Late delivery, very annoyed.", rating=1)
    _seed("Laptop", "Positive", "Product Quality", "Great laptop, very happy.", rating=5)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "How many total feedback records are there?")

    assert context["total_feedback"] == 2
    assert context["positive_count"] == 1
    assert context["negative_count"] == 1


def test_build_context_includes_product_breakdown():
    _seed("Phone", "Negative", "Delivery", "Order arrived very late.", rating=1)
    _seed("Phone", "Negative", "Delivery", "Package took forever to arrive.", rating=2)
    _seed("Laptop", "Positive", "Product Quality", "Excellent build quality.", rating=5)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "Which product has the most negative feedback?")

    product_text = " ".join(context["feedback_by_product"])
    assert "Phone" in product_text
    assert "Laptop" in product_text
    assert "100.0% negative" in product_text  # Phone: 2/2 negative


def test_build_context_precomputes_overall_percentages():
    # Regression test: percentages must be pre-computed in Python, not
    # left for the LLM to derive from raw counts -- language models are
    # unreliable at arithmetic, which was part of why answers were
    # sometimes inaccurate.
    _seed("Phone", "Negative", "Delivery", "Order arrived very late.", rating=1)
    _seed("Phone", "Positive", "Product Quality", "Works great.", rating=5)
    _seed("Phone", "Positive", "Product Quality", "Really happy with it.", rating=4)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "What percentage of feedback is negative?")

    assert context["negative_pct"] == round(1 / 3 * 100, 1)
    assert context["positive_pct"] == round(2 / 3 * 100, 1)


def test_build_context_includes_highest_and_lowest_rated_product():
    _seed("Phone", "Negative", "Product Quality", "Terrible, broke immediately.", rating=1)
    _seed("Laptop", "Positive", "Product Quality", "Excellent laptop.", rating=5)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "Which product has the best satisfaction?")

    assert "Laptop" in context["highest_rated_product"]
    assert "Phone" in context["lowest_rated_product"]


def test_build_context_includes_emotion_breakdown():
    fid = queries.insert_feedback(customer_id="C001", product="Camera", category="Electronics", rating=1, feedback_text="Support never responded.")
    queries.insert_analysis(feedback_id=fid, sentiment="Negative", emotion="Frustrated", topic="Customer Support")

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "Why are customers unhappy?")

    assert any("Frustrated" in e for e in context["emotion_breakdown"])


def test_build_context_focuses_on_mentioned_product():
    _seed("Smartwatch", "Negative", "Product Quality", "Screen stopped working.", rating=1)
    _seed("Laptop", "Positive", "Product Quality", "Great laptop.", rating=5)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "What are customers saying about the Smartwatch?")

    assert "focused_product_stats" in context
    assert context["focused_product_stats"]["product"] == "Smartwatch"
    assert context["focused_product_stats"]["total_feedback"] == 1
    assert context["focused_product_stats"]["negative_pct"] == 100.0


def test_build_context_no_focus_block_when_no_product_named():
    _seed("Laptop", "Positive", "Product Quality", "Great laptop.", rating=5)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "What are our biggest complaints?")

    assert "focused_product_stats" not in context


def test_build_context_includes_top_complaints():
    _seed("Camera", "Negative", "Refund", "Requested a refund, no response yet.", rating=1)
    _seed("Camera", "Negative", "Refund", "Still waiting on my refund request.", rating=1)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "What are the biggest complaints?")

    assert any("Refund" in c for c in context["top_complaints_overall"])


def test_retrieve_relevant_snippets_matches_keyword():
    df = pd.DataFrame(
        [
            {"product": "Laptop", "sentiment": "Negative", "topic": "Delivery", "feedback_text": "Delivery was very late and frustrating."},
            {"product": "Phone", "sentiment": "Positive", "topic": "Product Quality", "feedback_text": "Great camera quality on this phone."},
        ]
    )
    snippets = _retrieve_relevant_snippets(df, "What are customers saying about delivery?")
    assert len(snippets) == 1
    assert snippets[0]["product"] == "Laptop"


def test_retrieve_relevant_snippets_returns_empty_when_no_match():
    df = pd.DataFrame(
        [{"product": "Laptop", "sentiment": "Positive", "topic": "Product Quality", "feedback_text": "Great build quality overall."}]
    )
    snippets = _retrieve_relevant_snippets(df, "zzz nonexistent keyword xyz")
    assert snippets == []


def test_build_context_includes_snippets_when_relevant():
    _seed("Laptop", "Negative", "Delivery", "The delivery took way too long to arrive.", rating=1)

    from services.analytics import load_feedback_df
    df = load_feedback_df()
    context = build_assistant_context(df, "What are customers saying about delivery?")

    assert "relevant_feedback_snippets" in context
    assert any("delivery" in s.lower() for s in context["relevant_feedback_snippets"])
