"""
tests/test_complaints.py
---------------------------
Tests for database.queries.get_top_complaints (spec section 14: Recurring
Complaint Detection).
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database
from database.models import Base, init_db
from database import queries


@pytest.fixture(autouse=True)
def sqlite_db():
    test_engine = create_engine("sqlite:///:memory:")
    database.engine = test_engine
    database.SessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    init_db()
    yield
    Base.metadata.drop_all(bind=test_engine)


def _seed(product, sentiment, topic):
    fid = queries.insert_feedback(
        customer_id="C001", product=product, category="Electronics",
        rating=1 if sentiment == "Negative" else 5, feedback_text="text",
    )
    queries.insert_analysis(feedback_id=fid, sentiment=sentiment, topic=topic)
    return fid


def test_get_top_complaints_counts_and_percentages():
    _seed("Laptop", "Negative", "Delivery")
    _seed("Phone", "Negative", "Delivery")
    _seed("Phone", "Negative", "Delivery")
    _seed("Laptop", "Negative", "Product Quality")
    _seed("Laptop", "Positive", "Delivery")  # should NOT count (not negative)

    complaints = queries.get_top_complaints(limit=5)
    assert complaints[0]["topic"] == "Delivery"
    assert complaints[0]["count"] == 3
    # 3 out of 4 total negative records = 75%
    assert complaints[0]["percentage"] == 75.0
    assert complaints[1]["topic"] == "Product Quality"
    assert complaints[1]["count"] == 1


def test_get_top_complaints_related_products():
    _seed("Phone", "Negative", "Delivery")
    _seed("Phone", "Negative", "Delivery")
    _seed("Laptop", "Negative", "Delivery")

    complaints = queries.get_top_complaints(limit=5)
    assert complaints[0]["related_products"][0] == "Phone"
    assert "Laptop" in complaints[0]["related_products"]


def test_get_top_complaints_empty_when_no_negative_feedback():
    _seed("Laptop", "Positive", "Delivery")
    complaints = queries.get_top_complaints(limit=5)
    assert complaints == []


def test_get_top_complaints_respects_limit():
    _seed("A", "Negative", "Delivery")
    _seed("B", "Negative", "Refund")
    _seed("C", "Negative", "Payment")
    complaints = queries.get_top_complaints(limit=2)
    assert len(complaints) == 2


def test_get_feedback_needing_nlp_analysis_finds_missing_emotion():
    fid = queries.insert_feedback(
        customer_id="C001", product="Laptop", category="Electronics",
        rating=3, feedback_text="text",
    )
    queries.insert_analysis(feedback_id=fid, sentiment="Neutral")  # no emotion/topic yet
    pending = queries.get_feedback_needing_nlp_analysis()
    assert len(pending) == 1
    assert pending[0]["id"] == fid


def test_get_feedback_needing_nlp_analysis_excludes_fully_analyzed():
    fid = queries.insert_feedback(
        customer_id="C001", product="Laptop", category="Electronics",
        rating=3, feedback_text="text",
    )
    queries.insert_analysis(
        feedback_id=fid, sentiment="Neutral", emotion="Satisfied", topic="Product Quality"
    )
    pending = queries.get_feedback_needing_nlp_analysis()
    assert len(pending) == 0


def test_update_emotion_and_topic_preserves_sentiment():
    fid = queries.insert_feedback(
        customer_id="C001", product="Laptop", category="Electronics",
        rating=3, feedback_text="text",
    )
    queries.insert_analysis(feedback_id=fid, sentiment="Positive", sentiment_confidence=0.9)
    queries.update_emotion_and_topic(feedback_id=fid, emotion="Happy", emotion_confidence=0.8, topic="Product Quality")

    analysis = queries.get_analysis_by_feedback_id(fid)
    assert analysis["sentiment"] == "Positive"  # untouched
    assert analysis["emotion"] == "Happy"
    assert analysis["topic"] == "Product Quality"
