"""
tests/test_database.py
-----------------------
Tests for database insert/retrieve operations (spec section 30: Database).

Uses an isolated in-memory SQLite database instead of MySQL so these tests
run anywhere with no external services required. This only validates the
ORM/CRUD logic in database/models.py and database/queries.py -- it does not
replace testing against a real MySQL instance before deployment.
"""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database
from database.models import Base, init_db
from database import queries


@pytest.fixture(autouse=True)
def sqlite_db():
    """Point the database module at a fresh in-memory SQLite DB for each test."""
    test_engine = create_engine("sqlite:///:memory:")
    database.engine = test_engine
    database.SessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    init_db()
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_insert_and_get_feedback():
    feedback_id = queries.insert_feedback(
        customer_id="C001",
        product="Laptop",
        category="Electronics",
        rating=2,
        feedback_text="The product is good but delivery was very late.",
        customer_name="Rahul",
        feedback_date=date(2026, 7, 15),
    )
    assert feedback_id is not None

    row = queries.get_feedback_by_id(feedback_id)
    assert row["customer_id"] == "C001"
    assert row["product"] == "Laptop"
    assert row["rating"] == 2


def test_bulk_insert_feedback():
    ids = queries.bulk_insert_feedback(
        [
            {"customer_id": "C002", "product": "Phone", "category": "Mobile", "rating": 5, "feedback_text": "Great!"},
            {"customer_id": "C003", "product": "Phone", "category": "Mobile", "rating": 1, "feedback_text": "Bad support."},
        ]
    )
    assert len(ids) == 2
    assert queries.count_feedback() == 2


def test_insert_and_get_analysis():
    feedback_id = queries.insert_feedback(
        customer_id="C001", product="Laptop", category="Electronics",
        rating=2, feedback_text="Late delivery.",
    )
    queries.insert_analysis(
        feedback_id=feedback_id,
        sentiment="Negative",
        sentiment_confidence=0.91,
        emotion="Frustrated",
        emotion_confidence=0.88,
        topic="Delivery",
        processed_text="late delivery",
    )
    analysis = queries.get_analysis_by_feedback_id(feedback_id)
    assert analysis["sentiment"] == "Negative"
    assert analysis["topic"] == "Delivery"


def test_analysis_upsert_replaces_existing_row():
    feedback_id = queries.insert_feedback(
        customer_id="C001", product="Laptop", category="Electronics",
        rating=2, feedback_text="Late delivery.",
    )
    queries.insert_analysis(feedback_id=feedback_id, sentiment="Negative", topic="Delivery")
    queries.insert_analysis(feedback_id=feedback_id, sentiment="Positive", topic="Product Quality")

    analysis = queries.get_analysis_by_feedback_id(feedback_id)
    assert analysis["sentiment"] == "Positive"
    assert analysis["topic"] == "Product Quality"


def test_get_feedback_with_analysis_join():
    feedback_id = queries.insert_feedback(
        customer_id="C001", product="Laptop", category="Electronics",
        rating=2, feedback_text="Late delivery.",
    )
    queries.insert_analysis(feedback_id=feedback_id, sentiment="Negative", topic="Delivery")

    results = queries.get_feedback_with_analysis()
    assert len(results) == 1
    assert results[0]["sentiment"] == "Negative"
    assert results[0]["topic"] == "Delivery"


def test_filter_feedback_by_product():
    queries.bulk_insert_feedback(
        [
            {"customer_id": "C002", "product": "Phone", "category": "Mobile", "rating": 5, "feedback_text": "Great!"},
            {"customer_id": "C004", "product": "Laptop", "category": "Electronics", "rating": 3, "feedback_text": "OK."},
        ]
    )
    phones = queries.get_all_feedback(product="Phone")
    assert len(phones) == 1
    assert phones[0]["product"] == "Phone"


def test_insert_and_get_report():
    report_id = queries.insert_report(
        title="Weekly Report", summary="Customers are mostly satisfied.",
        recommendations="Improve delivery times.",
    )
    report = queries.get_report_by_id(report_id)
    assert report["title"] == "Weekly Report"

    all_reports = queries.get_all_reports()
    assert len(all_reports) == 1
