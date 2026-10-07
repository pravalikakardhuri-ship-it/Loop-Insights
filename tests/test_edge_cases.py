"""Phase 10: edge-case tests -- empty data, single product, all-negative
feedback, and bad upload input."""

import pandas as pd

from services.analytics import (
    compute_kpis,
    compute_loop_satisfaction_score,
    product_rating_extremes,
    product_summary,
)
from services.preprocessing import clean_dataframe


def _df(rows):
    return pd.DataFrame(rows)


def _row(product="Laptop", rating=1, sentiment="Negative", topic="Delivery", emotion="Angry"):
    return {"product": product, "category": "Electronics", "rating": rating,
            "sentiment": sentiment, "emotion": emotion, "topic": topic,
            "feedback_date": pd.Timestamp("2026-01-01")}


# --- analytics -------------------------------------------------------------

def test_empty_dataframe_gives_zeros_not_errors():
    empty = pd.DataFrame(columns=["product", "rating", "sentiment", "emotion", "topic"])
    kpis = compute_kpis(empty)
    assert kpis["total_feedback"] == 0 and kpis["average_rating"] == 0.0
    assert compute_loop_satisfaction_score(empty)["loop_score"] == 0.0
    assert product_rating_extremes(empty).empty


def test_single_product():
    df = _df([_row(rating=5, sentiment="Positive", topic="Product Quality", emotion="Happy")])
    summary = product_summary(df, "Laptop")
    assert summary["total_feedback"] == 1
    assert summary["positive_pct"] == 100.0
    assert summary["top_complaint"] is None  # no negative feedback
    assert len(product_rating_extremes(df)) == 1


def test_all_negative_feedback():
    df = _df([_row(rating=1), _row(rating=2), _row(rating=1)])
    kpis = compute_kpis(df)
    assert kpis["negative_feedback"] == 3
    assert kpis["positive_feedback"] == 0
    score = compute_loop_satisfaction_score(df)
    assert score["sentiment_score"] == 0.0
    assert 0 <= score["loop_score"] < 30
    assert product_summary(df, "Laptop")["top_complaint"] == "Delivery"


def test_product_summary_for_unknown_product():
    df = _df([_row()])
    summary = product_summary(df, "Does Not Exist")
    assert summary["total_feedback"] == 0 and summary["top_complaint"] is None


# --- bad upload input --------------------------------------------------------

def _upload(rows):
    return pd.DataFrame(
        rows,
        columns=["customer_id", "customer_name", "product", "category",
                 "rating", "feedback", "date"],
    )


def test_blank_feedback_rows_are_dropped():
    df = _upload([
        ["C1", "A", "Laptop", "Electronics", 5, "Great laptop", "2026-01-01"],
        ["C2", "B", "Laptop", "Electronics", 4, "   ", "2026-01-02"],
        ["C3", "C", "Laptop", "Electronics", 4, "", "2026-01-03"],
    ])
    cleaned, report = clean_dataframe(df)
    assert len(cleaned) == 1
    assert report.empty_feedback_dropped == 2


def test_invalid_ratings_are_dropped():
    df = _upload([
        ["C1", "A", "Laptop", "Electronics", 9, "Fine", "2026-01-01"],
        ["C2", "B", "Laptop", "Electronics", "abc", "Fine too", "2026-01-02"],
        ["C3", "C", "Laptop", "Electronics", 3, "Okay product", "2026-01-03"],
    ])
    cleaned, report = clean_dataframe(df)
    assert len(cleaned) == 1
    assert report.invalid_ratings_dropped == 2


def test_missing_product_and_duplicates_are_dropped():
    df = _upload([
        ["C1", "A", None, "Electronics", 5, "No product here", "2026-01-01"],
        ["C2", "B", "Laptop", "Electronics", 5, "Same text", "2026-01-01"],
        ["C2", "B", "Laptop", "Electronics", 5, "Same text", "2026-01-01"],
    ])
    cleaned, report = clean_dataframe(df)
    assert len(cleaned) == 1
    assert report.missing_values_dropped == 1
    assert report.duplicate_records_dropped == 1


def test_bad_dates_do_not_drop_the_row():
    df = _upload([["C1", "A", "Laptop", "Electronics", 5, "Good one", "not a date"]])
    cleaned, _ = clean_dataframe(df)
    assert len(cleaned) == 1
