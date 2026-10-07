"""
tests/test_report_service.py
------------------------------
Tests for Phase 9 report generation (spec section 30: Reports).
Uses plain DataFrames, so no database is needed.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from services import report_service


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "customer_name": ["A", "B", "C", "D", "E", "F"],
            "product": ["Laptop", "Laptop", "Phone", "Phone", "Phone", "Tablet"],
            "category": ["Electronics"] * 6,
            "rating": [5, 1, 2, 4, 1, 3],
            "feedback_text": ["Great", "Late delivery", "Late delivery", "Nice", "Rude support", "Okay"],
            "feedback_date": pd.to_datetime(["2026-07-01"] * 6),
            "sentiment": ["Positive", "Negative", "Negative", "Positive", "Negative", "Neutral"],
            "sentiment_confidence": [0.9] * 6,
            "emotion": ["Happy", "Angry", "Frustrated", "Satisfied", "Angry", "Neutral"],
            "topic": ["Product Quality", "Delivery", "Delivery", "Product Quality", "Customer Support", "Other"],
        }
    )


def test_feedback_csv_has_expected_columns_and_rows(df):
    raw = report_service.feedback_to_csv(df)
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM for Excel
    text = raw.decode("utf-8-sig").splitlines()
    assert text[0].startswith("id,customer_id,customer_name,product")
    assert len(text) == 7  # header + 6 rows


def test_product_summary_csv(df):
    text = report_service.product_summary_to_csv(df).decode("utf-8-sig")
    assert "product,total_feedback,min_rating,max_rating,avg_rating" in text
    assert "Phone" in text


def test_complaints_from_df_counts_and_orders(df):
    complaints = report_service.complaints_from_df(df)
    assert complaints[0]["topic"] == "Delivery"
    assert complaints[0]["count"] == 2
    assert complaints[0]["percentage"] == pytest.approx(66.7)
    assert set(complaints[0]["related_products"]) == {"Laptop", "Phone"}
    assert complaints[1]["topic"] == "Customer Support"


def test_complaints_from_df_empty_and_no_negatives(df):
    assert report_service.complaints_from_df(df.iloc[0:0]) == []
    positives = df[df["sentiment"] == "Positive"]
    assert report_service.complaints_from_df(positives) == []


def test_complaints_to_csv_handles_empty():
    text = report_service.complaints_to_csv([]).decode("utf-8-sig")
    assert text.strip() == "topic,mentions,percent_of_negative,related_products"


def test_complaints_to_csv_rows(df):
    text = report_service.complaints_to_csv(report_service.complaints_from_df(df)).decode("utf-8-sig")
    assert "Delivery,2,66.7" in text


def test_analytics_context_matches_kpis(df):
    ctx = report_service.build_analytics_context(df, report_service.complaints_from_df(df))
    assert ctx["total_feedback"] == 6
    assert ctx["positive_pct"] == pytest.approx(33.3)
    assert ctx["negative_pct"] == pytest.approx(50.0)
    assert ctx["top_complaints"][0].startswith("Delivery")


def test_recommendations_round_trip():
    recs = [{"issue": "Late delivery", "priority": "High", "evidence": "x", "recommended_action": "y", "expected_impact": "z"}]
    assert report_service.recommendations_from_text(report_service.recommendations_to_text(recs)) == recs


@pytest.mark.parametrize("bad", [None, "", "not json", json.dumps({"a": 1}), json.dumps(["str", 1])])
def test_recommendations_from_text_is_defensive(bad):
    assert report_service.recommendations_from_text(bad) == []


def test_pdf_full_report_is_valid_pdf(df):
    pdf = report_service.generate_pdf(
        "Test Report",
        df=df,
        complaints=report_service.complaints_from_df(df),
        summary="Customers like the Laptop.",
        recommendations=[
            {"issue": "Late delivery", "priority": "High", "evidence": "66.7% of negative", "recommended_action": "Fix it", "expected_impact": "Fewer complaints"}
        ],
        filter_description="All feedback",
        generated_at=datetime(2026, 10, 3, 12, 0),
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000


def test_pdf_survives_special_characters(df):
    pdf = report_service.generate_pdf(
        "Q&A <Report>",
        df=df,
        summary="Rating < 3 & growing > 5% <b>unclosed",
        recommendations=[{"issue": "A & B <x>", "priority": "Low", "evidence": "<e>", "recommended_action": "a&b", "expected_impact": "<i>"}],
    )
    assert pdf.startswith(b"%PDF")


def test_pdf_saved_report_without_stats():
    pdf = report_service.generate_pdf("Saved", summary="Only AI text.", recommendations=[])
    assert pdf.startswith(b"%PDF")


def test_pdf_with_no_content_still_builds():
    pdf = report_service.generate_pdf("Empty", df=pd.DataFrame(), filter_description="Products: X")
    assert pdf.startswith(b"%PDF")
