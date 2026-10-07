"""
tests/test_analytics.py
--------------------------
Tests for services/analytics.py (spec section 30: Analytics -- sentiment
percentages, average rating, satisfaction score).
"""

import pandas as pd

from services.analytics import (
    Filters,
    apply_filters,
    compute_kpis,
    compute_loop_satisfaction_score,
    sentiment_distribution,
    rating_distribution,
    product_wise_sentiment,
    product_rating_extremes,
    top_complaints_chart_data,
    product_summary,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 1, "product": "Laptop", "category": "Computers", "rating": 5, "sentiment": "Positive",
             "emotion": "Happy", "topic": "Product Quality", "feedback_date": pd.Timestamp("2026-01-01")},
            {"id": 2, "product": "Laptop", "category": "Computers", "rating": 1, "sentiment": "Negative",
             "emotion": "Angry", "topic": "Delivery", "feedback_date": pd.Timestamp("2026-01-02")},
            {"id": 3, "product": "Phone", "category": "Mobile", "rating": 3, "sentiment": "Neutral",
             "emotion": "Neutral", "topic": "Other", "feedback_date": pd.Timestamp("2026-01-03")},
            {"id": 4, "product": "Phone", "category": "Mobile", "rating": 2, "sentiment": "Negative",
             "emotion": "Frustrated", "topic": "Delivery", "feedback_date": pd.Timestamp("2026-01-04")},
        ]
    )


def test_compute_loop_satisfaction_score_formula():
    df = _sample_df()
    # avg rating = (5+1+3+2)/4 = 2.75 -> rating_score = 2.75/5*100 = 55.0
    # positive%=25, neutral%=25, negative%=50 -> sentiment_score = 25 + 25*0.5 = 37.5
    # loop_score = 55.0*0.5 + 37.5*0.5 = 46.25 -> rounds to 46.2 or 46.3
    result = compute_loop_satisfaction_score(df)
    assert result["rating_score"] == 55.0
    assert result["sentiment_score"] == 37.5
    assert abs(result["loop_score"] - 46.25) < 0.1


def test_compute_loop_satisfaction_score_empty_df():
    result = compute_loop_satisfaction_score(pd.DataFrame())
    assert result["loop_score"] == 0.0


def test_compute_kpis_counts_and_average():
    df = _sample_df()
    kpis = compute_kpis(df)
    assert kpis["total_feedback"] == 4
    assert kpis["positive_feedback"] == 1
    assert kpis["negative_feedback"] == 2
    assert kpis["neutral_feedback"] == 1
    assert kpis["average_rating"] == 2.75


def test_compute_kpis_empty_df():
    kpis = compute_kpis(pd.DataFrame())
    assert kpis["total_feedback"] == 0
    assert kpis["average_rating"] == 0.0


def test_sentiment_distribution():
    df = _sample_df()
    dist = sentiment_distribution(df)
    dist_dict = dict(zip(dist["sentiment"], dist["count"]))
    assert dist_dict["Positive"] == 1
    assert dist_dict["Negative"] == 2
    assert dist_dict["Neutral"] == 1


def test_rating_distribution_includes_all_five_stars():
    df = _sample_df()
    dist = rating_distribution(df)
    assert set(dist["rating"]) == {1, 2, 3, 4, 5}
    assert dist[dist["rating"] == 5]["count"].iloc[0] == 1
    assert dist[dist["rating"] == 4]["count"].iloc[0] == 0  # no 4-star ratings


def test_product_wise_sentiment():
    df = _sample_df()
    pws = product_wise_sentiment(df)
    laptop_positive = pws[(pws["product"] == "Laptop") & (pws["sentiment"] == "Positive")]
    assert laptop_positive["count"].iloc[0] == 1


def test_product_rating_extremes():
    df = _sample_df()
    extremes = product_rating_extremes(df)
    laptop_row = extremes[extremes["product"] == "Laptop"].iloc[0]
    assert laptop_row["min_rating"] == 1
    assert laptop_row["max_rating"] == 5
    assert laptop_row["avg_rating"] == 3.0
    # sorted by avg_rating descending -- Phone (avg 2.5) should rank below Laptop (avg 3.0)
    assert list(extremes["product"])[0] == "Laptop"


def test_top_complaints_chart_data_only_negative():
    df = _sample_df()
    complaints = top_complaints_chart_data(df)
    assert set(complaints["topic"]) == {"Delivery"}
    assert complaints[complaints["topic"] == "Delivery"]["count"].iloc[0] == 2


def test_apply_filters_by_product():
    df = _sample_df()
    filters = Filters(products=["Laptop"])
    filtered = apply_filters(df, filters)
    assert len(filtered) == 2
    assert set(filtered["product"]) == {"Laptop"}


def test_apply_filters_by_sentiment():
    df = _sample_df()
    filters = Filters(sentiments=["Negative"])
    filtered = apply_filters(df, filters)
    assert len(filtered) == 2
    assert set(filtered["sentiment"]) == {"Negative"}


def test_apply_filters_by_date_range():
    df = _sample_df()
    filters = Filters(date_range=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")))
    filtered = apply_filters(df, filters)
    assert len(filtered) == 2


def test_apply_filters_by_date_range_does_not_drop_rows_with_missing_date():
    # Regression test: a real bug where manually-entered feedback (which
    # has no feedback_date) was silently disappearing from every Dashboard
    # view, because the Dashboard's date filter is active by default and
    # a NaT-vs-Timestamp comparison is always False in pandas.
    df = _sample_df()
    df = pd.concat([df, pd.DataFrame([{
        "id": 5, "product": "Laptop", "category": "Computers", "rating": 4,
        "sentiment": "Positive", "emotion": "Happy", "topic": "Product Quality",
        "feedback_date": pd.NaT,
    }])], ignore_index=True)

    filters = Filters(date_range=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")))
    filtered = apply_filters(df, filters)
    # The 2 rows within the date range, PLUS the 1 row with no date at all.
    assert len(filtered) == 3
    assert filtered["id"].isin([5]).any()


def test_apply_filters_combines_multiple_conditions():
    df = _sample_df()
    filters = Filters(products=["Phone"], sentiments=["Negative"])
    filtered = apply_filters(df, filters)
    assert len(filtered) == 1
    assert filtered.iloc[0]["id"] == 4


def test_product_summary_computes_expected_fields():
    df = _sample_df()
    summary = product_summary(df, "Laptop")
    assert summary["total_feedback"] == 2
    assert summary["average_rating"] == 3.0
    assert summary["positive_pct"] == 50.0
    assert summary["negative_pct"] == 50.0
    assert summary["top_complaint"] == "Delivery"


def test_product_summary_handles_unknown_product():
    df = _sample_df()
    summary = product_summary(df, "Nonexistent")
    assert summary["total_feedback"] == 0
    assert summary["top_complaint"] is None
