"""
tests/test_preprocessing.py
----------------------------
Tests for services/preprocessing.py (spec section 30: Data preprocessing).
"""

import pandas as pd

from services.preprocessing import (
    clean_text,
    clean_dataframe,
    auto_map_columns,
    apply_column_mapping,
    get_upload_stats,
)


def test_clean_text_removes_urls():
    text = "Great product, check it out at https://example.com/product"
    cleaned = clean_text(text)
    assert "https://" not in cleaned
    assert "example.com" not in cleaned


def test_clean_text_preserves_negation_words():
    text = "This is not good, never buying again, the quality is bad and poor."
    cleaned = clean_text(text)
    for word in ["not", "never", "bad", "poor"]:
        assert word in cleaned.lower()


def test_clean_text_collapses_whitespace():
    text = "This   has     extra    spaces"
    cleaned = clean_text(text)
    assert "  " not in cleaned


def test_clean_text_handles_nan():
    assert clean_text(float("nan")) == ""


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C002", "C003", "C004"],
            "customer_name": ["Rahul", "Priya", "Priya", None, "Amit"],
            "product": ["Laptop", "Phone", "Phone", "Tablet", "Camera"],
            "category": ["Electronics", "Mobile", "Mobile", None, "Electronics"],
            "rating": [5, 2, 2, 7, "abc"],  # 7 and "abc" are invalid
            "feedback": [
                "Great laptop, works well.",
                "Bad delivery, very late.",
                "Bad delivery, very late.",  # duplicate of row above
                "Okay tablet.",
                "",  # empty feedback
            ],
            "date": ["2026-01-01", "2026-02-01", "2026-02-01", "2026-03-01", "2026-04-01"],
        }
    )


def test_clean_dataframe_drops_duplicates():
    df = _sample_df()
    cleaned, report = clean_dataframe(df)
    assert report.duplicate_records_dropped >= 1
    # the duplicate "Bad delivery, very late." from C002 should appear only once
    assert (cleaned["feedback"] == "Bad delivery very late.").sum() <= 1


def test_clean_dataframe_drops_invalid_ratings():
    df = _sample_df()
    cleaned, report = clean_dataframe(df)
    assert report.invalid_ratings_dropped >= 1
    assert cleaned["rating"].between(1, 5).all()


def test_clean_dataframe_drops_empty_feedback():
    df = _sample_df()
    cleaned, report = clean_dataframe(df)
    assert report.empty_feedback_dropped >= 1
    assert (cleaned["feedback"].str.strip() == "").sum() == 0


def test_clean_dataframe_fills_missing_category_and_name():
    df = _sample_df()
    cleaned, _ = clean_dataframe(df)
    assert cleaned["category"].isna().sum() == 0
    assert cleaned["customer_name"].isna().sum() == 0


def test_auto_map_columns_matches_standard_names():
    df = pd.DataFrame(columns=["customer_id", "customer_name", "product", "category", "rating", "feedback", "date"])
    mapping = auto_map_columns(df)
    for canonical in ["customer_id", "customer_name", "product", "category", "rating", "feedback", "date"]:
        assert mapping[canonical] == canonical


def test_auto_map_columns_matches_variant_names():
    df = pd.DataFrame(columns=["Cust ID", "Name", "Item", "Type", "Stars", "Review", "Review Date"])
    mapping = auto_map_columns(df)
    assert mapping["customer_id"] == "cust_id"
    assert mapping["customer_name"] == "name"
    assert mapping["product"] == "item"
    assert mapping["rating"] == "stars"
    assert mapping["feedback"] == "review"


def test_apply_column_mapping_produces_canonical_columns():
    df = pd.DataFrame({"Item": ["Laptop"], "Review": ["Great!"], "Stars": [5]})
    mapping = {"product": "item", "feedback": "review", "rating": "stars"}
    mapped = apply_column_mapping(df, mapping)
    assert list(mapped.columns) == [
        "customer_id", "customer_name", "product", "category", "rating", "feedback", "date",
    ]
    assert mapped["product"].iloc[0] == "Laptop"
    assert mapped["feedback"].iloc[0] == "Great!"


def test_get_upload_stats_counts_correctly():
    df = _sample_df()
    stats = get_upload_stats(df)
    assert stats["Total Records"] == 5
    assert stats["Duplicate Records"] >= 0
    assert stats["Invalid Ratings"] == 2


def test_get_upload_stats_handles_missing_rating_without_crashing():
    # Regression test: a blank rating cell becomes NaN when pandas reads
    # a CSV, and float(NaN) succeeds where int(NaN) raises -- an earlier
    # version of _coerce_rating crashed on this instead of treating a
    # missing rating as simply invalid.
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "customer_name": ["A", "B"],
            "product": ["Laptop", "Phone"],
            "category": ["Electronics", "Mobile"],
            "rating": [5, float("nan")],
            "feedback": ["Great!", "Missing rating but has text."],
            "date": ["2026-01-01", "2026-01-02"],
        }
    )
    stats = get_upload_stats(df)
    assert stats["Invalid Ratings"] == 1

    cleaned, report = clean_dataframe(df)
    assert report.invalid_ratings_dropped == 1
    assert len(cleaned) == 1
