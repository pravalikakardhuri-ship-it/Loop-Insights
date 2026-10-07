"""Tests for utils/helpers.py (file reading and record conversion)."""

import io

import pandas as pd
import pytest

from utils.helpers import dataframe_to_records, read_uploaded_file


class FakeUpload(io.BytesIO):
    """Minimal stand-in for a Streamlit UploadedFile."""

    def __init__(self, name: str, content: bytes):
        super().__init__(content)
        self.name = name


def test_read_csv():
    df = read_uploaded_file(FakeUpload("a.csv", b"product,rating\nLaptop,5\n"))
    assert list(df.columns) == ["product", "rating"]
    assert len(df) == 1


def test_read_empty_csv_gives_friendly_error():
    with pytest.raises(ValueError, match="empty"):
        read_uploaded_file(FakeUpload("a.csv", b""))


def test_read_unsupported_type():
    with pytest.raises(ValueError):
        read_uploaded_file(FakeUpload("a.txt", b"hello"))


def test_read_corrupt_xlsx_gives_friendly_error():
    with pytest.raises(ValueError, match="Could not read"):
        read_uploaded_file(FakeUpload("a.xlsx", b"not an excel file"))


def test_dataframe_to_records():
    df = pd.DataFrame(
        [{"customer_id": "C1", "customer_name": "Asha", "product": "Laptop",
          "category": "Electronics", "rating": 4, "feedback": "Good",
          "date": pd.Timestamp("2026-01-01").date()}]
    )
    rec = dataframe_to_records(df)[0]
    assert rec["customer_id"] == "C1"
    assert rec["feedback_text"] == "Good"
    assert rec["rating"] == 4
    assert rec["feedback_date"] == pd.Timestamp("2026-01-01").date()


def test_dataframe_to_records_defaults_when_optional_missing():
    df = pd.DataFrame(
        [{"customer_id": "C1", "product": "Laptop", "rating": 4, "feedback": "Good"}]
    )
    rec = dataframe_to_records(df)[0]
    assert rec["customer_name"] == "Anonymous"
    assert rec["category"] == "Other"
    assert rec["feedback_date"] is None
