"""Tests for utils/validators.py (upload and manual-entry validation)."""

import pandas as pd

from config import config
from utils.validators import (
    validate_feedback_text,
    validate_file_extension,
    validate_file_size,
    validate_not_empty,
    validate_rating_value,
    validate_required_fields_present,
)


def test_file_extension_accepts_csv_and_xlsx():
    assert validate_file_extension("data.csv")[0]
    assert validate_file_extension("DATA.XLSX")[0]


def test_file_extension_rejects_other_types_and_no_extension():
    assert not validate_file_extension("notes.txt")[0]
    assert not validate_file_extension("noextension")[0]


def test_file_size_limit():
    limit = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    assert validate_file_size(limit)[0]
    assert not validate_file_size(limit + 1)[0]


def test_not_empty():
    assert not validate_not_empty(None)[0]
    assert not validate_not_empty(pd.DataFrame())[0]
    assert validate_not_empty(pd.DataFrame({"a": [1]}))[0]


def test_rating_value_bounds_and_types():
    for good in (1, 3, 5, "4"):
        assert validate_rating_value(good)[0]
    for bad in (0, 6, -1, "abc", None):
        assert not validate_rating_value(bad)[0]


def test_feedback_text_rules():
    assert validate_feedback_text("Great product")[0]
    assert not validate_feedback_text("")[0]
    assert not validate_feedback_text("   ")[0]
    assert not validate_feedback_text(None)[0]
    assert not validate_feedback_text("ok")[0]  # under 3 characters


def test_required_fields_present():
    full = {field: field for field in config.REQUIRED_COLUMNS}
    assert validate_required_fields_present(full) == (True, [])

    partial = dict(full)
    first = config.REQUIRED_COLUMNS[0]
    partial[first] = None
    ok, missing = validate_required_fields_present(partial)
    assert not ok and missing == [first]
