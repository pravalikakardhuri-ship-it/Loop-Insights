"""
services/preprocessing.py
--------------------------
Reusable data cleaning module for uploaded/manual feedback.

Deliberately does NOT strip words like "not", "never", "bad", "poor" --
those carry the sentiment signal that later phases (sentiment/emotion
models) depend on. Only structural noise (URLs, stray whitespace, odd
punctuation) is removed.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

from config import config

# Canonical column names the rest of the app expects, mapped from common
# variants a user's spreadsheet might use.
COLUMN_ALIASES = {
    "customer_id": ["customer_id", "cust_id", "customerid", "id", "cust id"],
    "customer_name": ["customer_name", "name", "customername", "customer"],
    "product": ["product", "product_name", "item"],
    "category": ["category", "product_category", "type"],
    "rating": ["rating", "score", "stars", "star_rating"],
    "feedback": ["feedback", "feedback_text", "review", "comment", "comments", "review_text"],
    "date": ["date", "feedback_date", "review_date", "created_date"],
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
# Keep letters, numbers, whitespace, and basic sentence punctuation
# (including the apostrophe, since "wasn't", "doesn't" etc. matter for sentiment).
SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s.,!?'\-]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


@dataclass
class CleaningReport:
    """Summary stats shown to the user before they confirm an import."""

    total_records: int = 0
    valid_records: int = 0
    missing_values_dropped: int = 0
    duplicate_records_dropped: int = 0
    invalid_ratings_dropped: int = 0
    empty_feedback_dropped: int = 0
    column_mapping_used: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "Total Records": self.total_records,
            "Valid Records": self.valid_records,
            "Missing Values Dropped": self.missing_values_dropped,
            "Duplicate Records": self.duplicate_records_dropped,
            "Invalid Ratings": self.invalid_ratings_dropped,
            "Empty Feedback": self.empty_feedback_dropped,
        }


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and underscore-ify column names for reliable matching."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def auto_map_columns(df: pd.DataFrame) -> dict:
    """Best-effort mapping from the uploaded file's columns to canonical names.

    Returns a dict {canonical_name: source_column_name_or_None}. Any
    canonical field that couldn't be matched automatically is left as None
    so the UI can prompt the user to map it manually.
    """
    normalized = normalize_column_names(df)
    available = set(normalized.columns)

    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        match = next((a for a in aliases if a in available), None)
        mapping[canonical] = match
    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename the uploaded dataframe's columns to canonical names using the
    given {canonical_name: source_column_name} mapping. Unmapped canonical
    columns are created as empty (NaN) columns so downstream code can rely
    on their presence.
    """
    normalized = normalize_column_names(df)
    out = pd.DataFrame()
    for canonical in config.REQUIRED_COLUMNS:
        source = mapping.get(canonical)
        if source and source in normalized.columns:
            out[canonical] = normalized[source]
        else:
            out[canonical] = pd.NA
    return out


def clean_text(text) -> str:
    """Clean a single feedback string.

    Removes URLs and stray special characters, collapses whitespace, but
    intentionally preserves negation/sentiment words and basic punctuation
    that matters for sentiment analysis.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = URL_PATTERN.sub("", text)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def _coerce_rating(value):
    """Convert a rating value to an int in [1, 5], or None if invalid.

    Handles NaN explicitly: float(NaN) succeeds (NaN is a valid float),
    but int(NaN) raises ValueError, so a plain float() check alone isn't
    enough -- a blank/missing rating cell (which pandas reads as NaN) has
    to be caught here rather than falling through to a crash.
    """
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(rating):
        return None
    if rating != int(rating):
        return None
    rating = int(rating)
    if 1 <= rating <= 5:
        return rating
    return None


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Full cleaning pipeline for a dataframe already mapped to canonical
    columns (customer_id, customer_name, product, category, rating,
    feedback, date).

    Returns (cleaned_df, report).
    """
    report = CleaningReport(total_records=len(df))
    working = df.copy()

    # 1. Normalize whitespace on text columns.
    for col in ["customer_id", "customer_name", "product", "category"]:
        if col in working.columns:
            working[col] = working[col].astype(str).str.strip()
            working[col] = working[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    # 2. Clean feedback text (URLs, special chars, whitespace) while keeping
    #    sentiment-bearing words intact.
    if "feedback" in working.columns:
        working["feedback"] = working["feedback"].apply(clean_text)

    # 3. Drop rows with empty feedback text.
    before = len(working)
    working = working[working["feedback"].astype(str).str.strip() != ""]
    report.empty_feedback_dropped = before - len(working)

    # 4. Validate & coerce ratings; drop rows with invalid ratings.
    working["rating"] = working["rating"].apply(_coerce_rating)
    before = len(working)
    working = working[working["rating"].notna()]
    report.invalid_ratings_dropped = before - len(working)
    working["rating"] = working["rating"].astype(int)

    # 5. Convert dates; invalid dates become NaT rather than dropping the row
    #    (feedback_date is optional downstream).
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date

    # 6. Drop rows missing any other required field.
    before = len(working)
    required_non_null = ["customer_id", "product", "rating", "feedback"]
    working = working.dropna(subset=[c for c in required_non_null if c in working.columns])
    report.missing_values_dropped = before - len(working)

    # 7. Remove duplicate records (same customer, product, feedback text).
    before = len(working)
    dedup_cols = [c for c in ["customer_id", "product", "feedback"] if c in working.columns]
    working = working.drop_duplicates(subset=dedup_cols, keep="first")
    report.duplicate_records_dropped = before - len(working)

    # 8. Fill optional fields.
    if "category" in working.columns:
        working["category"] = working["category"].fillna("Other")
    if "customer_name" in working.columns:
        working["customer_name"] = working["customer_name"].fillna("Anonymous")

    working = working.reset_index(drop=True)
    report.valid_records = len(working)
    return working, report


def get_upload_stats(df: pd.DataFrame) -> dict:
    """Quick pre-cleaning stats shown right after upload (before the user
    confirms), matching the spec's "Total / Valid / Missing / Duplicate /
    Invalid Ratings" summary.
    """
    total = len(df)
    duplicate = int(df.duplicated().sum())
    missing = int(df.isna().any(axis=1).sum())

    invalid_ratings = 0
    if "rating" in df.columns:
        invalid_ratings = int(df["rating"].apply(_coerce_rating).isna().sum())

    valid_estimate = total - duplicate - missing
    valid_estimate = max(valid_estimate, 0)

    return {
        "Total Records": total,
        "Valid Records (est.)": valid_estimate,
        "Missing Values": missing,
        "Duplicate Records": duplicate,
        "Invalid Ratings": invalid_ratings,
    }
