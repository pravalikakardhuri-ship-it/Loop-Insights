"""
utils/validators.py
--------------------
File and input validation helpers used by the Upload Feedback and Manual
Feedback pages.
"""

from config import config

ALLOWED_EXTENSIONS = {"csv", "xlsx"}


def validate_file_extension(filename: str) -> tuple[bool, str]:
    """Check the uploaded file has an allowed extension."""
    if "." not in filename:
        return False, "File has no extension. Please upload a .csv or .xlsx file."
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '.{ext}'. Please upload a .csv or .xlsx file."
    return True, ""


def validate_file_size(size_bytes: int) -> tuple[bool, str]:
    """Check the uploaded file doesn't exceed the configured size limit."""
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        return False, f"File is {size_mb:.1f}MB, which exceeds the {config.MAX_UPLOAD_SIZE_MB}MB limit."
    return True, ""


def validate_not_empty(df) -> tuple[bool, str]:
    if df is None or len(df) == 0:
        return False, "The uploaded file is empty."
    return True, ""


def validate_rating_value(rating) -> tuple[bool, str]:
    """Validate a single rating value (used by the Manual Feedback page)."""
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return False, "Rating must be a whole number between 1 and 5."
    if not (1 <= rating <= 5):
        return False, "Rating must be between 1 and 5."
    return True, ""


def validate_feedback_text(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Feedback text cannot be empty."
    if len(text.strip()) < 3:
        return False, "Feedback text is too short to analyze meaningfully."
    return True, ""


def validate_required_fields_present(mapping: dict) -> tuple[bool, list[str]]:
    """Check that every required column has been mapped to something.

    `mapping` is {canonical_name: source_column_or_None}.
    Returns (all_present, list_of_missing_canonical_names).
    """
    missing = [field for field in config.REQUIRED_COLUMNS if not mapping.get(field)]
    return (len(missing) == 0), missing
