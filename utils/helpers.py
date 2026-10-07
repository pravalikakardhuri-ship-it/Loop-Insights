"""
utils/helpers.py
-----------------
Small shared helper functions used across pages/services.
"""

import io

import pandas as pd


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit UploadedFile (CSV or XLSX) into a DataFrame.

    Raises ValueError with a user-friendly message on failure rather than
    letting pandas' raw exception bubble up to the UI.
    """
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower()

    try:
        if ext == "csv":
            return pd.read_csv(uploaded_file)
        elif ext == "xlsx":
            return pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: .{ext}")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file is empty.") from exc
    except Exception as exc:  # noqa: BLE001 - re-raised as a friendly message
        raise ValueError(f"Could not read the file: {exc}") from exc


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a cleaned dataframe (canonical columns) into the list of
    dicts expected by database.queries.bulk_insert_feedback.
    """
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "customer_id": str(row["customer_id"]),
                "customer_name": row.get("customer_name") or "Anonymous",
                "product": str(row["product"]),
                "category": row.get("category") or "Other",
                "rating": int(row["rating"]),
                "feedback_text": str(row["feedback"]),
                "feedback_date": row.get("date") if pd.notna(row.get("date")) else None,
            }
        )
    return records
