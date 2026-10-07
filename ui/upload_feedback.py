"""
ui/upload_feedback.py
----------------------
The Upload Feedback page: CSV/XLSX upload -> preview -> validate -> map
columns -> clean -> confirm -> store in MySQL.

Kept out of app.py to avoid one giant file, per the project's development
rules (UI/ML/DB logic stays separated into modules).
"""

import pandas as pd
import streamlit as st

from ui.theme import page_header, problem

from config import config
from services.preprocessing import (
    auto_map_columns,
    apply_column_mapping,
    clean_dataframe,
    get_upload_stats,
)
from utils.helpers import read_uploaded_file, dataframe_to_records
from utils.validators import (
    validate_file_extension,
    validate_file_size,
    validate_not_empty,
    validate_required_fields_present,
)

STATE_PREFIX = "upload_"


def _reset_state():
    for key in list(st.session_state.keys()):
        if key.startswith(STATE_PREFIX):
            del st.session_state[key]


def render() -> None:
    page_header("📤", "Upload Feedback", "Import a CSV or Excel file of customer feedback.")
    st.write("Upload a CSV or Excel file of customer feedback to import into LOOP Insights.")

    uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.caption(
            f"Expected columns: {', '.join(config.REQUIRED_COLUMNS)}. "
            "Slightly different column names are fine -- you'll get a chance to map them."
        )
        return

    # --- Step 0: validate the file itself -----------------------------------
    ext_ok, ext_msg = validate_file_extension(uploaded_file.name)
    if not ext_ok:
        st.error(ext_msg)
        return

    size_ok, size_msg = validate_file_size(uploaded_file.size)
    if not size_ok:
        st.error(size_msg)
        return

    try:
        raw_df = read_uploaded_file(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    empty_ok, empty_msg = validate_not_empty(raw_df)
    if not empty_ok:
        st.error(empty_msg)
        return

    # --- Step 1: preview --------------------------------------------------
    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(20), width="stretch")
    st.caption(f"Showing first {min(20, len(raw_df))} of {len(raw_df)} rows.")

    # --- Step 2: column mapping --------------------------------------------
    st.subheader("Column Mapping")
    auto_mapping = auto_map_columns(raw_df)
    normalized_cols = [str(c).strip().lower().replace(" ", "_") for c in raw_df.columns]

    all_matched = all(auto_mapping.get(f) for f in config.REQUIRED_COLUMNS)
    if all_matched:
        st.success("All required columns were matched automatically.")
        with st.expander("View column mapping", expanded=False):
            st.json(auto_mapping)
        final_mapping = auto_mapping
    else:
        st.warning("Some columns couldn't be matched automatically. Please map them below.")
        final_mapping = {}
        options = ["-- none --"] + normalized_cols
        cols = st.columns(2)
        for i, field in enumerate(config.REQUIRED_COLUMNS):
            default = auto_mapping.get(field)
            default_index = options.index(default) if default in options else 0
            with cols[i % 2]:
                choice = st.selectbox(
                    f"'{field}' comes from:", options, index=default_index, key=f"map_{field}"
                )
            final_mapping[field] = None if choice == "-- none --" else choice

    fields_ok, missing_fields = validate_required_fields_present(final_mapping)
    if not fields_ok:
        st.error(f"Please map these required fields before continuing: {', '.join(missing_fields)}")
        return

    mapped_df = apply_column_mapping(raw_df, final_mapping)

    # --- Step 3: pre-clean stats --------------------------------------------
    st.subheader("Upload Summary")
    stats = get_upload_stats(mapped_df)
    stat_cols = st.columns(len(stats))
    for col, (label, value) in zip(stat_cols, stats.items()):
        col.metric(label, value)

    # --- Step 4: clean & preview --------------------------------------------
    st.subheader("Data Cleaning")
    if st.button("Clean & Preview", type="secondary"):
        cleaned_df, report = clean_dataframe(mapped_df)
        st.session_state[f"{STATE_PREFIX}cleaned_df"] = cleaned_df
        st.session_state[f"{STATE_PREFIX}report"] = report

    cleaned_df = st.session_state.get(f"{STATE_PREFIX}cleaned_df")
    report = st.session_state.get(f"{STATE_PREFIX}report")

    if cleaned_df is not None and report is not None:
        report_cols = st.columns(len(report.as_dict()))
        for col, (label, value) in zip(report_cols, report.as_dict().items()):
            col.metric(label, value)

        left, right = st.columns(2)
        with left:
            st.markdown("**Before cleaning**")
            st.dataframe(mapped_df.head(10), width="stretch")
        with right:
            st.markdown("**After cleaning**")
            st.dataframe(cleaned_df.head(10), width="stretch")

        if len(cleaned_df) == 0:
            st.error("No valid records remain after cleaning. Please check your file and try again.")
            return

        st.divider()
        st.write(f"Ready to import **{len(cleaned_df)}** valid records into MySQL.")

        if st.button("Confirm Import", type="primary"):
            from database.queries import bulk_insert_analysis, bulk_insert_feedback

            try:
                records = dataframe_to_records(cleaned_df)
                ids = bulk_insert_feedback(records)
                st.success(f"Imported {len(ids)} feedback records into the database.")
            except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
                problem(
                    "We couldn't save your records right now. Please try again in a moment.",
                    "Could not save records to the database. Make sure MySQL is connected and "
                    "the tables are initialized (see Settings).",
                    exc,
                )
                return

            texts = [r["feedback_text"] for r in records]
            sentiments, confidences = None, None
            topic_predictions = None

            try:
                from services.sentiment import (
                    predict_sentiment_batch, is_model_available, ModelNotTrainedError,
                )
                from services.emotion import predict_emotion_batch
                from services.topic_model import (
                    predict_topic_batch,
                    is_model_available as is_topic_model_available,
                    TopicModelNotTrainedError,
                )

                if is_model_available():
                    with st.spinner("Running sentiment analysis on imported records..."):
                        sentiment_predictions = predict_sentiment_batch(texts)
                        sentiments = [p[0] for p in sentiment_predictions]
                        confidences = [p[1] for p in sentiment_predictions]
                    st.success(f"Sentiment analysis completed for {len(ids)} records.")
                else:
                    st.info(
                        "No trained sentiment model found yet, so sentiment wasn't analyzed. "
                        "Train one from the Settings page."
                    )

                with st.spinner("Running emotion detection..."):
                    emotion_predictions = predict_emotion_batch(texts, sentiments)

                if is_topic_model_available():
                    with st.spinner("Running topic/issue detection..."):
                        topic_predictions = predict_topic_batch(texts)
                    st.success(f"Emotion and topic analysis completed for {len(ids)} records.")
                else:
                    st.info(
                        "No trained topic model found yet, so topics weren't detected. "
                        "Train one from the Settings page."
                    )

                with st.spinner("Saving analysis results..."):
                    bulk_insert_analysis([
                        {
                            "feedback_id": feedback_id,
                            "sentiment": sentiments[i] if sentiments else None,
                            "sentiment_confidence": confidences[i] if confidences else None,
                            "emotion": emotion_predictions[i][0],
                            "emotion_confidence": emotion_predictions[i][1],
                            "topic": topic_predictions[i][0] if topic_predictions else None,
                        }
                        for i, feedback_id in enumerate(ids)
                    ])
            except (ModelNotTrainedError, TopicModelNotTrainedError) as exc:
                st.warning(
                    f"Records were imported, but NLP analysis couldn't run: {exc}"
                )
            except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
                problem(
                    "Your records were imported, but the analysis didn't finish. Please tell the admin.",
                    "Records were imported, but something went wrong while analyzing or "
                    "saving the analysis results.",
                    exc,
                    level="warning",
                )

            _reset_state()
