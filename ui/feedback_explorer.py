"""
ui/feedback_explorer.py
--------------------------
The Feedback Explorer page: search + filter the raw feedback table
(spec section 17).
"""

import streamlit as st

from ui.theme import load_error, no_data_state, page_header

from services.analytics import load_feedback_df, Filters, apply_filters


def render() -> None:
    page_header("🔎", "Feedback Explorer", "Search and filter individual feedback records.")

    try:
        df = load_feedback_df()
    except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
        load_error(exc)
        return

    if df.empty:
        no_data_state()
        return

    search = st.text_input("Search feedback", placeholder="e.g. delivery")

    col1, col2, col3 = st.columns(3)
    with col1:
        products = st.multiselect("Product", sorted(df["product"].dropna().unique()))
        emotions = st.multiselect("Emotion", sorted(df["emotion"].dropna().unique()))
    with col2:
        sentiments = st.multiselect("Sentiment", sorted(df["sentiment"].dropna().unique()))
        topics = st.multiselect("Topic", sorted(df["topic"].dropna().unique()))
    with col3:
        ratings = st.multiselect("Rating", [1, 2, 3, 4, 5])
        date_range = None
        if df["feedback_date"].notna().any():
            min_date = df["feedback_date"].min().date()
            max_date = df["feedback_date"].max().date()
            picked = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(picked, tuple) and len(picked) == 2:
                date_range = picked

    filters = Filters(
        date_range=date_range, products=products, sentiments=sentiments, ratings=ratings, topics=topics,
    )
    filtered = apply_filters(df, filters)

    if emotions:
        filtered = filtered[filtered["emotion"].isin(emotions)]

    if search:
        filtered = filtered[filtered["feedback_text"].str.contains(search, case=False, na=False)]

    st.caption(f"Showing {len(filtered)} of {len(df)} records.")

    display_df = filtered.rename(
        columns={
            "customer_name": "Customer",
            "product": "Product",
            "rating": "Rating",
            "feedback_text": "Feedback",
            "sentiment": "Sentiment",
            "emotion": "Emotion",
            "topic": "Topic",
            "feedback_date": "Date",
        }
    )[["Customer", "Product", "Rating", "Feedback", "Sentiment", "Emotion", "Topic", "Date"]]

    st.dataframe(display_df, width="stretch", height=500)
