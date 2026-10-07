"""
ui/dashboard.py
-----------------
The Dashboard page: filters, KPI cards, and the 8 Plotly charts from spec
section 15.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from dataclasses import replace

from ui.period_picker import render_period_picker
from ui.theme import card, load_error, no_data_state, page_header, score_gauge, show_chart

from services.period import compare_periods
from services.analytics import (
    load_feedback_df,
    Filters,
    apply_filters,
    compute_kpis,
    compute_loop_satisfaction_score,
    sentiment_distribution,
    sentiment_over_time,
    feedback_by_product,
    feedback_by_category,
    top_complaints_chart_data,
    emotion_distribution,
    rating_distribution,
    product_wise_sentiment,
    product_rating_extremes,
)

SENTIMENT_COLORS = {"Positive": "#22C55E", "Neutral": "#94A3B8", "Negative": "#EF4444"}


def _render_filters(df: pd.DataFrame) -> Filters:
    with st.expander("🔎 Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            products = st.multiselect("Product", sorted(df["product"].dropna().unique()) if not df.empty else [])
            sentiments = st.multiselect(
                "Sentiment", sorted(df["sentiment"].dropna().unique()) if not df.empty else []
            )
        with col2:
            categories = st.multiselect("Category", sorted(df["category"].dropna().unique()) if not df.empty else [])
            ratings = st.multiselect("Rating", [1, 2, 3, 4, 5])
        with col3:
            topics = st.multiselect("Topic", sorted(df["topic"].dropna().unique()) if not df.empty else [])

    return Filters(
        products=products,
        categories=categories,
        sentiments=sentiments,
        ratings=ratings,
        topics=topics,
    )


def _delta(change: dict | None, kind: str) -> str | None:
    """Text for the arrow under a metric: percent for counts, points for scores."""
    if not change:
        return None
    if kind == "pct":
        return None if change["pct"] is None else f"{change['pct']:+.0f}%"
    return f"{change['diff']:+.2f}" if kind == "rating" else f"{change['diff']:+.1f}"


def _render_kpis(df: pd.DataFrame, comparison: dict | None = None) -> None:
    kpis = compute_kpis(df)
    c = comparison or {}
    cols = st.columns(6)
    cols[0].metric("Total Feedback", kpis["total_feedback"], _delta(c.get("total_feedback"), "pct"))
    cols[1].metric("Positive", kpis["positive_feedback"], _delta(c.get("positive_feedback"), "pct"))
    # More complaints is bad news, so an increase is shown in red.
    cols[2].metric("Negative", kpis["negative_feedback"], _delta(c.get("negative_feedback"), "pct"), delta_color="inverse")
    cols[3].metric("Neutral", kpis["neutral_feedback"], _delta(c.get("neutral_feedback"), "pct"), delta_color="off")
    cols[4].metric(
        "Average Rating",
        f"{kpis['average_rating']:.2f} / 5" if kpis["total_feedback"] else "—",
        _delta(c.get("average_rating"), "rating"),
    )
    cols[5].metric(
        "LOOP Score",
        f"{kpis['loop_satisfaction_score']:.1f}" if kpis["total_feedback"] else "—",
        _delta(c.get("loop_satisfaction_score"), "score"),
    )
    if comparison:
        st.caption(f"Arrows compare with the previous period ({comparison['previous_period'].text}).")

    with st.expander("How is the LOOP Satisfaction Score calculated?"):
        score = compute_loop_satisfaction_score(df)
        st.caption(
            "This is a custom metric defined by LOOP Insights, not an industry-standard score."
        )
        st.code(
            "Rating Score = Average Rating / 5 * 100\n"
            "Sentiment Score = Positive% + (Neutral% * 0.5)\n"
            "LOOP Score = (Rating Score * 0.5) + (Sentiment Score * 0.5)",
            language="text",
        )
        st.write(
            f"Current values — Rating Score: **{score['rating_score']}**, "
            f"Sentiment Score: **{score['sentiment_score']}**, "
            f"LOOP Score: **{score['loop_score']}**"
        )


def _chart_card(name: str, title: str, fig, empty_msg: str = "No data available.") -> None:
    """One chart inside a styled card, or a short message when there's no data."""
    with card(name):
        st.subheader(title)
        if fig is None:
            st.caption(empty_msg)
        else:
            show_chart(fig)


def _render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No feedback data matches the current filters.")
        return

    # Row 1: score gauge + sentiment donut + sentiment over time
    score = compute_loop_satisfaction_score(df)
    r1 = st.columns([1, 1, 2])
    with r1[0]:
        with card("gauge"):
            st.subheader("Satisfaction")
            show_chart(score_gauge(score["loop_score"], "LOOP Score"), height=250)
    data = sentiment_distribution(df)
    fig = (
        px.pie(data, names="sentiment", values="count", color="sentiment", color_discrete_map=SENTIMENT_COLORS)
        if not data.empty else None
    )
    with r1[1]:
        _chart_card("sent_dist", "Sentiment Distribution", fig, "No sentiment data available.")
    data = sentiment_over_time(df)
    if not data.empty:
        # Daily counts are very spiky with many products, so show weekly totals.
        data = data.assign(week=pd.to_datetime(data["date"]).dt.to_period("W").dt.start_time)
        data = data.groupby(["week", "sentiment"], as_index=False)["count"].sum()
        fig = px.line(data, x="week", y="count", color="sentiment", color_discrete_map=SENTIMENT_COLORS, markers=True)
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Feedback per week")
    else:
        fig = None
    with r1[2]:
        _chart_card("sent_time", "Sentiment Over Time (weekly)", fig, "No dated feedback available.")

    # Row 2: feedback by product (wide) + by category
    r2 = st.columns([3, 2])
    data = feedback_by_product(df)
    with r2[0]:
        fig = px.bar(data, x="count", y="product", orientation="h") if not data.empty else None
        if fig is not None:
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        _chart_card("by_product", "Feedback by Product", fig)
    data = feedback_by_category(df)
    with r2[1]:
        fig = px.pie(data, names="category", values="count") if not data.empty else None
        _chart_card("by_category", "Feedback by Category", fig)

    # Row 3: complaints + emotions
    r3 = st.columns(2)
    data = top_complaints_chart_data(df, top_n=8)
    # "Other" just means the topic model found no specific issue, so it is
    # not a useful complaint to show.
    data = data[data["topic"] != "Other"].head(5)
    with r3[0]:
        fig = px.bar(data, x="count", y="topic", orientation="h") if not data.empty else None
        if fig is not None:
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        _chart_card("complaints", "Top Customer Complaints", fig, "No negative feedback with topic analysis found.")
    data = emotion_distribution(df)
    with r3[1]:
        fig = px.bar(data, x="emotion", y="count") if not data.empty else None
        _chart_card("emotions", "Emotion Distribution", fig, "No emotion data available.")

    # Row 4: ratings + product-wise sentiment
    r4 = st.columns([2, 3])
    data = rating_distribution(df)
    with r4[0]:
        fig = px.bar(data, x="rating", y="count")
        fig.update_xaxes(dtick=1)
        _chart_card("ratings", "Rating Distribution", fig)
    data = product_wise_sentiment(df)
    with r4[1]:
        fig = (
            px.bar(data, x="product", y="count", color="sentiment", color_discrete_map=SENTIMENT_COLORS, barmode="stack")
            if not data.empty else None
        )
        _chart_card("prod_sent", "Product-wise Sentiment", fig, "No sentiment data available.")


def _render_product_rating_summary(df: pd.DataFrame) -> None:
    st.subheader("Product Rating Summary")
    extremes = product_rating_extremes(df)
    if extremes.empty:
        st.caption("No rated feedback available.")
        return

    best = extremes.iloc[0]
    worst = extremes.iloc[-1]
    cols = st.columns(2)
    cols[0].metric("🏆 Highest Rated Product", best["product"])
    cols[0].caption(f"Average rating {best['avg_rating']:.2f} / 5")
    cols[1].metric("⚠️ Lowest Rated Product", worst["product"])
    cols[1].caption(f"Average rating {worst['avg_rating']:.2f} / 5")

    display = extremes.rename(columns={
        "product": "Product",
        "total_feedback": "Total Feedback",
        "min_rating": "Lowest Rating Given",
        "max_rating": "Highest Rating Given",
        "avg_rating": "Average Rating",
    })
    st.dataframe(display, width="stretch", hide_index=True)


def render() -> None:
    page_header("📊", "Dashboard", "Sentiment, trends and recurring complaints across all feedback.")

    try:
        full_df = load_feedback_df()
    except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
        load_error(exc)
        return

    if full_df.empty:
        no_data_state()
        return

    period = render_period_picker(full_df)
    filters = _render_filters(full_df)
    filters.date_range = period.date_range
    filtered_df = apply_filters(full_df, filters)
    comparison = compare_periods(apply_filters(full_df, replace(filters, date_range=None)), period)

    if not filters.is_empty():
        st.caption(f"Showing {len(filtered_df)} of {len(full_df)} records matching your filters.")

    _render_kpis(filtered_df, comparison)
    st.divider()
    _render_product_rating_summary(filtered_df)
    st.divider()
    _render_charts(filtered_df)
