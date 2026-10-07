"""
ui/product_analysis.py
-------------------------
The Product Analysis page (spec section 24): pick a product, see its
stats, sentiment/rating charts, top complaints, and sample feedback.

The "AI-generated product summary" listed in the spec is added in
Phase 7 - Groq AI, once the LLM service exists; a note says so here
rather than faking a summary.
"""

import plotly.express as px
import streamlit as st

from ui.period_picker import render_period_picker
from ui.theme import ai_problem, load_error, no_data_state, page_header, show_chart

from services.analytics import (
    Filters,
    apply_filters,
    load_feedback_df,
    product_summary,
    rating_distribution,
)

SENTIMENT_COLORS = {"Positive": "#22C55E", "Neutral": "#94A3B8", "Negative": "#EF4444"}


def render() -> None:
    page_header("📈", "Product Analysis", "Performance, sentiment and complaints for one product.")

    try:
        df = load_feedback_df()
    except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
        load_error(exc)
        return

    if df.empty:
        no_data_state()
        return

    products = sorted(df["product"].dropna().unique())
    if not products:
        st.info("No products found in the feedback data yet.")
        return

    period = render_period_picker(df)
    selected_product = st.selectbox("Select a product", products)
    df = apply_filters(df, Filters(date_range=period.date_range))
    product_df = df[df["product"] == selected_product]
    if product_df.empty:
        st.warning(f"No feedback for {selected_product} in this period. Pick a longer period.")
        return
    summary = product_summary(df, selected_product)

    st.subheader(selected_product)
    cols = st.columns(6)
    cols[0].metric("Total Feedback", summary["total_feedback"])
    cols[1].metric("Average Rating", f"{summary['average_rating']:.2f} / 5")
    cols[2].metric("Positive %", f"{summary['positive_pct']}%")
    cols[3].metric("Negative %", f"{summary['negative_pct']}%")
    cols[4].metric("LOOP Satisfaction Score", summary["loop_satisfaction_score"])
    cols[5].metric("Top Emotion", summary["top_emotion"] or "—")

    st.caption(f"Top complaint: **{summary['top_complaint'] or 'None'}**")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Breakdown")
        sentiment_counts = product_df["sentiment"].dropna().value_counts().rename_axis("sentiment").reset_index(name="count")
        if not sentiment_counts.empty:
            fig = px.pie(sentiment_counts, names="sentiment", values="count", color="sentiment", color_discrete_map=SENTIMENT_COLORS)
            show_chart(fig, height=300)
        else:
            st.caption("No sentiment data available for this product.")

    with col2:
        st.subheader("Rating Distribution")
        rd = rating_distribution(product_df)
        fig = px.bar(rd, x="rating", y="count")
        fig.update_xaxes(dtick=1)
        show_chart(fig, height=300)

    st.subheader("Top Complaints")
    negative_topics = product_df[(product_df["sentiment"] == "Negative") & (product_df["topic"].notna())]
    if not negative_topics.empty:
        counts = negative_topics["topic"].value_counts().reset_index()
        counts.columns = ["Topic", "Mentions"]
        st.dataframe(counts, width="stretch", hide_index=True)
    else:
        st.caption("No negative feedback with topic analysis found for this product.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sample Positive Feedback")
        positive_samples = product_df[product_df["sentiment"] == "Positive"]["feedback_text"].dropna().head(5)
        if positive_samples.empty:
            st.caption("None found.")
        for text in positive_samples:
            st.write(f"- {text}")

    with col4:
        st.subheader("Sample Negative Feedback")
        negative_samples = product_df[product_df["sentiment"] == "Negative"]["feedback_text"].dropna().head(5)
        if negative_samples.empty:
            st.caption("None found.")
        for text in negative_samples:
            st.write(f"- {text}")

    st.divider()
    st.subheader("AI-Generated Product Summary")

    from config import config

    if not config.GROQ_API_KEY:
        ai_problem(admin_message="GROQ_API_KEY is not set. Add it to .env (or the host's secrets), then test it in Settings.", level="warning")
    else:
        session_key = f"product_ai_summary_{selected_product}"
        if st.button("Generate AI Summary for this Product"):
            with st.spinner("Asking Groq for a product summary..."):
                try:
                    from services.llm_service import generate_summary, LLMServiceError

                    product_context = {
                        "product": summary["product"],
                        "total_feedback": summary["total_feedback"],
                        "average_rating": summary["average_rating"],
                        "positive_pct": summary["positive_pct"],
                        "negative_pct": summary["negative_pct"],
                        "neutral_pct": summary["neutral_pct"],
                        "loop_satisfaction_score": summary["loop_satisfaction_score"],
                        "top_complaint": summary["top_complaint"],
                        "top_emotion": summary["top_emotion"],
                    }
                    st.session_state[session_key] = generate_summary(product_context)
                except LLMServiceError as exc:
                    ai_problem(exc)

        if st.session_state.get(session_key):
            st.info(st.session_state[session_key])
