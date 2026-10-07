"""
ui/ai_insights.py
--------------------
The AI Insights page: AI-generated summary (spec section 20) and
AI-generated recommendations (spec section 21), both built from
already-computed analytics, not raw feedback text.
"""

import streamlit as st

from ui.period_picker import render_period_picker
from ui.theme import ai_problem, load_error, no_data_state, page_header

from config import config
from services.analytics import (
    load_feedback_df,
    Filters,
    apply_filters,
    compute_kpis,
)
from database.queries import get_top_complaints


def _build_analytics_context(df) -> dict:
    """Assemble the compact analytics dict handed to the LLM -- exactly
    the numbers computed elsewhere in the app, nothing re-derived by the
    model itself."""
    kpis = compute_kpis(df)
    complaints = get_top_complaints(limit=5)

    total = kpis["total_feedback"]
    context = {
        "total_feedback": total,
        "positive_pct": round(kpis["positive_feedback"] / total * 100, 1) if total else 0,
        "neutral_pct": round(kpis["neutral_feedback"] / total * 100, 1) if total else 0,
        "negative_pct": round(kpis["negative_feedback"] / total * 100, 1) if total else 0,
        "average_rating": kpis["average_rating"],
        "loop_satisfaction_score": kpis["loop_satisfaction_score"],
        "top_complaints": [
            f"{c['topic']} – {c['percentage']}% of negative feedback ({c['count']} mentions)"
            for c in complaints
        ],
    }
    return context


def render() -> None:
    page_header("🤖", "AI Insights", "AI-written summary and prioritised recommendations.")
    st.caption("Powered by Groq. Summaries and recommendations are generated only from statistics already calculated by this app.")

    if not config.GROQ_API_KEY:
        ai_problem(admin_message="GROQ_API_KEY is not set. Add it to .env (or the host's secrets), then test it in Settings.", level="warning")
        return

    try:
        full_df = load_feedback_df()
    except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
        load_error(exc)
        return

    if full_df.empty:
        no_data_state()
        return

    period = render_period_picker(full_df)
    with st.expander("🔎 Filter the data used for AI Insights", expanded=False):
        products = st.multiselect("Product", sorted(full_df["product"].dropna().unique()))
    filtered_df = apply_filters(full_df, Filters(products=products, date_range=period.date_range))
    if filtered_df.empty:
        st.warning("No feedback in this period. Pick a longer period or clear the product filter.")
        return

    context = _build_analytics_context(filtered_df)

    with st.expander("Data being sent to the AI (nothing else)", expanded=False):
        st.json(context)

    st.divider()
    st.subheader("AI-Generated Summary")
    if st.button("Generate Summary"):
        with st.spinner("Asking Groq for a summary..."):
            try:
                from services.llm_service import generate_summary, LLMServiceError

                summary = generate_summary(context)
                st.session_state["ai_summary"] = summary
            except LLMServiceError as exc:
                ai_problem(exc)

    if st.session_state.get("ai_summary"):
        st.info(st.session_state["ai_summary"])

    st.divider()
    st.subheader("AI-Generated Recommendations")
    if st.button("Generate Recommendations"):
        with st.spinner("Asking Groq for recommendations..."):
            try:
                from services.llm_service import generate_recommendations, LLMServiceError

                recommendations = generate_recommendations(context)
                st.session_state["ai_recommendations"] = recommendations
            except LLMServiceError as exc:
                ai_problem(exc)

    recommendations = st.session_state.get("ai_recommendations")
    if recommendations:
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        for rec in sorted(recommendations, key=lambda r: priority_order.get(r.get("priority"), 3)):
            priority = rec.get("priority", "—")
            badge = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(priority, "⚪")
            with st.container(border=True):
                st.markdown(f"**{badge} {rec.get('issue', 'Untitled')}** — Priority: {priority}")
                st.write(f"**Evidence:** {rec.get('evidence', '—')}")
                st.write(f"**Recommended Action:** {rec.get('recommended_action', '—')}")
                st.write(f"**Expected Impact:** {rec.get('expected_impact', '—')}")
