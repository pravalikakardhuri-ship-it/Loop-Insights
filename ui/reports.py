"""
ui/reports.py
---------------
The Reports page (Phase 9): CSV exports, a downloadable PDF business
report with optional AI summary/recommendations, and a history of saved
reports (stored in the `reports` table).
"""

from datetime import datetime

import streamlit as st

from ui.period_picker import render_period_picker
from ui.theme import load_error, problem, no_data_state, page_header

from config import config
from database.queries import get_all_reports, insert_report
from services import report_service
from services.analytics import Filters, apply_filters, load_feedback_df


def _scope_description(products: list, sentiments: list, period_text: str = "All time") -> str:
    parts = []
    if period_text != "All time":
        parts.append("Period: " + period_text)
    if products:
        parts.append("Products: " + ", ".join(products))
    if sentiments:
        parts.append("Sentiment: " + ", ".join(sentiments))
    return " | ".join(parts) if parts else "All feedback"


def _render_exports(df) -> None:
    st.subheader("CSV Exports")
    stamp = datetime.now().strftime("%Y%m%d")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "⬇️ Feedback + analysis",
        data=report_service.feedback_to_csv(df),
        file_name=f"loop_feedback_{stamp}.csv",
        mime="text/csv",
    )
    col2.download_button(
        "⬇️ Product summary",
        data=report_service.product_summary_to_csv(df),
        file_name=f"loop_products_{stamp}.csv",
        mime="text/csv",
    )
    complaints = report_service.complaints_from_df(df, limit=10)
    col3.download_button(
        "⬇️ Top complaints",
        data=report_service.complaints_to_csv(complaints),
        file_name=f"loop_complaints_{stamp}.csv",
        mime="text/csv",
        disabled=not complaints,
    )


def _build_report(title: str, df, scope: str, use_ai: bool) -> dict:
    """Compute everything the report needs. AI calls are optional, and a
    Groq failure never blocks the (statistics-only) report."""
    complaints = report_service.complaints_from_df(df, limit=5)
    summary, recommendations, ai_error = None, [], None

    if use_ai:
        from services.llm_service import LLMServiceError, generate_recommendations, generate_summary

        context = report_service.build_analytics_context(df, complaints)
        try:
            summary = generate_summary(context)
            recommendations = generate_recommendations(context)
        except LLMServiceError as exc:
            ai_error = str(exc)

    pdf = report_service.generate_pdf(
        title,
        df=df,
        complaints=complaints,
        summary=summary,
        recommendations=recommendations,
        filter_description=scope,
    )
    return {
        "title": title,
        "pdf": pdf,
        "summary": summary,
        "recommendations": recommendations,
        "ai_error": ai_error,
    }


def _render_pdf_builder(df, scope: str) -> None:
    st.subheader("PDF Report")
    title = st.text_input("Report title", value=f"Customer Feedback Report - {datetime.now():%d %b %Y}")

    has_key = bool(config.GROQ_API_KEY)
    use_ai = st.checkbox(
        "Include AI summary and recommendations (uses Groq)",
        value=has_key,
        disabled=not has_key,
        help=None if has_key else "AI sections are unavailable right now.",
    )

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Building report..."):
            try:
                st.session_state["report_result"] = _build_report(title.strip() or "Customer Feedback Report", df, scope, use_ai)
            except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
                st.session_state.pop("report_result", None)
                problem("Could not build the report. Please try again.", exc=exc)

    result = st.session_state.get("report_result")
    if not result:
        return

    if result["ai_error"]:
        problem(
            "The report was built without the AI sections because AI is unavailable right now.",
            f"AI sections were skipped: {result['ai_error']}",
            level="warning",
        )
    else:
        st.success("Report ready.")

    col1, col2 = st.columns(2)
    col1.download_button(
        "⬇️ Download PDF",
        data=result["pdf"],
        file_name=f"loop_report_{datetime.now():%Y%m%d_%H%M}.pdf",
        mime="application/pdf",
    )
    can_save = bool(result["summary"] or result["recommendations"])
    if col2.button("💾 Save to report history", disabled=not can_save):
        try:
            insert_report(
                result["title"],
                result["summary"] or "",
                report_service.recommendations_to_text(result["recommendations"]),
            )
            st.toast("Saved to report history.")
        except Exception as exc:  # noqa: BLE001
            problem("Could not save the report. Please try again.", exc=exc)
    if not can_save:
        st.caption("Only reports with AI content can be saved to history (the statistics are always recomputable from your data).")


def _render_history() -> None:
    st.subheader("Saved Reports")
    try:
        reports = get_all_reports()
    except Exception as exc:  # noqa: BLE001
        problem("Could not load saved reports. Please try again in a moment.",
                "Could not load saved reports. Make sure MySQL is connected (see Settings).", exc)
        return

    if not reports:
        st.caption("No saved reports yet.")
        return

    for r in reports:
        created = r["created_at"].strftime("%d %b %Y, %H:%M") if r["created_at"] else ""
        with st.expander(f"{r['title']}  ({created})"):
            recs = report_service.recommendations_from_text(r["recommendations"])
            if r["summary"]:
                st.write(r["summary"])
            for rec in recs:
                st.markdown(f"- **{rec.get('issue', 'Untitled')}** ({rec.get('priority', '-')}): {rec.get('recommended_action', '-')}")
            pdf = report_service.generate_pdf(
                r["title"],
                summary=r["summary"],
                recommendations=recs,
                generated_at=r["created_at"],
            )
            st.download_button(
                "⬇️ Download PDF",
                data=pdf,
                file_name=f"loop_report_{r['id']}.pdf",
                mime="application/pdf",
                key=f"dl_report_{r['id']}",
            )


def render() -> None:
    page_header("📑", "Reports", "Export data and generate PDF reports.")

    try:
        full_df = load_feedback_df()
    except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
        load_error(exc)
        return

    if full_df.empty:
        no_data_state()
        return

    period = render_period_picker(full_df)
    with st.expander("🔎 Filter the data in the exports and PDF", expanded=False):
        products = st.multiselect("Product", sorted(full_df["product"].dropna().unique()))
        sentiments = st.multiselect("Sentiment", config.SENTIMENT_LABELS)

    df = apply_filters(full_df, Filters(products=products, sentiments=sentiments, date_range=period.date_range))
    st.caption(f"{len(df)} of {len(full_df)} records selected.")
    if df.empty:
        st.warning("No records match those filters.")
        return

    _render_exports(df)
    st.divider()
    _render_pdf_builder(df, _scope_description(products, sentiments, period.text))
    st.divider()
    _render_history()
