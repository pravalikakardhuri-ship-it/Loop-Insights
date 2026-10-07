"""
app.py
------
Entry point for LOOP Insights: page setup, sidebar navigation and routing.
Each page lives in its own module under ui/ and exposes a render() function.
"""

import streamlit as st

from config import config
from ui import theme

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="∞",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "🏠 Home",
    "📊 Dashboard",
    "📤 Upload Feedback",
    "✍️ Manual Feedback",
    "🔎 Feedback Explorer",
    "🤖 AI Insights",
    "💬 LOOP Assistant",
    "📈 Product Analysis",
    "📑 Reports",
    "⚙️ Settings",
]
SETTINGS_PAGE = "⚙️ Settings"
PUBLIC_PAGES = [page for page in PAGES if page != SETTINGS_PAGE]  # what clients see


def _home_stats() -> dict | None:
    """Live numbers for the home page, or None if the database isn't ready
    (the home page must still load without MySQL)."""
    try:
        from services.analytics import compute_kpis, load_feedback_df

        df = load_feedback_df()
        if df.empty:
            return None
        kpis = compute_kpis(df)
        kpis["products"] = int(df["product"].nunique())
        return kpis
    except Exception:  # noqa: BLE001
        return None


def render_home() -> None:
    st.markdown(
        f"""
        <div class="loop-hero">
          <div class="eyebrow">{config.APP_SUBTITLE}</div>
          <h1>{config.APP_TAGLINE}</h1>
          <p>{config.APP_TITLE} turns raw customer feedback into clear priorities. It combines
          machine learning and generative AI to show what customers feel, which problems keep
          coming back, and what the business should fix first.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = _home_stats()
    if stats:
        cols = st.columns(5)
        cols[0].metric("Feedback analysed", f"{stats['total_feedback']:,}")
        cols[1].metric("Products", stats["products"])
        cols[2].metric("Average rating", f"{stats['average_rating']:.2f} / 5")
        cols[3].metric("Positive", f"{stats['positive_feedback']:,}")
        cols[4].metric("LOOP score", f"{stats['loop_satisfaction_score']:.1f}")
        st.write("")
    else:
        from ui.admin_gate import is_admin

        if is_admin():
            st.info(
                "No data to show yet. Open **Settings** to connect MySQL and create the tables, "
                "then load the sample dataset or upload your own feedback."
            )
        else:
            st.info("No data yet. Open **Upload Feedback** to add your first file.")

    features = [
        ("🧠", "AI Analysis", "Sentiment, emotion and topic detection on every piece of feedback."),
        ("📊", "Smart Analytics", "Interactive dashboard with filters, trends and recurring complaints."),
        ("💡", "AI Recommendations", "Prioritised actions with the evidence behind each one."),
        ("💬", "Insight Assistant", "Ask questions in plain language, answered from your own data."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f'<div class="loop-feature"><div class="icon">{icon}</div><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("##### How it works")
    steps = [
        ("Collect", "Upload or enter"),
        ("Clean", "Validate data"),
        ("Analyse", "ML + NLP"),
        ("Visualise", "Dashboard"),
        ("Recommend", "Groq AI"),
        ("Report", "PDF / CSV"),
    ]
    cols = st.columns(len(steps))
    for i, (col, (label, desc)) in enumerate(zip(cols, steps), start=1):
        col.markdown(
            f'<div class="loop-step"><div class="num">{i}</div>'
            f'<div class="label">{label}</div><div class="desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right, _ = st.columns([1, 1, 3])
    if left.button("Open Dashboard", type="primary"):
        st.session_state["nav"] = "📊 Dashboard"
        st.rerun()
    if right.button("Upload Feedback"):
        st.session_state["nav"] = "📤 Upload Feedback"
        st.rerun()

    from ui.admin_gate import is_admin

    warnings = config.validate() if is_admin() else []  # server details are for the admin only
    if warnings:
        with st.expander("⚠️ Setup notices", expanded=False):
            for w in warnings:
                st.warning(w)


def render_settings() -> None:
    from ui import settings

    settings.render()


def main() -> None:
    from ui.admin_gate import is_admin, render_admin_box

    theme.inject_css()
    st.sidebar.markdown(theme.brand_block(), unsafe_allow_html=True)

    pages = PAGES if is_admin() else PUBLIC_PAGES
    default_page = st.session_state.get("nav", pages[0])
    default_index = pages.index(default_page) if default_page in pages else 0
    selected = st.sidebar.radio("Navigate", pages, index=default_index, label_visibility="collapsed")
    st.session_state["nav"] = selected
    st.sidebar.divider()
    render_admin_box()
    st.sidebar.caption(f"{config.APP_TITLE} v1.0")

    if selected == "🏠 Home":
        render_home()
    elif selected == SETTINGS_PAGE and is_admin():
        render_settings()
    elif selected == "📊 Dashboard":
        from ui import dashboard

        dashboard.render()
    elif selected == "📤 Upload Feedback":
        from ui import upload_feedback

        upload_feedback.render()
    elif selected == "✍️ Manual Feedback":
        from ui import manual_feedback

        manual_feedback.render()
    elif selected == "🔎 Feedback Explorer":
        from ui import feedback_explorer

        feedback_explorer.render()
    elif selected == "🤖 AI Insights":
        from ui import ai_insights

        ai_insights.render()
    elif selected == "💬 LOOP Assistant":
        from ui import loop_assistant

        loop_assistant.render()
    elif selected == "📈 Product Analysis":
        from ui import product_analysis

        product_analysis.render()
    elif selected == "📑 Reports":
        from ui import reports

        reports.render()


if __name__ == "__main__":
    main()
