"""
ui/theme.py
-------------
Shared look and feel for LOOP Insights: global CSS, page headers, card
containers, and one consistent style for every Plotly chart.

Pages call `inject_css()` once (app.py does this), `page_header()` at the
top of each page, and `show_chart(fig)` instead of st.plotly_chart.
"""

import plotly.graph_objects as go
import streamlit as st

from config import config

# Palette
BG = "#0B1020"
PANEL = "#121A30"
BORDER = "#25304D"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
ACCENT = "#6366F1"
ACCENT_2 = "#22D3EE"

SENTIMENT_COLORS = {"Positive": "#22C55E", "Neutral": "#94A3B8", "Negative": "#EF4444"}
COLORWAY = ["#6366F1", "#22D3EE", "#F59E0B", "#EC4899", "#22C55E", "#A78BFA", "#F97316", "#14B8A6"]

_CSS = f"""
<style>
/* --- layout ---------------------------------------------------------- */
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h3 {{ font-size: 1.05rem !important; color: {TEXT}; }}

/* --- sidebar --------------------------------------------------------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0E1530 0%, #0B1020 100%);
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 9px 12px; border-radius: 10px; width: 100%;
    transition: background .15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(99,102,241,.12); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(90deg, rgba(99,102,241,.30), rgba(34,211,238,.10));
    border: 1px solid rgba(99,102,241,.45);
}}

.loop-brand {{ display: flex; align-items: center; gap: 12px; padding: 4px 4px 14px 4px; }}
.loop-logo {{
    width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
    font-size: 22px; color: white; font-weight: 700;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
    box-shadow: 0 6px 20px rgba(99,102,241,.45);
}}
.loop-brand-name {{ font-weight: 700; font-size: 1.15rem; line-height: 1.1; color: white; }}
.loop-brand-sub {{ font-size: .72rem; color: {MUTED}; letter-spacing: .04em; text-transform: uppercase; }}

/* --- page header ----------------------------------------------------- */
.loop-page-header {{
    padding: 22px 26px; border-radius: 16px; margin-bottom: 22px;
    background: radial-gradient(1200px 200px at 0% 0%, rgba(99,102,241,.28), transparent 60%),
                radial-gradient(800px 200px at 100% 100%, rgba(34,211,238,.14), transparent 60%),
                {PANEL};
    border: 1px solid {BORDER};
}}
.loop-page-header h1 {{ margin: 0; padding: 0; font-size: 1.7rem; color: white; }}
.loop-page-header p {{ margin: 4px 0 0 0; color: {MUTED}; font-size: .95rem; }}

/* --- hero (home) ------------------------------------------------------ */
.loop-hero {{
    padding: 44px 40px; border-radius: 22px; margin-bottom: 26px;
    background: radial-gradient(900px 340px at 0% 0%, rgba(99,102,241,.40), transparent 60%),
                radial-gradient(700px 300px at 100% 100%, rgba(34,211,238,.22), transparent 60%),
                {PANEL};
    border: 1px solid {BORDER};
}}
.loop-hero .eyebrow {{ color: {ACCENT_2}; font-size: .78rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; }}
.loop-hero h1 {{ font-size: 2.6rem; margin: 8px 0 10px 0; color: white; line-height: 1.1; }}
.loop-hero p {{ color: #CBD5E1; font-size: 1.05rem; max-width: 720px; margin: 0; }}

/* --- cards ------------------------------------------------------------ */
[class*="st-key-card"] {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 18px 20px; margin-bottom: 4px;
}}
.loop-feature {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: 20px; min-height: 190px;
    transition: transform .15s ease, border-color .15s ease;
}}
.loop-feature:hover {{ transform: translateY(-2px); border-color: {ACCENT}; }}
.loop-feature .icon {{ font-size: 1.6rem; }}
.loop-feature h4 {{ margin: 10px 0 6px 0; color: white; font-size: 1.05rem; }}
.loop-feature p {{ margin: 0; color: {MUTED}; font-size: .9rem; }}

.loop-step {{ text-align: center; padding: 14px 6px; }}
.loop-step .num {{
    width: 34px; height: 34px; border-radius: 50%; margin: 0 auto 8px auto; display: grid; place-items: center;
    font-weight: 700; color: white; background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
}}
.loop-step .label {{ font-size: .85rem; color: {TEXT}; font-weight: 600; }}
.loop-step .desc {{ font-size: .75rem; color: {MUTED}; }}

/* --- metrics ---------------------------------------------------------- */
[data-testid="stMetric"] {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px;
    border-top: 3px solid {ACCENT};
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }}
[data-testid="stMetricValue"] {{ font-weight: 700; color: white; }}

/* --- controls --------------------------------------------------------- */
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 600; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT}, #4F46E5); border: none;
}}
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 12px; background: {PANEL}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
[data-testid="stAlert"] {{ border-radius: 12px; }}
hr {{ border-color: {BORDER}; }}

/* --- empty state ------------------------------------------------------- */
.loop-empty {{
    text-align: center; padding: 36px 20px; margin: 8px 0 14px 0; border-radius: 16px;
    background: {PANEL}; border: 1px dashed {BORDER};
}}
.loop-empty .icon {{ font-size: 2rem; }}
.loop-empty h3 {{ margin: 8px 0 4px 0; color: white; }}
.loop-empty p {{ margin: 0; color: {MUTED}; font-size: .92rem; }}

/* --- accessibility ------------------------------------------------------ */
button:focus-visible, [role="radiogroup"] label:has(input:focus-visible), a:focus-visible {{
    outline: 2px solid {ACCENT_2} !important; outline-offset: 2px;
}}
@media (prefers-reduced-motion: reduce) {{
    * {{ transition: none !important; animation: none !important; }}
}}

/* --- small screens ------------------------------------------------------ */
@media (max-width: 768px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    .loop-hero {{ padding: 26px 20px; }}
    .loop-hero h1 {{ font-size: 1.8rem; }}
    .loop-page-header {{ padding: 16px 18px; }}
    .loop-page-header h1 {{ font-size: 1.35rem; }}
    .loop-feature {{ min-height: 0; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def brand_block() -> str:
    return (
        '<div class="loop-brand"><div class="loop-logo">∞</div>'
        f'<div><div class="loop-brand-name">{config.APP_TITLE}</div>'
        f'<div class="loop-brand-sub">{config.APP_SUBTITLE}</div></div></div>'
    )


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="loop-page-header"><h1>{icon} {title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def _go(page: str) -> None:
    st.session_state["nav"] = page
    st.rerun()


def no_data_state() -> None:
    """Shown on any page that needs feedback when the database is empty."""
    st.markdown(
        '<div class="loop-empty"><div class="icon">📭</div><h3>No feedback yet</h3>'
        "<p>Add some feedback and this page fills in automatically.</p></div>",
        unsafe_allow_html=True,
    )
    left, right, _ = st.columns([1, 1, 3])
    if left.button("Upload a file", type="primary", key="empty_go_upload"):
        _go("📤 Upload Feedback")
    if right.button("Enter one manually", key="empty_go_manual"):
        _go("✍️ Manual Feedback")


AI_UNAVAILABLE = "AI features are unavailable right now. Please try again later."


def problem(client_message: str, admin_message: str | None = None, exc: Exception | None = None, level: str = "error") -> None:
    """Show a problem. Clients get a short plain message. The admin also gets
    the technical reason (missing key, database error) so it can be fixed."""
    from ui.admin_gate import is_admin

    show = st.warning if level == "warning" else st.error
    if not is_admin():
        show(client_message)
        return
    show(admin_message or client_message)
    if exc is not None:
        st.caption(f"Details: {exc}")


def ai_problem(exc: Exception | str | None = None, admin_message: str | None = None, level: str = "error") -> None:
    """Groq is missing or failed. Clients see a neutral message; the admin sees why."""
    reason = admin_message or (str(exc) if exc is not None else None)
    problem(AI_UNAVAILABLE, reason, level=level)


def load_error(exc: Exception) -> None:
    """Shown when feedback can't be read from MySQL. Says what to do next."""
    from ui.admin_gate import is_admin

    if not is_admin():
        st.error("We couldn't load the data right now. Please try again in a moment.")
        return
    st.error(
        "Couldn't load feedback data. Check that MySQL is running and the tables exist, "
        "then test the connection in Settings."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    if st.button("Open Settings", key="error_go_settings"):
        _go("⚙️ Settings")


def card(name: str):
    """A bordered card container. `name` must be unique on the page."""
    return st.container(key=f"card_{name}")


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=12),
        margin=dict(l=8, r=8, t=24, b=8),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    has_pie = False
    for trace in fig.data:
        if trace.type == "pie":
            has_pie = True
            trace.hole = 0.58
            trace.textinfo = "percent"
            trace.textfont = dict(color="white", size=12)
            trace.marker.line = dict(color=BG, width=2)
    if has_pie:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5, font=dict(size=11)))
    return fig


def show_chart(fig: go.Figure, height: int = 340) -> None:
    st.plotly_chart(style_fig(fig, height), width="stretch", theme=None)


def score_gauge(value: float, title: str = "LOOP Satisfaction Score") -> go.Figure:
    """Half-dial gauge for a 0-100 score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(font=dict(size=40, color="white")),
            title=dict(text=title, font=dict(size=14, color=MUTED)),
            gauge=dict(
                axis=dict(range=[0, 100], tickmode="array", tickvals=[0, 50, 100], tickcolor=MUTED),
                bar=dict(color=ACCENT, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color="rgba(239,68,68,.25)"),
                    dict(range=[50, 75], color="rgba(245,158,11,.22)"),
                    dict(range=[75, 100], color="rgba(34,197,94,.25)"),
                ],
            ),
        )
    )
    return style_fig(fig, height=250)
