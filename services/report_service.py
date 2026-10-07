"""
services/report_service.py
----------------------------
Report generation for Phase 9: CSV exports and a PDF business report.

Same rule as the rest of the app: every number in a report comes from
services/analytics.py or database/queries.py. The only generated text is
the optional AI summary / recommendations (services/llm_service.py), and
those are labelled as AI-generated inside the PDF.

Nothing here touches Streamlit or the database directly, so it is easy to
unit test with a plain DataFrame.
"""

import io
import json
from datetime import datetime
from typing import Optional
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from services.analytics import (
    compute_kpis,
    product_rating_extremes,
    sentiment_distribution,
)

SENTIMENT_COLORS = {"Positive": "#22C55E", "Neutral": "#94A3B8", "Negative": "#EF4444"}
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

LOOP_SCORE_NOTE = (
    "The LOOP Customer Satisfaction Score is a custom metric defined by this "
    "project (50% average rating, 50% sentiment mix). It is not an "
    "industry-standard metric."
)


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

EXPORT_COLUMNS = [
    "id",
    "customer_id",
    "customer_name",
    "product",
    "category",
    "rating",
    "feedback_text",
    "feedback_date",
    "sentiment",
    "sentiment_confidence",
    "emotion",
    "topic",
]


def feedback_to_csv(df: pd.DataFrame) -> bytes:
    """Feedback + analysis rows as UTF-8 CSV bytes (with BOM so Excel opens
    it correctly). Only known columns are exported, in a fixed order."""
    columns = [c for c in EXPORT_COLUMNS if c in df.columns]
    out = df[columns] if columns else df
    return out.to_csv(index=False).encode("utf-8-sig")


def product_summary_to_csv(df: pd.DataFrame) -> bytes:
    """One row per product: total feedback, min/max/avg rating."""
    return product_rating_extremes(df).to_csv(index=False).encode("utf-8-sig")


def complaints_to_csv(complaints: list) -> bytes:
    """Top recurring complaints (output of database.queries.get_top_complaints)."""
    rows = [
        {
            "topic": c["topic"],
            "mentions": c["count"],
            "percent_of_negative": c["percentage"],
            "related_products": ", ".join(c.get("related_products", [])),
        }
        for c in complaints
    ]
    columns = ["topic", "mentions", "percent_of_negative", "related_products"]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Complaints for a (possibly filtered) DataFrame
# ---------------------------------------------------------------------------

def complaints_from_df(df: pd.DataFrame, limit: int = 5) -> list:
    """Recurring complaints for exactly the rows in `df`, in the same shape
    as database.queries.get_top_complaints. That query always covers the
    whole database; reports can be filtered, so they use this instead so the
    complaints match the rest of the report."""
    if df.empty:
        return []
    negative = df[(df["sentiment"] == "Negative") & (df["topic"].notna())]
    if negative.empty:
        return []

    total_negative = len(negative)
    complaints = []
    for topic, group in negative.groupby("topic"):
        related = group["product"].value_counts().head(3).index.tolist()
        complaints.append(
            {
                "topic": topic,
                "count": int(len(group)),
                "percentage": round(len(group) / total_negative * 100, 1),
                "related_products": related,
            }
        )
    complaints.sort(key=lambda c: (-c["count"], c["topic"]))
    return complaints[:limit]


# ---------------------------------------------------------------------------
# Analytics context (what the LLM and the PDF both use)
# ---------------------------------------------------------------------------

def build_analytics_context(df: pd.DataFrame, complaints: list) -> dict:
    """Compact dict of already-computed numbers, in the same shape the AI
    Insights page sends to Groq."""
    kpis = compute_kpis(df)
    total = kpis["total_feedback"]

    def pct(n: int) -> float:
        return round(n / total * 100, 1) if total else 0

    return {
        "total_feedback": total,
        "positive_pct": pct(kpis["positive_feedback"]),
        "neutral_pct": pct(kpis["neutral_feedback"]),
        "negative_pct": pct(kpis["negative_feedback"]),
        "average_rating": kpis["average_rating"],
        "loop_satisfaction_score": kpis["loop_satisfaction_score"],
        "top_complaints": [
            f"{c['topic']} – {c['percentage']}% of negative feedback ({c['count']} mentions)"
            for c in complaints
        ],
    }


# ---------------------------------------------------------------------------
# Saved-report helpers (reports table stores recommendations as text)
# ---------------------------------------------------------------------------

def recommendations_to_text(recommendations: Optional[list]) -> str:
    """Serialize recommendations for the `reports.recommendations` column."""
    return json.dumps(recommendations or [], ensure_ascii=False)


def recommendations_from_text(text: Optional[str]) -> list:
    """Parse the stored column back into a list. Returns [] for anything
    that isn't a valid list of dicts (old rows, manual edits, bad data)."""
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [r for r in parsed if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("LoopTitle", parent=base["Title"], fontSize=22, spaceAfter=4, textColor=colors.HexColor("#0F172A")),
        "subtitle": ParagraphStyle("LoopSub", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#64748B"), spaceAfter=14),
        "h2": ParagraphStyle("LoopH2", parent=base["Heading2"], fontSize=14, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0F172A")),
        "body": ParagraphStyle("LoopBody", parent=base["Normal"], fontSize=10, leading=14),
        "small": ParagraphStyle("LoopSmall", parent=base["Normal"], fontSize=8, leading=11, textColor=colors.HexColor("#64748B")),
    }


def _table(data: list, col_widths: Optional[list] = None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _sentiment_chart(distribution: pd.DataFrame) -> Optional[Drawing]:
    """Small bar chart of sentiment counts. Returns None if there is nothing to draw."""
    if distribution.empty:
        return None
    order = [s for s in ["Positive", "Neutral", "Negative"] if s in set(distribution["sentiment"])]
    counts = {row["sentiment"]: int(row["count"]) for _, row in distribution.iterrows()}

    drawing = Drawing(400, 160)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 40, 25, 330, 115
    chart.data = [[counts[s] for s in order]]
    chart.categoryAxis.categoryNames = order
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(counts.values()) * 1.15 or 1
    chart.bars.strokeColor = None
    for i, s in enumerate(order):
        chart.bars[(0, i)].fillColor = colors.HexColor(SENTIMENT_COLORS[s])
    chart.categoryAxis.labels.fontSize = 9
    chart.valueAxis.labels.fontSize = 8
    drawing.add(chart)
    return drawing


def _p(text, style) -> Paragraph:
    """Paragraph with user/LLM text escaped so stray < or & can't break the PDF."""
    return Paragraph(escape(str(text)), style)


def generate_pdf(
    title: str,
    *,
    df: Optional[pd.DataFrame] = None,
    complaints: Optional[list] = None,
    summary: Optional[str] = None,
    recommendations: Optional[list] = None,
    filter_description: str = "",
    generated_at: Optional[datetime] = None,
) -> bytes:
    """Build the PDF report and return its bytes.

    `df` and `complaints` are optional: a report re-opened from the saved
    history only has the stored AI text, so the statistics sections are
    skipped rather than recomputed from today's (possibly different) data.
    """
    st = _styles()
    generated_at = generated_at or datetime.now()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=title,
        author="Project LOOP",
    )

    story = [
        _p(title, st["title"]),
        _p(f"Project LOOP  |  Generated {generated_at:%d %b %Y, %H:%M}", st["subtitle"]),
    ]
    if filter_description:
        story.append(_p(f"Scope: {filter_description}", st["small"]))
        story.append(Spacer(1, 6))

    if df is not None and not df.empty:
        kpis = compute_kpis(df)
        story.append(_p("Key Metrics", st["h2"]))
        story.append(
            _table(
                [
                    ["Metric", "Value"],
                    ["Total feedback", kpis["total_feedback"]],
                    ["Positive", kpis["positive_feedback"]],
                    ["Neutral", kpis["neutral_feedback"]],
                    ["Negative", kpis["negative_feedback"]],
                    ["Average rating", f"{kpis['average_rating']:.2f} / 5"],
                    ["LOOP Satisfaction Score", kpis["loop_satisfaction_score"]],
                ],
                col_widths=[8 * cm, 5 * cm],
            )
        )
        story.append(Spacer(1, 4))
        story.append(_p(LOOP_SCORE_NOTE, st["small"]))

        chart = _sentiment_chart(sentiment_distribution(df))
        if chart is not None:
            story.append(_p("Sentiment Distribution", st["h2"]))
            story.append(chart)

        products = product_rating_extremes(df)
        if not products.empty:
            story.append(_p("Product Ratings", st["h2"]))
            rows = [["Product", "Feedback", "Lowest", "Highest", "Average"]]
            for _, r in products.iterrows():
                rows.append([str(r["product"]), int(r["total_feedback"]), int(r["min_rating"]), int(r["max_rating"]), f"{r['avg_rating']:.2f}"])
            story.append(_table(rows, col_widths=[6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]))

    if complaints:
        story.append(_p("Top Recurring Complaints", st["h2"]))
        rows = [["Topic", "Mentions", "% of negative", "Most affected products"]]
        for c in complaints:
            rows.append(
                [
                    str(c["topic"]),
                    c["count"],
                    f"{c['percentage']}%",
                    Paragraph(escape(", ".join(c.get("related_products", [])) or "-"), st["body"]),
                ]
            )
        story.append(_table(rows, col_widths=[4 * cm, 2.2 * cm, 2.8 * cm, 8 * cm]))

    if summary:
        story.append(_p("AI-Generated Summary", st["h2"]))
        story.append(_p(summary, st["body"]))

    if recommendations:
        story.append(_p("AI-Generated Recommendations", st["h2"]))
        ordered = sorted(recommendations, key=lambda r: PRIORITY_ORDER.get(r.get("priority"), 3))
        for rec in ordered:
            story.append(_p(f"{rec.get('issue', 'Untitled')}  (Priority: {rec.get('priority', '-')})", st["body"]))
            story.append(_p(f"Evidence: {rec.get('evidence', '-')}", st["small"]))
            story.append(_p(f"Recommended action: {rec.get('recommended_action', '-')}", st["small"]))
            story.append(_p(f"Expected impact: {rec.get('expected_impact', '-')}", st["small"]))
            story.append(Spacer(1, 8))

    if summary or recommendations:
        story.append(
            _p(
                "The summary and recommendations above were generated by an AI model "
                "(Groq) from the statistics in this report. Review them before acting on them.",
                st["small"],
            )
        )

    has_content = (df is not None and not df.empty) or complaints or summary or recommendations
    if not has_content:
        story.append(_p("This report has no data to show.", st["body"]))

    doc.build(story)
    return buffer.getvalue()
