"""
services/assistant_context.py
--------------------------------
Builds the context handed to the LLM for the LOOP Insight Assistant
(spec section 22). Follows the spec's own diagram exactly:

    User Question -> Retrieve relevant database analytics ->
    Create structured context -> Send context to Groq -> Generate answer

This module only does the retrieval/context-building step. The actual
Groq call lives in services/llm_service.answer_question(), which already
enforces "don't invent numbers, say so if data is missing" at the prompt
level -- this module's job is just to make sure that context is honest
and grounded in real stored data, never fabricated.

Retrieval approach: rather than full embedding-based RAG (spec section 23
marks this optional), relevant individual feedback snippets are retrieved
by simple keyword overlap between the question and stored feedback text.
This is simpler than FAISS/embeddings, needs no extra model download, and
is transparent about how a snippet got selected -- a reasonable choice
for a mini-project's scope. It can be swapped for real embedding search
later without changing anything outside this module.
"""

import re

import pandas as pd

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "whom", "this", "that", "these", "those", "and", "or", "but", "for",
    "of", "to", "in", "on", "at", "by", "with", "about", "our", "we",
    "us", "customers", "customer", "feedback", "do", "does", "did",
    "have", "has", "had", "be", "been", "being", "it", "its", "as",
}
_WORD_PATTERN = re.compile(r"[a-z']+")


def _keywords(text: str) -> set:
    words = _WORD_PATTERN.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _retrieve_relevant_snippets(df: pd.DataFrame, question: str, limit: int = 6) -> list[dict]:
    """Keyword-overlap retrieval of individual feedback rows relevant to
    the question -- e.g. a question mentioning "delivery" pulls feedback
    rows whose text mentions delivery-related words."""
    question_words = _keywords(question)
    if not question_words or df.empty:
        return []

    def score(text: str) -> int:
        return len(_keywords(text) & question_words)

    working = df.copy()
    working["_score"] = working["feedback_text"].fillna("").apply(score)
    matches = working[working["_score"] > 0].sort_values("_score", ascending=False).head(limit)

    return [
        {
            "product": row["product"],
            "sentiment": row.get("sentiment"),
            "topic": row.get("topic"),
            "text": row["feedback_text"],
        }
        for _, row in matches.iterrows()
    ]


def _detect_mentioned_product(question: str, known_products: list) -> str | None:
    """Case-insensitive check for whether the question names a specific
    product (e.g. "What are customers saying about the Smartwatch?").
    When one is found, the assistant gets a dedicated, focused block of
    that product's own stats and sample feedback -- rather than relying
    purely on keyword-overlap text search, which can miss relevant rows
    whose text never mentions the product name at all.
    """
    question_lower = question.lower()
    for product in known_products:
        if product.lower() in question_lower:
            return product
    return None


def _product_focus_block(df: pd.DataFrame, product: str) -> dict:
    """Detailed, dedicated stats + sample feedback for one specific
    product, used when the question names it directly."""
    product_df = df[df["product"] == product]
    total = len(product_df)
    if total == 0:
        return {"product": product, "note": "No feedback found for this product."}

    sentiments = product_df["sentiment"].dropna()
    sentiment_total = len(sentiments) if len(sentiments) > 0 else 1
    positive_samples = product_df[product_df["sentiment"] == "Positive"]["feedback_text"].dropna().head(3).tolist()
    negative_samples = product_df[product_df["sentiment"] == "Negative"]["feedback_text"].dropna().head(3).tolist()

    return {
        "product": product,
        "total_feedback": total,
        "positive_pct": round((sentiments == "Positive").sum() / sentiment_total * 100, 1),
        "negative_pct": round((sentiments == "Negative").sum() / sentiment_total * 100, 1),
        "neutral_pct": round((sentiments == "Neutral").sum() / sentiment_total * 100, 1),
        "average_rating": round(float(product_df["rating"].mean()), 2) if product_df["rating"].notna().any() else None,
        "sample_positive_feedback": positive_samples,
        "sample_negative_feedback": negative_samples,
    }


def build_assistant_context(df: pd.DataFrame, question: str) -> dict | None:
    """Build the structured context dict for a given question.

    Returns None if there's no feedback data at all -- callers should
    treat that as "nothing to answer from" and show the standard
    not-enough-information message without even calling the LLM.
    """
    from services.analytics import (
        compute_kpis, compute_loop_satisfaction_score, product_rating_extremes,
    )

    if df.empty:
        return None

    kpis = compute_kpis(df)
    score = compute_loop_satisfaction_score(df)
    total = kpis["total_feedback"]

    # Per-product breakdown. Percentages and counts for ALL THREE sentiment
    # classes are pre-computed here rather than left for the LLM to derive
    # from raw counts -- language models are not reliable at arithmetic, so
    # doing the math in Python (and just handing over the answer) removes
    # an entire class of "inaccurate answer" bug.
    product_rows = []
    for product, group in df.groupby("product"):
        sentiments = group["sentiment"].dropna()
        p_total = len(sentiments) if len(sentiments) > 0 else 1
        pos_pct = round((sentiments == "Positive").sum() / p_total * 100, 1)
        neg_pct = round((sentiments == "Negative").sum() / p_total * 100, 1)
        neu_pct = round((sentiments == "Neutral").sum() / p_total * 100, 1)
        avg_rating = round(group["rating"].mean(), 2) if group["rating"].notna().any() else "N/A"
        product_rows.append(
            f"{product}: {len(group)} feedback records, avg rating {avg_rating}/5, "
            f"{pos_pct}% positive / {neu_pct}% neutral / {neg_pct}% negative"
        )

    from database.queries import get_top_complaints

    complaints = get_top_complaints(limit=8)
    extremes = product_rating_extremes(df)

    context = {
        "total_feedback": total,
        "positive_count": kpis["positive_feedback"],
        "positive_pct": round(kpis["positive_feedback"] / total * 100, 1) if total else 0,
        "negative_count": kpis["negative_feedback"],
        "negative_pct": round(kpis["negative_feedback"] / total * 100, 1) if total else 0,
        "neutral_count": kpis["neutral_feedback"],
        "neutral_pct": round(kpis["neutral_feedback"] / total * 100, 1) if total else 0,
        "average_rating": kpis["average_rating"],
        "loop_satisfaction_score": score["loop_score"],
        "feedback_by_product": product_rows,
        "top_complaints_overall": [
            f"{c['topic']} - {c['percentage']}% of negative feedback ({c['count']} mentions), "
            f"top products: {', '.join(c['related_products'])}"
            for c in complaints
        ],
    }

    if not extremes.empty:
        best = extremes.iloc[0]
        worst = extremes.iloc[-1]
        context["highest_rated_product"] = f"{best['product']} (avg {best['avg_rating']}/5)"
        context["lowest_rated_product"] = f"{worst['product']} (avg {worst['avg_rating']}/5)"

    emotions = df["emotion"].dropna()
    if not emotions.empty:
        context["emotion_breakdown"] = [
            f"{emotion}: {count} mentions" for emotion, count in emotions.value_counts().items()
        ]

    known_products = sorted(df["product"].dropna().unique().tolist())
    mentioned_product = _detect_mentioned_product(question, known_products)
    if mentioned_product:
        context["focused_product_stats"] = _product_focus_block(df, mentioned_product)

    snippets = _retrieve_relevant_snippets(df, question)
    if snippets:
        context["relevant_feedback_snippets"] = [
            f'[{s["product"]} | {s["sentiment"] or "unanalyzed"} | {s["topic"] or "no topic"}] "{s["text"]}"'
            for s in snippets
        ]

    return context
