"""
database/queries.py
--------------------
Clean CRUD helpers built on top of database/models.py.

Every function here takes plain Python types in and returns plain Python
types out (dicts / lists / ints) so calling code -- Streamlit pages,
services, tests -- never has to touch a SQLAlchemy session directly.
"""

import functools
from datetime import date, datetime
from typing import Optional

from sqlalchemy import delete, insert, select

from database import get_session
from database.models import Feedback, FeedbackAnalysis, Report


def _invalidate_cache() -> None:
    """Tell the analytics layer its cached DataFrame is now out of date."""
    from services.analytics import clear_feedback_cache

    clear_feedback_cache()


def _invalidates_cache(func):
    """Decorator for write functions: clear the analytics cache afterwards."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            _invalidate_cache()
    return wrapper


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@_invalidates_cache
def insert_feedback(
    customer_id: str,
    product: str,
    category: str,
    rating: int,
    feedback_text: str,
    customer_name: Optional[str] = None,
    feedback_date: Optional[date] = None,
) -> int:
    """Insert one feedback row. Returns the new row's id."""
    with get_session() as session:
        row = Feedback(
            customer_id=customer_id,
            customer_name=customer_name,
            product=product,
            category=category,
            rating=rating,
            feedback_text=feedback_text,
            feedback_date=feedback_date,
        )
        session.add(row)
        session.flush()  # populate row.id before commit
        return row.id


@_invalidates_cache
def bulk_insert_feedback(records: list[dict], chunk_size: int = 500) -> list[int]:
    """Insert many feedback rows in batches (e.g. from an uploaded CSV).

    Sends rows to the database in chunks instead of one at a time, which is
    much faster when the database is remote (e.g. Aiven). Returns the new
    row ids in the same order as `records`.

    Each dict should have keys matching insert_feedback's fields:
    customer_id, product, category, rating, feedback_text, and optionally
    customer_name, feedback_date.
    """
    if not records:
        return []

    table = Feedback.__table__
    columns = ["customer_id", "customer_name", "product", "category",
               "rating", "feedback_text", "feedback_date"]
    rows = [{c: record.get(c) for c in columns} for record in records]

    with get_session() as session:
        for start in range(0, len(rows), chunk_size):
            session.execute(insert(table), rows[start:start + chunk_size])
        # Ids are assigned in insert order; read back the newest N ids.
        newest = session.scalars(
            select(table.c.id).order_by(table.c.id.desc()).limit(len(rows))
        ).all()
    return list(reversed(newest))


def get_feedback_by_id(feedback_id: int) -> Optional[dict]:
    with get_session() as session:
        row = session.get(Feedback, feedback_id)
        return _feedback_to_dict(row) if row else None


def get_all_feedback(
    product: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Retrieve feedback rows, optionally filtered by product/category.

    Richer filtering (sentiment, date range, rating, topic) is added in
    Phase 6 - Dashboard once feedback_analysis is populated by Phase 4/5.
    """
    with get_session() as session:
        stmt = select(Feedback).order_by(Feedback.created_at.desc())
        if product:
            stmt = stmt.where(Feedback.product == product)
        if category:
            stmt = stmt.where(Feedback.category == category)
        if limit:
            stmt = stmt.limit(limit)
        rows = session.scalars(stmt).all()
        return [_feedback_to_dict(row) for row in rows]


def count_feedback() -> int:
    with get_session() as session:
        return session.query(Feedback).count()


def _feedback_to_dict(row: Feedback) -> dict:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "customer_name": row.customer_name,
        "product": row.product,
        "category": row.category,
        "rating": row.rating,
        "feedback_text": row.feedback_text,
        "feedback_date": row.feedback_date,
        "created_at": row.created_at,
    }


# ---------------------------------------------------------------------------
# Feedback Analysis
# ---------------------------------------------------------------------------

@_invalidates_cache
def insert_analysis(
    feedback_id: int,
    sentiment: Optional[str] = None,
    sentiment_confidence: Optional[float] = None,
    emotion: Optional[str] = None,
    emotion_confidence: Optional[float] = None,
    topic: Optional[str] = None,
    processed_text: Optional[str] = None,
) -> int:
    """Insert (or replace) the analysis row for a given feedback_id."""
    with get_session() as session:
        existing = session.scalar(
            select(FeedbackAnalysis).where(FeedbackAnalysis.feedback_id == feedback_id)
        )
        if existing:
            session.delete(existing)
            session.flush()

        row = FeedbackAnalysis(
            feedback_id=feedback_id,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            topic=topic,
            processed_text=processed_text,
        )
        session.add(row)
        session.flush()
        return row.id


@_invalidates_cache
def bulk_insert_analysis(analyses: list[dict], chunk_size: int = 500) -> None:
    """Insert (or replace) many analysis rows in batches.

    Each dict needs `feedback_id` and may include sentiment,
    sentiment_confidence, emotion, emotion_confidence, topic, processed_text.
    """
    if not analyses:
        return

    table = FeedbackAnalysis.__table__
    columns = ["feedback_id", "sentiment", "sentiment_confidence", "emotion",
               "emotion_confidence", "topic", "processed_text"]
    rows = [{c: a.get(c) for c in columns} for a in analyses]

    with get_session() as session:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            session.execute(
                delete(table).where(table.c.feedback_id.in_([r["feedback_id"] for r in chunk]))
            )
            session.execute(insert(table), chunk)


def get_analysis_by_feedback_id(feedback_id: int) -> Optional[dict]:
    with get_session() as session:
        row = session.scalar(
            select(FeedbackAnalysis).where(FeedbackAnalysis.feedback_id == feedback_id)
        )
        if not row:
            return None
        return {
            "id": row.id,
            "feedback_id": row.feedback_id,
            "sentiment": row.sentiment,
            "sentiment_confidence": row.sentiment_confidence,
            "emotion": row.emotion,
            "emotion_confidence": row.emotion_confidence,
            "topic": row.topic,
            "processed_text": row.processed_text,
            "created_at": row.created_at,
        }


def get_feedback_with_analysis(limit: Optional[int] = None) -> list[dict]:
    """Join feedback + feedback_analysis for dashboard/explorer use."""
    with get_session() as session:
        stmt = (
            select(Feedback, FeedbackAnalysis)
            .join(FeedbackAnalysis, FeedbackAnalysis.feedback_id == Feedback.id, isouter=True)
            .order_by(Feedback.created_at.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).all()

        results = []
        for feedback, analysis in rows:
            record = _feedback_to_dict(feedback)
            record.update(
                {
                    "sentiment": analysis.sentiment if analysis else None,
                    "sentiment_confidence": analysis.sentiment_confidence if analysis else None,
                    "emotion": analysis.emotion if analysis else None,
                    "emotion_confidence": analysis.emotion_confidence if analysis else None,
                    "topic": analysis.topic if analysis else None,
                }
            )
            results.append(record)
        return results


def get_feedback_needing_nlp_analysis(limit: Optional[int] = None) -> list[dict]:
    """Feedback rows that don't have emotion/topic analysis yet (e.g.
    imported before Phase 5, or where sentiment-only analysis was run).

    Used by the Settings page's "Run Emotion & Topic Analysis" backfill.
    """
    with get_session() as session:
        stmt = (
            select(Feedback, FeedbackAnalysis)
            .join(FeedbackAnalysis, FeedbackAnalysis.feedback_id == Feedback.id, isouter=True)
            .where(
                (FeedbackAnalysis.id.is_(None))
                | (FeedbackAnalysis.emotion.is_(None))
                | (FeedbackAnalysis.topic.is_(None))
            )
            .order_by(Feedback.created_at.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).all()

        results = []
        for feedback, analysis in rows:
            record = _feedback_to_dict(feedback)
            record["sentiment"] = analysis.sentiment if analysis else None
            results.append(record)
        return results


@_invalidates_cache
def update_emotion_and_topic(
    feedback_id: int,
    emotion: Optional[str] = None,
    emotion_confidence: Optional[float] = None,
    topic: Optional[str] = None,
) -> None:
    """Update just the emotion/topic fields on an existing analysis row,
    creating the row if it doesn't exist yet (leaves sentiment untouched
    if the row already exists).
    """
    with get_session() as session:
        existing = session.scalar(
            select(FeedbackAnalysis).where(FeedbackAnalysis.feedback_id == feedback_id)
        )
        if existing:
            existing.emotion = emotion
            existing.emotion_confidence = emotion_confidence
            existing.topic = topic
        else:
            row = FeedbackAnalysis(
                feedback_id=feedback_id,
                emotion=emotion,
                emotion_confidence=emotion_confidence,
                topic=topic,
            )
            session.add(row)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def insert_report(title: str, summary: str, recommendations: str) -> int:
    with get_session() as session:
        row = Report(title=title, summary=summary, recommendations=recommendations)
        session.add(row)
        session.flush()
        return row.id


def get_all_reports() -> list[dict]:
    with get_session() as session:
        rows = session.scalars(select(Report).order_by(Report.created_at.desc())).all()
        return [
            {
                "id": r.id,
                "title": r.title,
                "summary": r.summary,
                "recommendations": r.recommendations,
                "created_at": r.created_at,
            }
            for r in rows
        ]


def get_report_by_id(report_id: int) -> Optional[dict]:
    with get_session() as session:
        row = session.get(Report, report_id)
        if not row:
            return None
        return {
            "id": row.id,
            "title": row.title,
            "summary": row.summary,
            "recommendations": row.recommendations,
            "created_at": row.created_at,
        }


# ---------------------------------------------------------------------------
# Recurring Complaint Detection (spec section 14)
# ---------------------------------------------------------------------------

def get_top_complaints(limit: int = 5) -> list[dict]:
    """Aggregate negative feedback by topic to surface recurring complaints.

    Because near-duplicate complaints ("Delivery was late." / "My package
    arrived very late.") were already grouped under the same topic label
    by the topic classifier (services/topic_model.py) at analysis time,
    this aggregation is a straightforward group-by -- the semantic grouping
    itself happens earlier, in the model, not here.

    Returns a list of dicts sorted by mention count descending:
        {topic, count, percentage, related_products}
    """
    with get_session() as session:
        stmt = (
            select(Feedback, FeedbackAnalysis)
            .join(FeedbackAnalysis, FeedbackAnalysis.feedback_id == Feedback.id)
            .where(FeedbackAnalysis.sentiment == "Negative")
            .where(FeedbackAnalysis.topic.is_not(None))
        )
        rows = session.execute(stmt).all()
        # Extract plain values while the session is still open -- ORM
        # objects become detached once `with get_session()` exits, and
        # accessing their attributes afterward raises DetachedInstanceError.
        extracted = [(feedback.product, analysis.topic) for feedback, analysis in rows]

    if not extracted:
        return []

    total_negative = len(extracted)
    topic_counts: dict[str, int] = {}
    topic_products: dict[str, dict[str, int]] = {}

    for product, topic in extracted:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        product_counts = topic_products.setdefault(topic, {})
        product_counts[product] = product_counts.get(product, 0) + 1

    complaints = []
    for topic, count in topic_counts.items():
        related_products = sorted(
            topic_products[topic].items(), key=lambda item: item[1], reverse=True
        )
        complaints.append(
            {
                "topic": topic,
                "count": count,
                "percentage": round(count / total_negative * 100, 1),
                "related_products": [p for p, _ in related_products[:3]],
            }
        )

    complaints.sort(key=lambda c: c["count"], reverse=True)
    return complaints[:limit]
