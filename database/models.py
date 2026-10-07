"""
database/models.py
-------------------
SQLAlchemy ORM models for Project LOOP.

Three tables, matching the spec exactly:
    - feedback           raw (cleaned) customer feedback rows
    - feedback_analysis  ML/NLP output per feedback row (1:1 with feedback)
    - reports            saved AI-generated reports

Run `python -m database.models` (or call `init_db()` from anywhere) to
create these tables in the configured MySQL database.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Date,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

import database

Base = declarative_base()


class Feedback(Base):
    """One row of raw, cleaned customer feedback."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), nullable=False, index=True)
    customer_name = Column(String(150), nullable=True)
    product = Column(String(150), nullable=False, index=True)
    category = Column(String(100), nullable=True, index=True)
    rating = Column(Integer, nullable=False)
    feedback_text = Column(Text, nullable=False)
    feedback_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # One feedback row has at most one analysis row.
    analysis = relationship(
        "FeedbackAnalysis",
        back_populates="feedback",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_feedback_product_category", "product", "category"),
    )

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} product={self.product!r} rating={self.rating}>"


class FeedbackAnalysis(Base):
    """ML/NLP analysis results for a single feedback row."""

    __tablename__ = "feedback_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(
        Integer, ForeignKey("feedback.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )

    sentiment = Column(String(20), nullable=True, index=True)  # Positive / Neutral / Negative
    sentiment_confidence = Column(Float, nullable=True)

    emotion = Column(String(30), nullable=True, index=True)
    emotion_confidence = Column(Float, nullable=True)

    topic = Column(String(50), nullable=True, index=True)
    processed_text = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    feedback = relationship("Feedback", back_populates="analysis")

    def __repr__(self) -> str:
        return f"<FeedbackAnalysis feedback_id={self.feedback_id} sentiment={self.sentiment!r}>"


class Report(Base):
    """A saved AI-generated report (summary + recommendations)."""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    summary = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self) -> str:
        return f"<Report id={self.id} title={self.title!r}>"


def init_db() -> None:
    """Create all tables that don't already exist. Safe to call repeatedly.

    Resolves `database.engine` at call time (rather than importing it by
    value at module load) so tests can point it at a different engine
    (e.g. in-memory SQLite) before calling this.
    """
    Base.metadata.create_all(bind=database.engine)


if __name__ == "__main__":
    init_db()
    print("Tables created (or already existed): feedback, feedback_analysis, reports")
