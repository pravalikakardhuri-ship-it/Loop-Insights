"""
services/analytics.py
------------------------
Core analytics computations shared by the Dashboard, Feedback Explorer,
and Product Analysis pages.

Design: fetch all feedback+analysis rows once into a pandas DataFrame
(database.queries.get_feedback_with_analysis already joins the two
tables), then do filtering and aggregation in pandas. For a dataset this
size (thousands, not millions, of rows) this is simpler and just as fast
as building dynamic SQL for every filter combination, and keeps all the
aggregation logic in one place that's easy to unit test.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# Small in-memory cache so every click/filter change doesn't re-download all
# rows from the (possibly remote) database. It is cleared automatically
# whenever data is written (see database/queries.py) and expires after
# _CACHE_TTL_SECONDS as a safety net.
_CACHE_TTL_SECONDS = 300
_cache: dict = {"key": None, "time": 0.0, "df": None}


def clear_feedback_cache() -> None:
    """Forget the cached DataFrame so the next load re-reads the database."""
    _cache["key"] = None
    _cache["df"] = None


def load_feedback_df() -> pd.DataFrame:
    """Load all feedback + analysis rows as a DataFrame ready for filtering
    and aggregation. Results are cached briefly (see above)."""
    import time

    import database
    from database.queries import get_feedback_with_analysis

    key = id(database.engine)
    fresh = time.time() - _cache["time"] < _CACHE_TTL_SECONDS
    if _cache["df"] is not None and _cache["key"] == key and fresh:
        return _cache["df"].copy()

    records = get_feedback_with_analysis()
    df = pd.DataFrame(records)
    if not df.empty:
        df["feedback_date"] = pd.to_datetime(df["feedback_date"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    _cache.update({"key": key, "time": time.time(), "df": df})
    return df.copy()


@dataclass
class Filters:
    date_range: Optional[tuple] = None  # (start_date, end_date)
    products: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    sentiments: list = field(default_factory=list)
    ratings: list = field(default_factory=list)
    topics: list = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            [self.date_range, self.products, self.categories, self.sentiments, self.ratings, self.topics]
        )


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    if df.empty or filters is None:
        return df

    out = df.copy()

    if filters.date_range and filters.date_range[0] and filters.date_range[1]:
        start, end = filters.date_range
        # Rows with a missing/unparseable date (feedback_date is NaT) must
        # NOT be silently dropped by a date filter -- a comparison against
        # NaT is always False in pandas, so without this explicit
        # "keep missing dates" clause, any date filter (even the dashboard's
        # default full-range one, which is active from the moment the page
        # loads) would invisibly exclude those rows from every chart and
        # KPI. This bit real: manually-entered feedback never gets a date
        # set, so every manual entry was disappearing from the Dashboard.
        out = out[
            out["feedback_date"].isna()
            | ((out["feedback_date"] >= pd.Timestamp(start)) & (out["feedback_date"] <= pd.Timestamp(end)))
        ]
    if filters.products:
        out = out[out["product"].isin(filters.products)]
    if filters.categories:
        out = out[out["category"].isin(filters.categories)]
    if filters.sentiments:
        out = out[out["sentiment"].isin(filters.sentiments)]
    if filters.ratings:
        out = out[out["rating"].isin(filters.ratings)]
    if filters.topics:
        out = out[out["topic"].isin(filters.topics)]

    return out


# ---------------------------------------------------------------------------
# KPIs & LOOP Customer Satisfaction Score (spec section 16)
# ---------------------------------------------------------------------------

def compute_loop_satisfaction_score(df: pd.DataFrame) -> dict:
    """LOOP Customer Satisfaction Score -- a custom metric defined by this
    project, NOT an industry-standard metric (the app says so wherever
    it's shown, per the spec).

    Rating Score = Average Rating / 5 * 100
    Sentiment Score = Positive% + (Neutral% * 0.5)
    LOOP Score = (Rating Score * 0.5) + (Sentiment Score * 0.5)
    """
    if df.empty or df["rating"].dropna().empty:
        return {"rating_score": 0.0, "sentiment_score": 0.0, "loop_score": 0.0}

    avg_rating = df["rating"].mean()
    rating_score = (avg_rating / 5) * 100

    sentiments = df["sentiment"].dropna()
    total = len(sentiments)
    if total == 0:
        sentiment_score = 0.0
    else:
        positive_pct = (sentiments == "Positive").sum() / total * 100
        neutral_pct = (sentiments == "Neutral").sum() / total * 100
        sentiment_score = positive_pct + (neutral_pct * 0.5)

    loop_score = (rating_score * 0.5) + (sentiment_score * 0.5)

    return {
        "rating_score": round(rating_score, 1),
        "sentiment_score": round(sentiment_score, 1),
        "loop_score": round(loop_score, 1),
    }


def compute_kpis(df: pd.DataFrame) -> dict:
    """Core KPI numbers for the Dashboard's top metric row."""
    total = len(df)
    if total == 0:
        return {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "neutral_feedback": 0,
            "average_rating": 0.0,
            "loop_satisfaction_score": 0.0,
        }

    sentiments = df["sentiment"].dropna()
    positive = int((sentiments == "Positive").sum())
    negative = int((sentiments == "Negative").sum())
    neutral = int((sentiments == "Neutral").sum())
    avg_rating = round(float(df["rating"].mean()), 2) if df["rating"].notna().any() else 0.0

    score = compute_loop_satisfaction_score(df)

    return {
        "total_feedback": total,
        "positive_feedback": positive,
        "negative_feedback": negative,
        "neutral_feedback": neutral,
        "average_rating": avg_rating,
        "loop_satisfaction_score": score["loop_score"],
    }


# ---------------------------------------------------------------------------
# Chart-ready aggregations (spec section 15)
# ---------------------------------------------------------------------------

def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Chart 1: Sentiment distribution."""
    if df.empty:
        return pd.DataFrame(columns=["sentiment", "count"])
    return df["sentiment"].dropna().value_counts().rename_axis("sentiment").reset_index(name="count")


def sentiment_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Chart 2: Sentiment over time (daily counts per sentiment)."""
    if df.empty or df["feedback_date"].dropna().empty:
        return pd.DataFrame(columns=["date", "sentiment", "count"])
    working = df.dropna(subset=["feedback_date", "sentiment"]).copy()
    working["date"] = working["feedback_date"].dt.date
    grouped = working.groupby(["date", "sentiment"]).size().reset_index(name="count")
    return grouped


def feedback_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Chart 3: Feedback by product."""
    if df.empty:
        return pd.DataFrame(columns=["product", "count"])
    return df["product"].value_counts().rename_axis("product").reset_index(name="count")


def feedback_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Chart 4: Feedback by category."""
    if df.empty:
        return pd.DataFrame(columns=["category", "count"])
    return df["category"].value_counts().rename_axis("category").reset_index(name="count")


def top_complaints_chart_data(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Chart 5: Top customer complaints (negative feedback grouped by topic)."""
    if df.empty:
        return pd.DataFrame(columns=["topic", "count"])
    negative = df[(df["sentiment"] == "Negative") & (df["topic"].notna())]
    if negative.empty:
        return pd.DataFrame(columns=["topic", "count"])
    counts = negative["topic"].value_counts().head(top_n).rename_axis("topic").reset_index(name="count")
    return counts


def emotion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Chart 6: Emotion distribution."""
    if df.empty:
        return pd.DataFrame(columns=["emotion", "count"])
    return df["emotion"].dropna().value_counts().rename_axis("emotion").reset_index(name="count")


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Chart 7: Rating distribution (1-5 stars)."""
    if df.empty:
        return pd.DataFrame({"rating": [1, 2, 3, 4, 5], "count": [0, 0, 0, 0, 0]})
    counts = df["rating"].dropna().astype(int).value_counts().reindex(range(1, 6), fill_value=0)
    return counts.rename_axis("rating").reset_index(name="count")


def product_wise_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Chart 8: Product-wise sentiment (stacked)."""
    if df.empty:
        return pd.DataFrame(columns=["product", "sentiment", "count"])
    working = df.dropna(subset=["sentiment"])
    if working.empty:
        return pd.DataFrame(columns=["product", "sentiment", "count"])
    grouped = working.groupby(["product", "sentiment"]).size().reset_index(name="count")
    return grouped


def product_rating_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product rating summary: total feedback, lowest rating received,
    highest rating received, and average rating -- sorted by average
    rating descending so the best/worst-performing products are obvious
    at a glance.
    """
    if df.empty or df["rating"].dropna().empty:
        return pd.DataFrame(columns=["product", "total_feedback", "min_rating", "max_rating", "avg_rating"])

    working = df.dropna(subset=["rating"])
    grouped = working.groupby("product")["rating"].agg(
        total_feedback="count", min_rating="min", max_rating="max", avg_rating="mean"
    ).reset_index()
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    grouped[["min_rating", "max_rating"]] = grouped[["min_rating", "max_rating"]].astype(int)
    return grouped.sort_values("avg_rating", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Product Analysis (spec section 24)
# ---------------------------------------------------------------------------

def product_summary(df: pd.DataFrame, product: str) -> dict:
    """Stats for a single product's Product Analysis page."""
    product_df = df[df["product"] == product]
    total = len(product_df)
    if total == 0:
        return {
            "product": product, "total_feedback": 0, "average_rating": 0.0,
            "positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 0.0,
            "loop_satisfaction_score": 0.0, "top_complaint": None, "top_emotion": None,
        }

    sentiments = product_df["sentiment"].dropna()
    sentiment_total = len(sentiments) if len(sentiments) > 0 else 1
    positive_pct = round((sentiments == "Positive").sum() / sentiment_total * 100, 1)
    negative_pct = round((sentiments == "Negative").sum() / sentiment_total * 100, 1)
    neutral_pct = round((sentiments == "Neutral").sum() / sentiment_total * 100, 1)

    score = compute_loop_satisfaction_score(product_df)

    negative_topics = product_df[
        (product_df["sentiment"] == "Negative") & (product_df["topic"].notna())
    ]["topic"]
    top_complaint = negative_topics.value_counts().idxmax() if not negative_topics.empty else None

    emotions = product_df["emotion"].dropna()
    top_emotion = emotions.value_counts().idxmax() if not emotions.empty else None

    return {
        "product": product,
        "total_feedback": total,
        "average_rating": round(float(product_df["rating"].mean()), 2) if product_df["rating"].notna().any() else 0.0,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "neutral_pct": neutral_pct,
        "loop_satisfaction_score": score["loop_score"],
        "top_complaint": top_complaint,
        "top_emotion": top_emotion,
    }
