"""
services/period.py
------------------
Time-period selection and period-over-period comparison.

"Last 7 days" etc. are counted back from the LATEST feedback date in the
data, not from today's date. Clients usually upload a batch and analyse it
straight away, and this also keeps demos with older sample data working.

Rows with no date are left out of comparisons (they cannot be placed in
either period); the normal date filter still keeps them in the main view.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from services.analytics import compute_kpis

PERIOD_OPTIONS = ["Last 7 days", "Last 30 days", "Last 90 days", "Last 1 year", "All time", "Custom"]
_DAYS = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "Last 1 year": 365}


@dataclass
class Period:
    label: str
    start: Optional[date] = None  # None for "All time"
    end: Optional[date] = None

    @property
    def date_range(self) -> Optional[tuple]:
        return (self.start, self.end) if self.start and self.end else None

    @property
    def text(self) -> str:
        if not self.date_range:
            return "All time"
        return f"{self.start:%d %b %Y} to {self.end:%d %b %Y}"


def data_date_bounds(df: pd.DataFrame) -> Optional[tuple]:
    """(earliest, latest) feedback date in the data, or None if nothing is dated."""
    if df.empty or "feedback_date" not in df or df["feedback_date"].dropna().empty:
        return None
    return df["feedback_date"].min().date(), df["feedback_date"].max().date()


def resolve_period(choice: str, df: pd.DataFrame, custom: Optional[tuple] = None) -> Period:
    bounds = data_date_bounds(df)
    if choice == "Custom" and custom and len(custom) == 2:
        return Period("Custom", custom[0], custom[1])
    if choice in _DAYS and bounds:
        end = bounds[1]
        start = max(end - timedelta(days=_DAYS[choice] - 1), bounds[0])
        return Period(choice, start, end)
    return Period("All time")


def previous_period(period: Period) -> Optional[Period]:
    """The window of the same length that ends the day before `period` starts."""
    if not period.date_range:
        return None
    length = (period.end - period.start).days + 1
    prev_end = period.start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)
    return Period("Previous period", prev_start, prev_end)


def _slice(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    dated = df.dropna(subset=["feedback_date"])
    return dated[
        (dated["feedback_date"] >= pd.Timestamp(period.start))
        & (dated["feedback_date"] < pd.Timestamp(period.end) + pd.Timedelta(days=1))
    ]


def _change(current: float, previous: float) -> dict:
    diff = current - previous
    pct = (diff / previous * 100) if previous else None
    return {"current": current, "previous": previous, "diff": diff, "pct": pct}


def compare_periods(df: pd.DataFrame, period: Period) -> Optional[dict]:
    """Compare KPIs for `period` against the period just before it.

    Returns None for "All time" or when the previous window has no data
    (a change against nothing would be meaningless).
    """
    prev = previous_period(period)
    if prev is None or df.empty:
        return None
    cur_df, prev_df = _slice(df, period), _slice(df, prev)
    if prev_df.empty:
        return None
    cur, old = compute_kpis(cur_df), compute_kpis(prev_df)
    result = {k: _change(cur[k], old[k]) for k in cur}
    result["previous_period"] = prev
    return result
