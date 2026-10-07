"""Tests for time-period selection and period-over-period comparison."""

from datetime import date

import pandas as pd

from services.period import Period, compare_periods, previous_period, resolve_period


def _df(rows):
    """rows: list of (date string or None, rating, sentiment)"""
    return pd.DataFrame(
        {
            "feedback_date": pd.to_datetime([r[0] for r in rows]),
            "rating": [r[1] for r in rows],
            "sentiment": [r[2] for r in rows],
        }
    )


def test_last_7_days_counts_back_from_latest_feedback_date():
    df = _df([("2026-01-01", 5, "Positive"), ("2026-03-31", 4, "Positive")])
    p = resolve_period("Last 7 days", df)
    assert p.end == date(2026, 3, 31)
    assert p.start == date(2026, 3, 25)


def test_period_never_starts_before_the_data_does():
    df = _df([("2026-03-30", 5, "Positive"), ("2026-03-31", 4, "Positive")])
    assert resolve_period("Last 1 year", df).start == date(2026, 3, 30)


def test_all_time_has_no_date_range():
    df = _df([("2026-03-31", 5, "Positive")])
    assert resolve_period("All time", df).date_range is None


def test_custom_range_is_used_as_given():
    df = _df([("2026-03-31", 5, "Positive")])
    p = resolve_period("Custom", df, (date(2026, 3, 1), date(2026, 3, 10)))
    assert p.date_range == (date(2026, 3, 1), date(2026, 3, 10))


def test_data_without_any_dates_falls_back_to_all_time():
    df = _df([(None, 5, "Positive")])
    assert resolve_period("Last 7 days", df).date_range is None


def test_previous_period_is_same_length_and_directly_before():
    prev = previous_period(Period("x", date(2026, 3, 8), date(2026, 3, 14)))  # 7 days
    assert prev.start == date(2026, 3, 1)
    assert prev.end == date(2026, 3, 7)


def test_all_time_has_no_previous_period():
    assert previous_period(Period("All time")) is None


def test_compare_shows_rating_drop_and_complaint_increase():
    df = _df(
        [
            ("2026-03-02", 5, "Positive"), ("2026-03-03", 5, "Positive"),  # previous week
            ("2026-03-04", 4, "Positive"), ("2026-03-05", 4, "Positive"),
            ("2026-03-09", 2, "Negative"), ("2026-03-10", 2, "Negative"),  # current week
            ("2026-03-11", 2, "Negative"), ("2026-03-12", 4, "Positive"),
        ]
    )
    period = Period("x", date(2026, 3, 8), date(2026, 3, 14))
    result = compare_periods(df, period)
    assert result["average_rating"]["previous"] == 4.5
    assert result["average_rating"]["current"] == 2.5
    assert result["average_rating"]["diff"] == -2.0
    assert result["negative_feedback"]["previous"] == 0
    assert result["negative_feedback"]["current"] == 3
    assert result["negative_feedback"]["pct"] is None  # change from zero has no percentage


def test_compare_percent_change_in_volume():
    df = _df([("2026-03-02", 4, "Positive")] * 2 + [("2026-03-10", 4, "Positive")] * 3)
    result = compare_periods(df, Period("x", date(2026, 3, 8), date(2026, 3, 14)))
    assert result["total_feedback"]["pct"] == 50.0


def test_compare_returns_none_when_previous_period_is_empty():
    df = _df([("2026-03-10", 4, "Positive")])
    assert compare_periods(df, Period("x", date(2026, 3, 8), date(2026, 3, 14))) is None


def test_compare_ignores_undated_rows():
    df = _df([("2026-03-02", 4, "Positive"), (None, 1, "Negative"), ("2026-03-10", 4, "Positive")])
    result = compare_periods(df, Period("x", date(2026, 3, 8), date(2026, 3, 14)))
    assert result["total_feedback"]["current"] == 1
    assert result["total_feedback"]["previous"] == 1


def test_last_day_of_period_includes_the_whole_day():
    df = _df([("2026-03-02 09:00", 4, "Positive"), ("2026-03-14 18:30", 4, "Positive")])
    result = compare_periods(df, Period("x", date(2026, 3, 8), date(2026, 3, 14)))
    assert result["total_feedback"]["current"] == 1
