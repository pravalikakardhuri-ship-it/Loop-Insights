"""
ui/period_picker.py
-------------------
One shared time-period control for Dashboard, Product Analysis, AI Insights
and Reports. The choice is remembered while you move between pages.
"""

import pandas as pd
import streamlit as st

from services.period import PERIOD_OPTIONS, Period, data_date_bounds, resolve_period


def render_period_picker(df: pd.DataFrame) -> Period:
    """Show the period buttons and return the chosen Period.

    "Last N days" counts back from the newest feedback in the data.
    """
    bounds = data_date_bounds(df)
    if bounds is None:
        return Period("All time")  # nothing is dated, so there is nothing to narrow

    saved = st.session_state.get("period_choice", "All time")
    choice = st.segmented_control(
        "Time period", PERIOD_OPTIONS, default=saved, key="period_widget", label_visibility="collapsed"
    ) or "All time"
    st.session_state["period_choice"] = choice

    custom = None
    if choice == "Custom":
        picked = st.date_input(
            "Dates", value=(bounds[0], bounds[1]), min_value=bounds[0], max_value=bounds[1], key="period_custom"
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            custom = picked
        else:
            st.caption("Pick an end date to apply the range.")

    period = resolve_period(choice, df, custom)
    if period.date_range:
        st.caption(f"Showing {period.text}. Data available: {bounds[0]:%d %b %Y} to {bounds[1]:%d %b %Y}.")
    return period
