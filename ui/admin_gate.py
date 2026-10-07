"""
ui/admin_gate.py
----------------
Keeps Settings (connections, model training, data tools) away from clients.

Everyone sees the normal pages. The admin opens the "Admin" box at the bottom
of the sidebar, types the password, and the Settings page appears. The
password comes from ADMIN_PASSWORD in .env (or the hosting platform's secrets),
never from the code.
"""

import hmac
import time

import streamlit as st

from config import config

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def is_admin() -> bool:
    return bool(st.session_state.get("is_admin"))


def lock_admin() -> None:
    st.session_state.pop("is_admin", None)


def _password_ok(password: str) -> bool:
    return bool(config.ADMIN_PASSWORD) and hmac.compare_digest(password.encode(), config.ADMIN_PASSWORD.encode())


def _locked_for() -> int:
    return max(0, int(st.session_state.get("admin_locked_until", 0) - time.time()))


def render_admin_box() -> None:
    """The small unlock box at the bottom of the sidebar."""
    if is_admin():
        st.sidebar.success("Admin mode is on.")
        if st.sidebar.button("Lock admin"):
            lock_admin()
            st.rerun()
        return

    with st.sidebar.expander("🔒 Admin"):
        if not config.ADMIN_PASSWORD:
            st.caption("Admin sign-in is turned off. Set ADMIN_PASSWORD in .env to use Settings.")
            return
        with st.form("admin_form", border=False):
            password = st.text_input("Admin password", type="password")
            submitted = st.form_submit_button("Unlock")
        if not submitted:
            return
        wait = _locked_for()
        if wait:
            st.error(f"Too many wrong attempts. Try again in {wait} seconds.")
        elif _password_ok(password):
            st.session_state["is_admin"] = True
            st.session_state["admin_fails"] = 0
            st.rerun()
        else:
            fails = st.session_state.get("admin_fails", 0) + 1
            st.session_state["admin_fails"] = fails
            if fails >= MAX_FAILED_ATTEMPTS:
                st.session_state["admin_locked_until"] = time.time() + LOCKOUT_SECONDS
                st.session_state["admin_fails"] = 0
            st.error("Wrong password.")
