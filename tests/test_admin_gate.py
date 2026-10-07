"""
Tests that Settings stays hidden from clients and opens only with the admin
password. They run the real app (app.py) with Streamlit's AppTest helper.
"""

import pathlib

import pytest
from streamlit.testing.v1 import AppTest

from config import config

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
PASSWORD = "admin-password-123"


@pytest.fixture(autouse=True)
def admin_password(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_PASSWORD", PASSWORD)


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _nav(at: AppTest) -> list:
    return list(at.sidebar.radio[0].options)


def _unlock(at: AppTest, password: str) -> AppTest:
    at.sidebar.text_input[0].input(password)
    at.sidebar.button[0].click().run()
    return at


def test_clients_do_not_see_settings():
    at = _app()
    assert not at.exception
    assert "⚙️ Settings" not in _nav(at)
    assert "📊 Dashboard" in _nav(at) and "📤 Upload Feedback" in _nav(at)


def test_right_password_shows_settings():
    at = _unlock(_app(), PASSWORD)
    assert not at.exception
    assert "⚙️ Settings" in _nav(at)


def test_wrong_password_keeps_settings_hidden():
    at = _unlock(_app(), "wrong-guess")
    assert any("Wrong password" in e.value for e in at.sidebar.error)
    assert "⚙️ Settings" not in _nav(at)


def test_five_wrong_attempts_lock_the_box_even_for_the_right_password():
    at = _app()
    for _ in range(5):
        _unlock(at, "wrong-guess")
    _unlock(at, PASSWORD)
    assert any("Too many wrong attempts" in e.value for e in at.sidebar.error)
    assert "⚙️ Settings" not in _nav(at)


def test_lock_admin_hides_settings_again():
    at = _unlock(_app(), PASSWORD)
    next(b for b in at.sidebar.button if b.label == "Lock admin").click().run()
    assert "⚙️ Settings" not in _nav(at)


def test_locking_while_on_settings_falls_back_to_a_normal_page():
    at = _unlock(_app(), PASSWORD)
    at.session_state["nav"] = "⚙️ Settings"
    at.run()
    next(b for b in at.sidebar.button if b.label == "Lock admin").click().run()
    assert at.sidebar.radio[0].value == "🏠 Home"
    assert not at.exception


def test_faking_the_page_name_does_not_open_settings():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = "⚙️ Settings"
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "🏠 Home"
    assert not any(h.value == "Settings" for h in at.get("header"))


def test_settings_page_itself_refuses_when_not_admin():
    root = str(pathlib.Path(APP).parent)
    script = f"import sys\nsys.path.insert(0, {root!r})\nfrom ui import settings\nsettings.render()\n"
    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    assert any("only available to the admin" in e.value for e in at.error)


def test_no_admin_password_set_means_nobody_can_unlock(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "")
    at = _app()
    assert any("turned off" in c.value for c in at.sidebar.caption)
    assert not at.sidebar.text_input          # no password box to try
    assert "⚙️ Settings" not in _nav(at)


def test_setup_notices_are_hidden_from_clients_and_shown_to_admin():
    client = _app()
    assert not any("Setup notices" in e.label for e in client.expander)
    admin = _unlock(_app(), PASSWORD)
    assert any("Setup notices" in e.label for e in admin.expander)


# ---------------------------------------------------------------------------
# What clients see when Groq or the database has a problem
# ---------------------------------------------------------------------------

def _page(page: str, unlocked: bool) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = page
    if unlocked:
        at.session_state["is_admin"] = True
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    parts = [w.value for w in list(at.warning) + list(at.error) + list(at.info) + list(at.caption)]
    return " ".join(str(p) for p in parts)


@pytest.mark.parametrize("page", ["🤖 AI Insights", "💬 LOOP Assistant"])
def test_missing_groq_key_shows_clients_a_neutral_message(monkeypatch, page):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    text = _all_text(_page(page, unlocked=False))
    assert "unavailable right now" in text
    assert "GROQ_API_KEY" not in text and ".env" not in text and "Settings" not in text


@pytest.mark.parametrize("page", ["🤖 AI Insights", "💬 LOOP Assistant"])
def test_missing_groq_key_tells_the_admin_what_to_do(monkeypatch, page):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    text = _all_text(_page(page, unlocked=True))
    assert "GROQ_API_KEY is not set" in text


def test_database_error_hides_technical_details_from_clients(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret-host-name refused the connection")

    monkeypatch.setattr("services.analytics.load_feedback_df", broken)
    client = _all_text(_page("📊 Dashboard", unlocked=False))
    assert "couldn't load the data" in client
    assert "secret-host-name" not in client and "MySQL" not in client and "Settings" not in client

    admin_page = _page("📊 Dashboard", unlocked=True)
    admin_text = " ".join(str(c.value) for c in admin_page.code)
    assert "secret-host-name" in admin_text
