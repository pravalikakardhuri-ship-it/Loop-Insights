"""
database/__init__.py
---------------------
SQLAlchemy engine and session management for Project LOOP.

This module only sets up the connection layer. Table definitions live in
`database/models.py` and query helpers live in `database/queries.py`.

Note: this lives in `database/__init__.py` rather than a top-level
`database.py` so that `import database` is unambiguous -- Python does not
allow a package directory and a module file to share the same name in the
same parent directory without one silently shadowing the other.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import config

# `pool_pre_ping` avoids stale-connection errors after periods of inactivity.
engine = create_engine(config.sqlalchemy_uri(), pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session():
    """Context-managed database session.

    Usage:
        with get_session() as session:
            session.execute(...)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_connection() -> tuple[bool, str]:
    """Quick connectivity check used by the Settings page and startup checks.

    Returns (success, message) instead of raising, so the Streamlit UI can
    show a friendly message rather than a stack trace.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected to MySQL successfully."
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a message
        return False, f"Could not connect to MySQL: {exc}"
