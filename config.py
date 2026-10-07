"""
config.py
---------
Central configuration for LOOP Insights.

Loads all environment variables from `.env` using python-dotenv and exposes
them as typed constants. No other module should call os.getenv() directly for
these values -- import from here instead, so there is a single source of truth.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of the current working directory.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Config:
    """Application-wide configuration."""

    # --- Paths ---
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    SAMPLE_DATASET_PATH = DATA_DIR / "sample_feedback.csv"

    SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
    TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
    TOPIC_MODEL_PATH = MODELS_DIR / "topic_model.pkl"
    TOPIC_VECTORIZER_PATH = MODELS_DIR / "topic_vectorizer.pkl"

    # --- Groq LLM ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    # Default model updated Aug 2026: llama-3.3-70b-versatile was deprecated
    # by Groq in June 2026. openai/gpt-oss-120b is their recommended
    # replacement for similar quality/use-case. Override via .env if Groq's
    # lineup changes again -- check https://console.groq.com/docs/models.
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # --- Admin unlock for the Settings page ---
    # Set in .env locally, or in the hosting platform's secrets after
    # deployment. Leave empty to keep Settings locked for everyone.
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

    # --- MySQL ---
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "project_loop")

    @classmethod
    def sqlalchemy_uri(cls) -> str:
        """Build the SQLAlchemy connection string for MySQL."""
        return (
            f"mysql+mysqlconnector://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}"
            f"@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}"
        )

    # --- App constants ---
    # Change these three lines to rebrand the whole app (sidebar, home page,
    # browser tab, PDF reports).
    APP_TITLE = "LOOP Insights"
    APP_SUBTITLE = "Customer Feedback Intelligence"
    APP_TAGLINE = "Listen. Understand. Improve."

    # Product catalog: category -> products. Single source of truth for the
    # sample data generator and the Manual Feedback form.
    PRODUCT_CATALOG = {
        "Electronics": ["Smartphone", "Laptop", "Tablet", "Camera", "Monitor", "Smart TV"],
        "Accessories": ["Headphones", "Keyboard", "Wireless Mouse", "Power Bank", "Phone Case"],
        "Wearables": ["Smartwatch", "Fitness Band"],
        "Fashion": ["Running Shoes", "Denim Jacket", "Backpack", "Sunglasses"],
        "Home & Kitchen": ["Air Fryer", "Mixer Grinder", "Vacuum Cleaner", "Office Chair"],
        "Beauty & Personal Care": ["Face Serum", "Perfume", "Electric Trimmer"],
        "Health & Fitness": ["Yoga Mat", "Protein Powder"],
    }

    @classmethod
    def product_category_map(cls) -> dict:
        return {p: c for c, products in cls.PRODUCT_CATALOG.items() for p in products}

    SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]
    EMOTION_LABELS = ["Happy", "Satisfied", "Angry", "Frustrated", "Disappointed", "Neutral"]
    TOPIC_LABELS = [
        "Delivery",
        "Product Quality",
        "Price",
        "Customer Support",
        "Refund",
        "Packaging",
        "Payment",
        "Website/App",
        "Product Features",
        "Availability",
        "Other",
    ]

    REQUIRED_COLUMNS = [
        "customer_id",
        "customer_name",
        "product",
        "category",
        "rating",
        "feedback",
        "date",
    ]

    MAX_UPLOAD_SIZE_MB = 25

    def validate(self) -> list:
        """Return a list of human-readable warnings for missing/incomplete config."""
        warnings = []
        if not self.GROQ_API_KEY:
            warnings.append(
                "GROQ_API_KEY is not set. AI Summary, AI Recommendations, and the "
                "LOOP Insight Assistant will be unavailable until it is configured in .env."
            )
        if not self.ADMIN_PASSWORD:
            warnings.append(
                "ADMIN_PASSWORD is not set, so the Settings page cannot be unlocked. "
                "Add it to .env."
            )
        elif len(self.ADMIN_PASSWORD) < 10:
            warnings.append("ADMIN_PASSWORD is short. Use at least 10 characters before deploying.")
        if not self.MYSQL_PASSWORD:
            warnings.append(
                "MYSQL_PASSWORD is empty. Confirm this is intentional for your local MySQL setup."
            )
        return warnings


config = Config()
