"""
services/sentiment.py
----------------------
Sentiment prediction service. Loads the trained TF-IDF vectorizer and
Logistic Regression model (see training/train_sentiment.py) once and
reuses them for all predictions.

The architecture here mirrors what emotion.py and topic_model.py will do
in Phase 5 -- load artifacts once, expose a simple predict(text) function
-- so the rest of the app (Upload, Manual Feedback, Dashboard) can call
all three the same way.
"""

from functools import lru_cache

import joblib

from config import config


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested but no trained model exists yet."""


@lru_cache(maxsize=1)
def _load_artifacts():
    if not config.SENTIMENT_MODEL_PATH.exists() or not config.TFIDF_VECTORIZER_PATH.exists():
        raise ModelNotTrainedError(
            "No trained sentiment model found. Run `python -m training.train_sentiment` "
            "(or use the 'Train Model' button in Settings) before analyzing feedback."
        )
    model = joblib.load(config.SENTIMENT_MODEL_PATH)
    vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)
    return model, vectorizer


def is_model_available() -> bool:
    return config.SENTIMENT_MODEL_PATH.exists() and config.TFIDF_VECTORIZER_PATH.exists()


def predict_sentiment(text: str) -> tuple[str, float]:
    """Predict sentiment for a single piece of feedback text.

    Returns (label, confidence) where confidence is the model's predicted
    probability for the chosen label (0.0-1.0). Raises ModelNotTrainedError
    if no model has been trained yet, or if the saved model is incompatible
    with the installed scikit-learn version -- callers should catch this
    and show a friendly message rather than crash.
    """
    from services.preprocessing import clean_text

    model, vectorizer = _load_artifacts()
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "Neutral", 0.0

    try:
        vec = vectorizer.transform([cleaned])
        label = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
    except AttributeError as exc:
        raise ModelNotTrainedError(
            "The saved sentiment model appears to be incompatible with the installed "
            "scikit-learn version (this happens if the .pkl file was created with a "
            "different scikit-learn version than the one now installed). Go to Settings "
            "and click 'Train / Retrain Sentiment Model' to regenerate it locally."
        ) from exc

    class_index = list(model.classes_).index(label)
    confidence = float(proba[class_index])
    return label, confidence


def predict_sentiment_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Predict sentiment for many texts at once (used after bulk upload)."""
    from services.preprocessing import clean_text

    model, vectorizer = _load_artifacts()
    cleaned_texts = [clean_text(t) for t in texts]

    non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]
    results: list[tuple[str, float]] = [("Neutral", 0.0)] * len(texts)

    if not non_empty_indices:
        return results

    try:
        vecs = vectorizer.transform([cleaned_texts[i] for i in non_empty_indices])
        labels = model.predict(vecs)
        probas = model.predict_proba(vecs)
    except AttributeError as exc:
        raise ModelNotTrainedError(
            "The saved sentiment model appears to be incompatible with the installed "
            "scikit-learn version. Go to Settings and click 'Train / Retrain Sentiment "
            "Model' to regenerate it locally."
        ) from exc

    for pos, idx in enumerate(non_empty_indices):
        label = labels[pos]
        class_index = list(model.classes_).index(label)
        confidence = float(probas[pos][class_index])
        results[idx] = (label, confidence)

    return results


def reload_model() -> None:
    """Clear the cached model/vectorizer so the next prediction reloads
    them from disk (call this after retraining)."""
    _load_artifacts.cache_clear()
