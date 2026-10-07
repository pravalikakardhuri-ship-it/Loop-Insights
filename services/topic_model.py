"""
services/topic_model.py
--------------------------
Topic/issue prediction service. Mirrors services/sentiment.py: loads the
trained TF-IDF vectorizer + Logistic Regression model once (see
training/train_topic.py) and exposes simple predict functions.

IMPORTANT caveat (see training/train_topic.py docstring for the full
explanation): the training labels come from keyword matching, and TF-IDF
is also keyword-based, so this model's very high accuracy mainly confirms
it learned to reproduce that keyword lexicon from raw text -- not that it
generalizes to topics described in genuinely novel vocabulary. It's still
more useful than raw keyword matching at inference time (it picks up on
word combinations and partial/related phrasing), but don't oversell the
accuracy number on its own.
"""

from functools import lru_cache

import joblib

from config import config


class TopicModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested but no trained topic model exists yet."""


@lru_cache(maxsize=1)
def _load_artifacts():
    if not config.TOPIC_MODEL_PATH.exists() or not config.TOPIC_VECTORIZER_PATH.exists():
        raise TopicModelNotTrainedError(
            "No trained topic model found. Run `python -m training.train_topic` "
            "(or use the 'Train Model' button in Settings) before analyzing feedback."
        )
    model = joblib.load(config.TOPIC_MODEL_PATH)
    vectorizer = joblib.load(config.TOPIC_VECTORIZER_PATH)
    return model, vectorizer


def is_model_available() -> bool:
    return config.TOPIC_MODEL_PATH.exists() and config.TOPIC_VECTORIZER_PATH.exists()


def predict_topic(text: str) -> tuple[str, float]:
    """Predict the main topic/issue for a single piece of feedback text.

    Returns (label, confidence). Raises TopicModelNotTrainedError if no
    model has been trained yet, or if the saved model is incompatible with
    the installed scikit-learn version.
    """
    from services.preprocessing import clean_text

    model, vectorizer = _load_artifacts()
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "Other", 0.0

    try:
        vec = vectorizer.transform([cleaned])
        label = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
    except AttributeError as exc:
        raise TopicModelNotTrainedError(
            "The saved topic model appears to be incompatible with the installed "
            "scikit-learn version. Go to Settings and click 'Train / Retrain Topic "
            "Model' to regenerate it locally."
        ) from exc

    class_index = list(model.classes_).index(label)
    confidence = float(proba[class_index])
    return label, confidence


def predict_topic_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Predict topic for many texts at once (used after bulk upload)."""
    from services.preprocessing import clean_text

    model, vectorizer = _load_artifacts()
    cleaned_texts = [clean_text(t) for t in texts]

    non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]
    results: list[tuple[str, float]] = [("Other", 0.0)] * len(texts)

    if not non_empty_indices:
        return results

    try:
        vecs = vectorizer.transform([cleaned_texts[i] for i in non_empty_indices])
        labels = model.predict(vecs)
        probas = model.predict_proba(vecs)
    except AttributeError as exc:
        raise TopicModelNotTrainedError(
            "The saved topic model appears to be incompatible with the installed "
            "scikit-learn version. Go to Settings and click 'Train / Retrain Topic "
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
