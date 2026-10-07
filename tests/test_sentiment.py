"""
tests/test_sentiment.py
-------------------------
Tests for services/sentiment.py (spec section 30: Sentiment -- model
loading, prediction, confidence).

These tests require a trained model to exist (models/sentiment_model.pkl
and models/tfidf_vectorizer.pkl). Run `python -m training.train_sentiment`
first if they're missing -- the tests skip gracefully rather than fail if
no model is present, since a fresh checkout won't have one until trained.
"""

import pytest

from config import config
from services import sentiment

MODEL_MISSING = not (config.SENTIMENT_MODEL_PATH.exists() and config.TFIDF_VECTORIZER_PATH.exists())

pytestmark = pytest.mark.skipif(
    MODEL_MISSING, reason="No trained model found -- run `python -m training.train_sentiment` first."
)


@pytest.fixture(autouse=True)
def _reload_model_between_tests():
    sentiment.reload_model()
    yield


def test_is_model_available_true_when_trained():
    assert sentiment.is_model_available() is True


def test_predict_sentiment_returns_label_and_confidence():
    label, confidence = sentiment.predict_sentiment("This product is absolutely wonderful, I love it!")
    assert label in ("Positive", "Neutral", "Negative")
    assert 0.0 <= confidence <= 1.0


def test_predict_sentiment_positive_example():
    label, confidence = sentiment.predict_sentiment(
        "Excellent quality and fast delivery, highly recommend this product."
    )
    assert label == "Positive"
    assert confidence > 0.3  # sanity check, not a hard-coded exact number


def test_predict_sentiment_negative_example():
    label, confidence = sentiment.predict_sentiment(
        "Terrible experience, the product broke immediately and support never responded."
    )
    assert label == "Negative"
    assert confidence > 0.3


def test_predict_sentiment_handles_short_everyday_negative_words():
    # Regression test: an earlier version of the training dataset never
    # used plain, everyday words like "bad" in its sentence templates, so
    # the model had literally zero training signal for them and would
    # default to Positive on inputs like "very bad" (a real bug caught via
    # user testing). The dataset now includes short, plain-language
    # templates specifically to cover this.
    for text in ["very bad", "bad", "this is bad", "awful", "worst product ever"]:
        label, _ = sentiment.predict_sentiment(text)
        assert label == "Negative", f"expected Negative for {text!r}, got {label}"


def test_predict_sentiment_handles_empty_text():
    label, confidence = sentiment.predict_sentiment("")
    assert label == "Neutral"
    assert confidence == 0.0


def test_predict_sentiment_batch_matches_single_predictions():
    texts = [
        "Absolutely great product, very satisfied.",
        "Awful, broke after one day, waste of money.",
    ]
    batch_results = sentiment.predict_sentiment_batch(texts)
    assert len(batch_results) == 2
    for text, (batch_label, batch_conf) in zip(texts, batch_results):
        single_label, single_conf = sentiment.predict_sentiment(text)
        assert batch_label == single_label
        assert abs(batch_conf - single_conf) < 1e-6


def test_predict_sentiment_batch_handles_mixed_empty_and_valid():
    texts = ["", "Great service and fast shipping!", ""]
    results = sentiment.predict_sentiment_batch(texts)
    assert results[0] == ("Neutral", 0.0)
    assert results[2] == ("Neutral", 0.0)
    assert results[1][0] in ("Positive", "Neutral", "Negative")
