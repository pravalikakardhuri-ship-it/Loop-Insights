"""
tests/test_topic.py
----------------------
Tests for services/topic_model.py and training/train_topic.py (weak
labeling + the trained classifier).
"""

import pytest

from config import config
from training.train_topic import weak_label_topic, load_training_data
from services import topic_model

MODEL_MISSING = not (config.TOPIC_MODEL_PATH.exists() and config.TOPIC_VECTORIZER_PATH.exists())


def test_weak_label_delivery():
    assert weak_label_topic("My order arrived three days late, very disappointing.") == "Delivery"


def test_weak_label_refund():
    assert weak_label_topic("Requested a refund but it's been weeks with no update.") == "Refund"


def test_weak_label_customer_support():
    assert weak_label_topic("Customer support never responded to my complaint.") == "Customer Support"


def test_weak_label_payment():
    assert weak_label_topic("Payment was deducted twice for the same order.") == "Payment"


def test_weak_label_falls_back_to_other():
    assert weak_label_topic("Average experience overall with the product.") == "Other"


def test_load_training_data_produces_topic_column():
    df = load_training_data()
    assert "topic" in df.columns
    assert len(df) > 0
    assert (df["topic"].isna()).sum() == 0


@pytest.mark.skipif(MODEL_MISSING, reason="No trained topic model -- run `python -m training.train_topic` first.")
class TestTopicModel:
    def setup_method(self):
        topic_model.reload_model()

    def test_is_model_available(self):
        assert topic_model.is_model_available() is True

    def test_predict_topic_returns_label_and_confidence(self):
        label, confidence = topic_model.predict_topic("Customer support never responded to my complaint.")
        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0

    def test_predict_topic_delivery_example(self):
        label, _ = topic_model.predict_topic("My order arrived three days late, very disappointing.")
        assert label == "Delivery"

    def test_predict_topic_handles_empty_text(self):
        label, confidence = topic_model.predict_topic("")
        assert label == "Other"
        assert confidence == 0.0

    def test_predict_topic_batch_matches_single(self):
        texts = [
            "Requested a refund but it's been weeks with no update.",
            "The app kept crashing when I tried to track my order.",
        ]
        batch_results = topic_model.predict_topic_batch(texts)
        for text, (batch_label, batch_conf) in zip(texts, batch_results):
            single_label, single_conf = topic_model.predict_topic(text)
            assert batch_label == single_label
            assert abs(batch_conf - single_conf) < 1e-6
