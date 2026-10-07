"""
tests/test_emotion.py
------------------------
Tests for services/emotion.py (spec section 30 area: emotion detection).
"""

from services.emotion import predict_emotion, predict_emotion_batch


def test_frustrated_example_matches_spec():
    # Spec's own worked example (section 12).
    label, confidence = predict_emotion("Your support team never responds.")
    assert label == "Frustrated"
    assert confidence > 0


def test_happy_example():
    label, _ = predict_emotion("I love it, absolutely amazing product, exceeded my expectations!")
    assert label == "Happy"


def test_angry_example():
    label, _ = predict_emotion("This is outrageous, worst experience ever, totally unacceptable.")
    assert label == "Angry"


def test_disappointed_example():
    label, _ = predict_emotion("Really disappointed, not what I expected at all, I regret buying this.")
    assert label == "Disappointed"


def test_satisfied_example():
    label, _ = predict_emotion("Good value, works well, does the job reliably.")
    assert label == "Satisfied"


def test_empty_text_is_neutral():
    label, confidence = predict_emotion("")
    assert label == "Neutral"
    assert confidence == 0.0


def test_no_lexicon_match_falls_back_to_sentiment():
    positive_label, _ = predict_emotion("It is what it is.", sentiment="Positive")
    negative_label, _ = predict_emotion("It is what it is.", sentiment="Negative")
    assert positive_label == "Satisfied"
    assert negative_label == "Disappointed"


def test_negation_prevents_false_positive():
    # "not happy" should not be scored as Happy.
    label, _ = predict_emotion("I am not happy with this at all, quite disappointed.")
    assert label != "Happy"


def test_predict_emotion_batch_matches_single_predictions():
    texts = ["I love it, amazing!", "This is outrageous and unacceptable."]
    sentiments = ["Positive", "Negative"]
    batch_results = predict_emotion_batch(texts, sentiments)
    for text, sentiment, (batch_label, batch_conf) in zip(texts, sentiments, batch_results):
        single_label, single_conf = predict_emotion(text, sentiment)
        assert batch_label == single_label
        assert batch_conf == single_conf
