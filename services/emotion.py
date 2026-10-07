"""
services/emotion.py
---------------------
Emotion detection service.

Approach: documented lexicon/rule-based baseline (spec section 12 explicitly
allows "an NLP model or a documented baseline approach"). There is no
labeled emotion dataset available -- feedback text alone doesn't specify
"the customer felt frustrated" -- so training a supervised classifier here
would mean inventing ground truth, which the project rules explicitly
prohibit ("Do not fake AI responses"). A transparent, inspectable lexicon
is the honest choice for a baseline.

Swap-in point for later: replace `predict_emotion()`'s body with a call to
a transformer emotion-classification pipeline (e.g. a HuggingFace model)
while keeping the same function signature -- every caller in the app
(Upload, Manual Feedback) only depends on `predict_emotion(text, sentiment)
-> (label, confidence)`, so nothing else needs to change.
"""

import re

EMOTION_LABELS = ["Happy", "Satisfied", "Angry", "Frustrated", "Disappointed", "Neutral"]

# Keyword lexicons. Each phrase is matched as a whole word / short phrase
# against the lowercased, cleaned feedback text. Longer, more specific
# phrases are listed first within each category so a specific match can be
# weighted appropriately.
_LEXICON = {
    "Angry": [
        "furious", "outrageous", "unacceptable", "disgusted", "rude",
        "worst experience", "never again", "scam", "ripped off", "angry",
        "horrible", "terrible experience", "shouting", "yelled",
    ],
    "Frustrated": [
        "never responds", "never responded", "no response", "waited on hold",
        "kept transferring", "keeps crashing", "crashed", "confusing",
        "frustrated", "frustrating", "hassle", "no update", "still waiting",
        "not resolved", "unresolved", "again and again", "repeatedly",
    ],
    "Disappointed": [
        "disappointed", "disappointing", "expected better", "let down",
        "not worth", "not what i expected", "underwhelmed", "regret",
        "wouldn't recommend", "would not recommend", "downgrade",
    ],
    "Happy": [
        "love it", "amazing", "fantastic", "excellent", "wonderful",
        "delighted", "thrilled", "impressed", "best purchase", "happy",
        "exceeded my expectations", "couldn't be happier",
    ],
    "Satisfied": [
        "satisfied", "good value", "works well", "works perfectly",
        "recommend", "worth the price", "as expected", "does the job",
        "reliable", "solid", "pleased",
    ],
}

# Negation words that flip local sentiment/emotion polarity for the phrase
# immediately following them (simple, transparent heuristic -- not a full
# dependency parse, but good enough to avoid "not happy" scoring as Happy).
_NEGATIONS = {"not", "never", "no", "n't", "without"}

_WORD_PATTERN = re.compile(r"[a-z']+")


def _negation_adjusted_score(text_lower: str, phrase: str) -> float:
    """Return 1.0 for a normal match, 0.0 if the match is immediately
    preceded by a negation word within a short window (e.g. "not happy")."""
    idx = text_lower.find(phrase)
    if idx == -1:
        return 0.0
    window_start = max(0, idx - 15)
    preceding = text_lower[window_start:idx]
    preceding_words = _WORD_PATTERN.findall(preceding)
    if preceding_words and preceding_words[-1] in _NEGATIONS:
        return 0.0
    return 1.0


def _score_text(text_lower: str) -> dict:
    scores = {label: 0.0 for label in _LEXICON}
    for label, phrases in _LEXICON.items():
        for phrase in phrases:
            if phrase in text_lower:
                scores[label] += _negation_adjusted_score(text_lower, phrase)
    return scores


def predict_emotion(text: str, sentiment: str | None = None) -> tuple[str, float]:
    """Predict the customer's emotion for a piece of feedback text.

    `sentiment` (Positive/Neutral/Negative), if provided, is used only as a
    tie-breaker / fallback when the lexicon finds no keyword matches --
    e.g. clearly negative text with no specific angry/frustrated/disappointed
    keyword still shouldn't default to "Happy". This mirrors how the spec's
    own example ("Your support team never responds." -> Frustrated) implies
    combining lexical cues with overall tone.

    Returns (label, confidence) where confidence in [0, 1] reflects how
    strong the lexicon signal was, not a calibrated probability.
    """
    if not text or not text.strip():
        return "Neutral", 0.0

    text_lower = text.lower()
    scores = _score_text(text_lower)
    total_hits = sum(scores.values())

    if total_hits == 0:
        # No lexicon signal -- fall back to overall sentiment as a coarse proxy.
        if sentiment == "Positive":
            return "Satisfied", 0.4
        if sentiment == "Negative":
            return "Disappointed", 0.4
        return "Neutral", 0.5

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    confidence = min(0.5 + 0.15 * best_score, 0.95)
    return best_label, round(confidence, 4)


def predict_emotion_batch(texts: list[str], sentiments: list[str] | None = None) -> list[tuple[str, float]]:
    """Predict emotion for many texts at once."""
    if sentiments is None:
        sentiments = [None] * len(texts)
    return [predict_emotion(t, s) for t, s in zip(texts, sentiments)]
