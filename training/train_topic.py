"""
training/train_topic.py
--------------------------
Trains the topic/issue classification model:

    Feedback Text -> Text Cleaning -> TF-IDF -> Logistic Regression -> Topic

Labeling strategy (read this before trusting the metrics)
------------------------------------------------------------
Like sentiment (see train_sentiment.py), there is no pre-existing "topic"
label in the raw feedback -- it has to come from somewhere. Here it comes
from `weak_label_topic()` below: a keyword lexicon built from the same
categories the spec defines (Delivery, Product Quality, Price, Customer
Support, Refund, Packaging, Payment, Website/App, Product Features,
Availability, Other). This is a WEAK / heuristic label, not ground truth,
and it is explicitly named "weak_label_" to make that visible everywhere
it's used.

Why train a model at all instead of just keyword-matching at inference
time? Because the spec asks for "NLP techniques rather than relying
entirely on hard-coded if/else conditions" and for the topic pipeline to
be "Text -> TF-IDF/embeddings -> Topic classification". A TF-IDF +
Logistic Regression model trained on these weak labels generalizes beyond
the exact keyword list (it learns related vocabulary, word combinations,
and context), which plain if/else keyword matching cannot do. But the
ceiling on how good this model can be is bounded by how good the weak
labels are -- this is disclosed in the README and in Settings, not hidden.

Note: "Availability" has no representation in the bundled sample dataset
(no template mentions stock/availability issues), so it's automatically
excluded from training if it has too few weak-label examples. It will
simply never be predicted until the dataset includes real examples of it.

Run:
    python -m training.train_topic

Outputs:
    models/topic_model.pkl
    models/topic_vectorizer.pkl
    models/topic_metrics.json
"""

import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit

from config import config
from training.train_sentiment import add_template_key
from services.preprocessing import clean_text

RANDOM_STATE = 42
MIN_EXAMPLES_PER_CLASS = 8  # classes with fewer weak-label examples than this are excluded

# Priority-ordered keyword lexicon. Checked top-to-bottom; the first topic
# whose phrase appears in the text wins. Order matters -- more specific
# categories are checked before more general ones (e.g. "Refund" before
# the more generic "Customer Support").
_TOPIC_LEXICON = [
    ("Refund", ["refund", "return window", "money back"]),
    ("Payment", [
        "payment was deducted", "charged twice", "charged extra",
        "charged shipping fees", "deducted twice",
    ]),
    ("Website/App", [
        "app crashed", "app kept crashing", "website glitched", "crashed twice",
    ]),
    ("Packaging", [
        "packaging was flimsy", "packaging looked", "packaging was neat",
        "packaging was simple", "arrived damaged", "opened before",
        "missing from the box", "seal of the", "leaking",
    ]),
    ("Customer Support", [
        "customer support", "customer service", "support team", "support never",
        "support kept", "waited on hold", "support hasn't helped",
        "warranty support", "support was rude", "resolved my", "solved my",
    ]),
    ("Delivery", [
        "arrived late", "days late", "delivery was delayed", "delivery was super fast",
        "fast shipping", "arrived in perfect condition", "tracking showed",
        "never arrived", "delivery was on time", "delivery was normal",
        "arrived early", "arrived on schedule", "communication from the courier",
        "late delivery", "order arrived",
    ]),
    ("Price", [
        "overpriced", "worth every rupee", "worth the price", "not worth the price",
        "price range", "value for money", "good value", "reasonable for the price",
    ]),
    ("Product Features", [
        "battery life", "battery", "setup was quick", "the interface", "sound and display quality",
        "easy to set up", "setup instructions", "listing photos",
        "looks similar to what was advertised", "took a moment to get used to",
        "does exactly what it promises",
        "size chart", "colour of", "easy to assemble", "assembly instructions",
        "easy to clean", "hard to clean", "installation of", "saves me so much time",
        "bit bulky", "fits okay", "fragrance is mild", "fragrance lasts",
    ]),
    ("Product Quality", [
        "build quality", "quality is poor", "quality is outstanding", "quality feels cheap",
        "stopped working", "broke on", "broke immediately", "strange noise",
        "crack near the seams", "downgrade", "overheats", "held up well",
        "impressive performance", "works perfectly", "works flawlessly",
        "functions perfectly", "no complaints", "hasn't let me down",
        "performs day to day", "impressed by how well",
        "fabric quality", "stitching", "fits perfectly", "faded", "feels durable",
        "very sturdy", "motor is weak", "wore out", "caused irritation",
        "makes a lot of noise", "great for daily workouts", "skin feels great",
    ]),
    ("Availability", ["out of stock", "sold out", "back order", "unavailable", "back in stock"]),
]


def weak_label_topic(text: str) -> str:
    """Heuristic (NOT ground-truth) topic label used only to bootstrap
    training data. See module docstring."""
    text_lower = text.lower()
    for topic, phrases in _TOPIC_LEXICON:
        if any(phrase in text_lower for phrase in phrases):
            return topic
    return "Other"


def load_training_data(csv_path=None) -> pd.DataFrame:
    csv_path = csv_path or config.SAMPLE_DATASET_PATH
    df = pd.read_csv(csv_path)
    df["clean_feedback"] = df["feedback"].apply(clean_text)
    df["topic"] = df["feedback"].apply(weak_label_topic)
    df = df[df["clean_feedback"].str.strip() != ""].reset_index(drop=True)
    return add_template_key(df)


def _group_split(df: pd.DataFrame, test_size: float = 0.2):
    """Same exact-text grouping as train_sentiment.py -- see that module
    for why this matters (prevents duplicate-sentence leakage)."""
    groups = df["template_key"] if "template_key" in df.columns else df["clean_feedback"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[test_idx]


def train(csv_path=None) -> dict:
    df = load_training_data(csv_path)

    # Drop classes with too few weak-label examples to train/evaluate meaningfully.
    class_counts = df["topic"].value_counts()
    excluded = class_counts[class_counts < MIN_EXAMPLES_PER_CLASS].index.tolist()
    if excluded:
        df = df[~df["topic"].isin(excluded)].reset_index(drop=True)

    train_df, test_df = _group_split(df, test_size=0.2)
    X_train, y_train = train_df["clean_feedback"], train_df["topic"]
    X_test, y_test = test_df["clean_feedback"], test_df["topic"]

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    labels = sorted(df["topic"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "labels": labels,
        "excluded_labels_insufficient_data": excluded,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.TOPIC_MODEL_PATH)
    joblib.dump(vectorizer, config.TOPIC_VECTORIZER_PATH)
    with open(config.MODELS_DIR / "topic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _print_metrics(metrics: dict) -> None:
    print(f"Train size: {metrics['train_size']}  Test size: {metrics['test_size']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1 (weighted):        {metrics['f1_weighted']:.4f}")
    print(f"Labels trained on: {metrics['labels']}")
    if metrics["excluded_labels_insufficient_data"]:
        print(f"Excluded (too few examples): {metrics['excluded_labels_insufficient_data']}")


if __name__ == "__main__":
    print("Training topic model...")
    result_metrics = train()
    _print_metrics(result_metrics)
    print("\nSaved model to:", config.TOPIC_MODEL_PATH)
    print("Saved vectorizer to:", config.TOPIC_VECTORIZER_PATH)
