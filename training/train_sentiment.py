"""
training/train_sentiment.py
-----------------------------
Trains the sentiment classification model:

    Feedback Text -> Text Cleaning -> TF-IDF -> Logistic Regression -> Sentiment

Labeling strategy
------------------
The sample dataset (and any uploaded dataset) has a numeric `rating`
(1-5) but no pre-existing sentiment label. To train a supervised model we
derive labels from rating as a reasonable, transparent bootstrap:

    rating 4-5 -> Positive
    rating 3   -> Neutral
    rating 1-2 -> Negative

This mapping is a labeling heuristic, not a claim about ground truth. It
lets the model learn language patterns associated with each class from
real text, and all metrics below (accuracy, precision, recall, F1) are
computed honestly on a held-out test split -- nothing here is invented or
hard-coded.

Run:
    python -m training.train_sentiment

Outputs:
    models/sentiment_model.pkl
    models/tfidf_vectorizer.pkl
    models/metrics.json          (accuracy / precision / recall / f1 / confusion matrix)
"""

import json
import re

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
from services.preprocessing import clean_text

RANDOM_STATE = 42


def rating_to_sentiment(rating: int) -> str:
    """Bootstrap label derived from the star rating (see module docstring)."""
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def load_training_data(csv_path=None) -> pd.DataFrame:
    csv_path = csv_path or config.SAMPLE_DATASET_PATH
    df = pd.read_csv(csv_path)
    df["clean_feedback"] = df["feedback"].apply(clean_text)
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    # Drop any rows that ended up with empty text after cleaning.
    df = df[df["clean_feedback"].str.strip() != ""].reset_index(drop=True)
    return add_template_key(df)


def add_template_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `template_key` column: the cleaned feedback with the product name
    removed, so "great laptop" and "great camera" share one key. Used to keep
    a whole sentence template on one side of the train/test split."""
    if "product" not in df.columns:
        df["template_key"] = df["clean_feedback"]
        return df

    def strip_product(row) -> str:
        product = str(row["product"]).strip()
        text = row["clean_feedback"]
        if not product:
            return text
        # Case-insensitive, so "Vacuum Cleaner" matches "the vacuum cleaner".
        return re.sub(re.escape(product), " ", text, flags=re.IGNORECASE).strip()

    df["template_key"] = df.apply(strip_product, axis=1)
    return df


def _group_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split by EXACT feedback text (not by row).

    This dataset is generated from a fixed set of sentence templates, so
    the same exact sentence can appear more than once (e.g. two different
    customers giving the same product the same complaint). Splitting by
    row alone lets an identical sentence land in both train and test,
    which lets the model "memorize" rather than generalize and inflates
    accuracy to a misleading ~100%.

    With many products, the same sentence template appears once per product
    ("great laptop", "great camera", ...), so exact-text grouping alone still
    leaks templates across the split. When a `template_key` column is present
    (see add_template_key) the split groups by that instead, so every test
    sentence is a template the model never saw in training. Without the
    column it falls back to exact-text grouping.
    """
    groups = df["template_key"] if "template_key" in df.columns else df["clean_feedback"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[test_idx]


def train(csv_path=None) -> dict:
    """Train the model and persist model/vectorizer/metrics to disk.

    Returns the metrics dict (also written to models/metrics.json).
    """
    df = load_training_data(csv_path)

    train_df, test_df = _group_split(df, test_size=0.2)
    X_train, y_train = train_df["clean_feedback"], train_df["sentiment"]
    X_test, y_test = test_df["clean_feedback"], test_df["sentiment"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
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
    labels = sorted(df["sentiment"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.SENTIMENT_MODEL_PATH)
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)
    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _print_metrics(metrics: dict) -> None:
    print(f"Train size: {metrics['train_size']}  Test size: {metrics['test_size']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1 (weighted):        {metrics['f1_weighted']:.4f}")
    print(f"Labels: {metrics['labels']}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    for label, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        print(f"  {label:>10}: {row}")


if __name__ == "__main__":
    print("Training sentiment model...")
    result_metrics = train()
    _print_metrics(result_metrics)
    print("\nSaved model to:", config.SENTIMENT_MODEL_PATH)
    print("Saved vectorizer to:", config.TFIDF_VECTORIZER_PATH)
    print("Saved metrics to:", config.MODELS_DIR / "metrics.json")
