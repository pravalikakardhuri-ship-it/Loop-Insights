"""
training/evaluate_model.py
----------------------------
Standalone evaluation of the currently saved sentiment model
(models/sentiment_model.pkl + models/tfidf_vectorizer.pkl).

This is independent of train_sentiment.py: it reloads the saved artifacts
from disk (rather than the in-memory objects from a training run) so you
can verify what's actually deployed, e.g. after copying a model file
between environments.

Run:
    python -m training.evaluate_model
"""

import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from config import config
from training.train_sentiment import load_training_data, _group_split


def evaluate(csv_path=None) -> dict:
    if not config.SENTIMENT_MODEL_PATH.exists() or not config.TFIDF_VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "No saved model found. Run `python -m training.train_sentiment` first."
        )

    model = joblib.load(config.SENTIMENT_MODEL_PATH)
    vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)

    df = load_training_data(csv_path)
    # Same grouped split as training, so this evaluates on the same
    # held-out rows the model never trained on (see train_sentiment.py for
    # why the split is grouped by template rather than by row).
    _, test_df = _group_split(df, test_size=0.2)
    X_test, y_test = test_df["clean_feedback"], test_df["sentiment"]

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    labels = sorted(df["sentiment"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    print(f"Evaluated on {len(X_test)} held-out records")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1 (weighted):        {f1:.4f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print("Labels:", labels)
    print(cm)
    print("\nFull classification report:")
    print(report)

    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    evaluate()
