"""
tests/test_training.py
------------------------
Tests for training/train_sentiment.py -- the labeling heuristic and the
group-based train/test split that prevents duplicate-text leakage.
"""

import pandas as pd

from training.train_sentiment import rating_to_sentiment, _group_split, load_training_data


def test_rating_to_sentiment_mapping():
    assert rating_to_sentiment(5) == "Positive"
    assert rating_to_sentiment(4) == "Positive"
    assert rating_to_sentiment(3) == "Neutral"
    assert rating_to_sentiment(2) == "Negative"
    assert rating_to_sentiment(1) == "Negative"


def test_group_split_has_no_exact_text_overlap():
    df = pd.DataFrame(
        {
            "clean_feedback": [
                "Great laptop, fast delivery.",
                "Great laptop, fast delivery.",  # exact duplicate
                "Terrible phone, broke immediately.",
                "Average tablet, does the job.",
                "Great laptop, fast delivery.",  # another duplicate
                "Terrible phone, broke immediately.",
                "Okay camera for the price.",
                "Excellent headphones, love them.",
            ],
            "sentiment": ["Positive", "Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive"],
        }
    )
    train_df, test_df = _group_split(df, test_size=0.3)
    overlap = set(train_df["clean_feedback"]) & set(test_df["clean_feedback"])
    assert len(overlap) == 0
    # Every duplicate of "Great laptop, fast delivery." must land on the same side.
    laptop_rows_in_train = (train_df["clean_feedback"] == "Great laptop, fast delivery.").sum()
    laptop_rows_in_test = (test_df["clean_feedback"] == "Great laptop, fast delivery.").sum()
    assert laptop_rows_in_train == 0 or laptop_rows_in_test == 0


def test_load_training_data_produces_expected_columns():
    df = load_training_data()
    assert "clean_feedback" in df.columns
    assert "sentiment" in df.columns
    assert set(df["sentiment"].unique()) <= {"Positive", "Neutral", "Negative"}
    assert len(df) > 0
    # No empty cleaned text should remain.
    assert (df["clean_feedback"].str.strip() == "").sum() == 0
