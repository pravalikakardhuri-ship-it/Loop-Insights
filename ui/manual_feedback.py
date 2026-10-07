"""
ui/manual_feedback.py
-----------------------
The Manual Feedback page: enter one piece of feedback, analyze it, and
save it to the database.

Phase 4 scope: sentiment only. Emotion and topic fields are wired in once
services/emotion.py and services/topic_model.py exist (Phase 5) -- until
then they're simply not shown, rather than faked.
"""

import streamlit as st

from ui.theme import page_header, problem

from config import config

from datetime import date

from services.sentiment import predict_sentiment, is_model_available, ModelNotTrainedError
from services.emotion import predict_emotion
from services.topic_model import (
    predict_topic,
    is_model_available as is_topic_model_available,
    TopicModelNotTrainedError,
)
from utils.validators import validate_rating_value, validate_feedback_text

PRODUCT_TO_CATEGORY = config.product_category_map()
PRODUCTS = sorted(PRODUCT_TO_CATEGORY)
CATEGORIES = list(config.PRODUCT_CATALOG)


def render() -> None:
    page_header("✍️", "Manual Feedback", "Analyse and save a single piece of feedback.")
    st.write("Enter a single piece of customer feedback to analyze and save.")

    if not is_model_available():
        st.warning(
            "No trained sentiment model found yet. Run `python -m training.train_sentiment`, "
            "or use the **Train Model** button on the Settings page, then come back here."
        )

    with st.form("manual_feedback_form"):
        col1, col2 = st.columns(2)
        with col1:
            product = st.selectbox("Product", PRODUCTS)
        with col2:
            # A form can't update one field from another, so the category
            # is looked up from the product when the form is submitted.
            st.text_input("Category", value="Set automatically from the product", disabled=True)
        category = PRODUCT_TO_CATEGORY[product]

        rating = st.slider("Rating", min_value=1, max_value=5, value=3)
        feedback_text = st.text_area(
            "Customer Feedback",
            placeholder="e.g. The phone is excellent but the delivery was very late.",
            height=120,
        )

        analyze_clicked = st.form_submit_button("Analyze Feedback", type="primary")

    if not analyze_clicked:
        return

    rating_ok, rating_msg = validate_rating_value(rating)
    if not rating_ok:
        st.error(rating_msg)
        return

    text_ok, text_msg = validate_feedback_text(feedback_text)
    if not text_ok:
        st.error(text_msg)
        return

    try:
        sentiment, confidence = predict_sentiment(feedback_text)
    except ModelNotTrainedError as exc:
        st.error(str(exc))
        return

    emotion, emotion_confidence = predict_emotion(feedback_text, sentiment=sentiment)

    topic, topic_confidence = None, None
    if is_topic_model_available():
        try:
            topic, topic_confidence = predict_topic(feedback_text)
        except TopicModelNotTrainedError:
            pass

    st.subheader("Analysis Result")
    result_cols = st.columns(4 if topic else 3)
    result_cols[0].metric("Sentiment", sentiment)
    result_cols[1].metric("Confidence", f"{confidence * 100:.1f}%")
    result_cols[2].metric("Emotion", emotion)
    if topic:
        result_cols[3].metric("Topic", topic)
    if not topic:
        st.caption("Topic detection unavailable — train the topic model from Settings.")

    try:
        from database.queries import insert_feedback, insert_analysis

        feedback_id = insert_feedback(
            customer_id="MANUAL",
            product=product,
            category=category,
            rating=rating,
            feedback_text=feedback_text,
            customer_name="Manual Entry",
            feedback_date=date.today(),
        )
        insert_analysis(
            feedback_id=feedback_id,
            sentiment=sentiment,
            sentiment_confidence=confidence,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            topic=topic,
            processed_text=feedback_text,
        )
        st.success("Saved to the database.")
    except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
        problem(
            "The analysis worked, but we couldn't save it right now. Please try again in a moment.",
            "Analysis succeeded, but the result could not be saved to the database. "
            "Make sure MySQL is connected and tables are initialized (see Settings).",
            exc,
        )
