"""
ui/settings.py
----------------
The Settings page: environment/connection status, database initialization,
and model training/backfill controls.

Factored out of app.py (same reasoning as ui/upload_feedback.py and
ui/manual_feedback.py -- keep individual pages out of the main file as
they grow).
"""

import json

import streamlit as st

from ui.theme import page_header

from config import config


def _render_database_section() -> None:
    st.markdown("**MySQL**")
    st.code(
        f"Host: {config.MYSQL_HOST}\n"
        f"Port: {config.MYSQL_PORT}\n"
        f"Database: {config.MYSQL_DATABASE}",
        language="text",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Connection"):
            from database import test_connection

            ok, message = test_connection()
            (st.success if ok else st.error)(message)
    with col2:
        if st.button("Initialize Tables"):
            try:
                from database.models import init_db, Base

                init_db()
                table_names = list(Base.metadata.tables.keys())
                st.success(f"Tables ready: {', '.join(table_names)}")
            except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
                st.error(f"Could not create tables: {exc}")


def _render_groq_section() -> None:
    st.markdown("**Groq API**")
    if config.GROQ_API_KEY:
        st.success("GROQ_API_KEY is set.")
        st.caption(f"Model: `{config.GROQ_MODEL}`")
        if st.button("Test Groq Connection"):
            with st.spinner("Sending a test request to Groq..."):
                try:
                    from services.llm_service import generate_summary, LLMServiceError

                    generate_summary({"total_feedback": 1, "note": "connection test"})
                    st.success("Groq API is reachable and responded successfully.")
                except LLMServiceError as exc:
                    st.error(str(exc))
    else:
        st.warning("GROQ_API_KEY is not set. Add it to your .env file.")


def _render_sentiment_model_section() -> None:
    from services.sentiment import is_model_available, reload_model

    st.markdown("**Sentiment Model**")
    if is_model_available():
        st.success(f"Model found at `{config.SENTIMENT_MODEL_PATH.name}`.")
        metrics_path = config.MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                st.caption(
                    f"Last training run — accuracy: {metrics['accuracy']:.2%}, "
                    f"F1 (weighted): {metrics['f1_weighted']:.2%}, "
                    f"test set size: {metrics['test_size']}"
                )
            except Exception:  # noqa: BLE001 - metrics display is best-effort
                pass
    else:
        st.warning("No trained sentiment model found yet.")

    if st.button("Train / Retrain Sentiment Model"):
        with st.spinner("Training model on data/sample_feedback.csv..."):
            try:
                from training.train_sentiment import train

                metrics = train()
                reload_model()
                st.success(
                    f"Model trained. Accuracy: {metrics['accuracy']:.2%}, "
                    f"F1 (weighted): {metrics['f1_weighted']:.2%}"
                )
            except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
                st.error(f"Training failed: {exc}")

    st.caption(
        "If Manual Feedback or Upload Feedback shows a scikit-learn compatibility error, "
        "click this button to regenerate the model file using your installed scikit-learn version."
    )


def _render_topic_model_section() -> None:
    from services.topic_model import is_model_available, reload_model

    st.markdown("**Topic / Issue Model**")
    if is_model_available():
        st.success(f"Model found at `{config.TOPIC_MODEL_PATH.name}`.")
        metrics_path = config.MODELS_DIR / "topic_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                st.caption(
                    f"Last training run — accuracy: {metrics['accuracy']:.2%}, "
                    f"F1 (weighted): {metrics['f1_weighted']:.2%}, "
                    f"test set size: {metrics['test_size']}"
                )
                if metrics.get("excluded_labels_insufficient_data"):
                    st.caption(
                        f"Not trained on (too few examples in the dataset): "
                        f"{', '.join(metrics['excluded_labels_insufficient_data'])}"
                    )
            except Exception:  # noqa: BLE001 - metrics display is best-effort
                pass
        with st.expander("How these labels were generated (read before trusting the accuracy)"):
            st.write(
                "Training labels came from keyword matching (there's no human-annotated topic "
                "data), and the model itself is TF-IDF-based, which is also keyword-driven. "
                "That means this accuracy only measures how well the model reproduces the keyword "
                "rules on sentence templates it has not seen — not how well it generalizes to "
                "topics described in completely novel vocabulary. It's still more useful than plain "
                "keyword matching at prediction time, since it picks up on word combinations "
                "and related phrasing, not just exact matches."
            )
    else:
        st.warning("No trained topic model found yet.")

    if st.button("Train / Retrain Topic Model"):
        with st.spinner("Training model on data/sample_feedback.csv..."):
            try:
                from training.train_topic import train

                metrics = train()
                reload_model()
                st.success(
                    f"Model trained. Accuracy: {metrics['accuracy']:.2%}, "
                    f"F1 (weighted): {metrics['f1_weighted']:.2%}"
                )
            except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
                st.error(f"Training failed: {exc}")


def _render_nlp_backfill_section() -> None:
    st.markdown("**Backfill Emotion & Topic Analysis**")
    st.caption(
        "For feedback records that were imported before Emotion/Topic detection existed "
        "(e.g. during Phase 3/4), or that only have sentiment analyzed."
    )

    if st.button("Run Emotion & Topic Analysis on Records Missing It"):
        try:
            from database.queries import get_feedback_needing_nlp_analysis, update_emotion_and_topic
            from services.emotion import predict_emotion_batch
            from services.topic_model import predict_topic_batch, is_model_available as topic_available

            pending = get_feedback_needing_nlp_analysis()
            if not pending:
                st.info("Nothing to backfill — every record already has emotion and topic analysis.")
                return

            with st.spinner(f"Analyzing {len(pending)} record(s)..."):
                texts = [r["feedback_text"] for r in pending]
                sentiments = [r["sentiment"] for r in pending]
                emotion_predictions = predict_emotion_batch(texts, sentiments)

                topic_predictions = None
                if topic_available():
                    topic_predictions = predict_topic_batch(texts)

                for i, record in enumerate(pending):
                    update_emotion_and_topic(
                        feedback_id=record["id"],
                        emotion=emotion_predictions[i][0],
                        emotion_confidence=emotion_predictions[i][1],
                        topic=topic_predictions[i][0] if topic_predictions else None,
                    )

            st.success(f"Backfilled emotion{' and topic' if topic_predictions else ''} for {len(pending)} record(s).")
            if not topic_predictions:
                st.info("Topic model isn't trained yet, so only emotion was backfilled.")
        except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
            st.error(f"Backfill failed: {exc}")

    st.markdown("**Re-analyze All Records**")
    st.caption(
        "Overwrites sentiment, emotion, and topic for EVERY feedback record using the "
        "currently trained models. Use this after retraining a model, or if you notice "
        "wrong-looking sentiment/emotion/topic on older records (e.g. saved before a model "
        "fix) — those records keep whatever was predicted at the time they were analyzed, "
        "they don't update automatically."
    )
    if st.button("Re-analyze ALL Feedback Records"):
        try:
            from database.queries import get_all_feedback, insert_analysis
            from services.sentiment import predict_sentiment_batch, is_model_available as sentiment_available
            from services.emotion import predict_emotion_batch
            from services.topic_model import predict_topic_batch, is_model_available as topic_available

            all_feedback = get_all_feedback()
            if not all_feedback:
                st.info("No feedback records found.")
                return

            texts = [r["feedback_text"] for r in all_feedback]
            ids = [r["id"] for r in all_feedback]

            sentiments, confidences = None, None
            if sentiment_available():
                with st.spinner(f"Re-running sentiment analysis on {len(ids)} record(s)..."):
                    sentiment_predictions = predict_sentiment_batch(texts)
                    sentiments = [p[0] for p in sentiment_predictions]
                    confidences = [p[1] for p in sentiment_predictions]

            with st.spinner("Re-running emotion detection..."):
                emotion_predictions = predict_emotion_batch(texts, sentiments)

            topic_predictions = None
            if topic_available():
                with st.spinner("Re-running topic detection..."):
                    topic_predictions = predict_topic_batch(texts)

            with st.spinner("Saving updated analysis..."):
                for i, feedback_id in enumerate(ids):
                    insert_analysis(
                        feedback_id=feedback_id,
                        sentiment=sentiments[i] if sentiments else None,
                        sentiment_confidence=confidences[i] if confidences else None,
                        emotion=emotion_predictions[i][0],
                        emotion_confidence=emotion_predictions[i][1],
                        topic=topic_predictions[i][0] if topic_predictions else None,
                    )

            st.success(f"Re-analyzed all {len(ids)} record(s) with the current models.")
        except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
            st.error(f"Re-analysis failed: {exc}")


def _render_complaints_preview_section() -> None:
    st.markdown("**Top Complaints (preview)**")
    st.caption(
        "Recurring negative-feedback topics, aggregated from analyzed records. "
        "A polished version of this lands on the Dashboard in Phase 6."
    )
    if st.button("Show Top Complaints"):
        try:
            from database.queries import get_top_complaints

            complaints = get_top_complaints(limit=5)
            if not complaints:
                st.info("No negative feedback with topic analysis found yet.")
            else:
                for c in complaints:
                    st.write(
                        f"**{c['topic']}** — {c['count']} mentions ({c['percentage']}% of negative feedback) "
                        f"· top products: {', '.join(c['related_products'])}"
                    )
        except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
            st.error(f"Could not load complaints: {exc}")


def render() -> None:
    from ui.admin_gate import is_admin

    if not is_admin():  # the sidebar already hides this page; this is the second lock
        st.error("Settings is only available to the admin.")
        return
    page_header("⚙️", "Settings", "Connections, models and data tools.")
    st.write("Environment, connection, and model status.")

    _render_database_section()
    _render_groq_section()
    _render_sentiment_model_section()
    _render_topic_model_section()
    _render_nlp_backfill_section()
    _render_complaints_preview_section()
