"""
ui/loop_assistant.py
-----------------------
The LOOP Insight Assistant page (spec section 22): a chatbot that
answers questions about the feedback data, grounded strictly in data
this app has actually stored and computed -- never inventing numbers.
"""

import streamlit as st

from ui.admin_gate import is_admin
from ui.theme import AI_UNAVAILABLE, ai_problem, page_header

from config import config

EXAMPLE_QUESTIONS = [
    "What are our biggest complaints?",
    "Which product has the most negative feedback?",
    "Why are customers unhappy?",
    "Which product has the best customer satisfaction?",
    "Summarize negative feedback.",
    "What should management focus on?",
    "What are customers saying about delivery?",
]

NOT_ENOUGH_INFO = (
    "I don't have enough information in the available feedback data to answer that accurately."
)


def _render_history():
    for msg in st.session_state.get("assistant_history", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def _handle_question(question: str):
    st.session_state.setdefault("assistant_history", [])
    st.session_state["assistant_history"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking at the feedback data..."):
            try:
                from services.analytics import load_feedback_df
                from services.assistant_context import build_assistant_context
                from services.llm_service import answer_question, LLMServiceError

                df = load_feedback_df()
                context = build_assistant_context(df, question)

                if context is None:
                    answer = NOT_ENOUGH_INFO
                else:
                    answer = answer_question(question, context)

            except LLMServiceError as exc:
                answer = f"⚠️ {exc}" if is_admin() else f"⚠️ {AI_UNAVAILABLE}"
            except Exception as exc:  # noqa: BLE001 - shown to the user, not a stack trace
                answer = (
                    f"⚠️ Something went wrong answering that. Details: {exc}"
                    if is_admin()
                    else "⚠️ Something went wrong answering that. Please try again in a moment."
                )

        st.write(answer)

    st.session_state["assistant_history"].append({"role": "assistant", "content": answer})


def render() -> None:
    page_header("💬", "LOOP Assistant", "Ask questions about your feedback in plain language.")
    st.caption(
        "Ask questions about your feedback data in plain English. Answers are grounded strictly "
        "in data already stored and analyzed by this app — nothing is invented."
    )

    if not config.GROQ_API_KEY:
        ai_problem(admin_message="GROQ_API_KEY is not set. Add it to .env (or the host's secrets), then test it in Settings.", level="warning")
        return

    if not st.session_state.get("assistant_history"):
        st.write("Try one of these, or type your own question below:")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"example_q_{i}", width="stretch"):
                    _handle_question(q)
                    st.rerun()

    _render_history()

    question = st.chat_input("Ask about your feedback data...")
    if question:
        _handle_question(question)
        st.rerun()

    if st.session_state.get("assistant_history"):
        if st.button("Clear conversation"):
            st.session_state["assistant_history"] = []
            st.rerun()
