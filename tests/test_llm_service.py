"""
tests/test_llm_service.py
----------------------------
Tests for services/llm_service.py (spec section 30: LLM -- handle
successful response, handle API failure).

IMPORTANT: this sandbox has no network access to api.groq.com, so these
tests mock the Groq client entirely rather than making real API calls.
They verify the request shape (system/user messages, json mode) and the
response parsing/validation logic, but they cannot verify real-world
Groq API behavior. Test manually with a real GROQ_API_KEY once the app
is running on your machine.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from services import llm_service
from services.llm_service import LLMServiceError


@pytest.fixture(autouse=True)
def _reset_client_cache():
    llm_service.reload_client()
    yield
    llm_service.reload_client()


def _mock_completion(content: str):
    """Build a MagicMock shaped like a groq ChatCompletion response."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


def test_get_client_raises_without_api_key():
    with patch("services.llm_service.config") as mock_config:
        mock_config.GROQ_API_KEY = ""
        with pytest.raises(LLMServiceError, match="GROQ_API_KEY is not set"):
            llm_service._get_client()


def test_generate_summary_returns_text_on_success():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion(
        "Customers are largely satisfied, though delivery delays are a recurring concern."
    )
    with patch("services.llm_service._get_client", return_value=fake_client):
        result = llm_service.generate_summary({"total_feedback": 100, "positive_pct": 60})
    assert "delivery" in result.lower() or len(result) > 0

    # Verify the prompt actually included the data we gave it.
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    combined_content = " ".join(m["content"] for m in call_kwargs["messages"])
    assert "total_feedback: 100" in combined_content
    assert "positive_pct: 60" in combined_content


def test_generate_summary_raises_on_api_failure():
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = Exception("500 internal server error")
    with patch("services.llm_service._get_client", return_value=fake_client):
        with pytest.raises(LLMServiceError, match="Groq API request failed"):
            llm_service.generate_summary({"total_feedback": 100})


def test_generate_summary_raises_friendly_message_on_timeout():
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = Exception("Request timed out.")
    with patch("services.llm_service._get_client", return_value=fake_client):
        with pytest.raises(LLMServiceError, match="network issue"):
            llm_service.generate_summary({"total_feedback": 100})


def test_generate_summary_raises_on_empty_response():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion("")
    with patch("services.llm_service._get_client", return_value=fake_client):
        with pytest.raises(LLMServiceError, match="empty response"):
            llm_service.generate_summary({"total_feedback": 100})


def test_generate_recommendations_parses_valid_json():
    payload = {
        "recommendations": [
            {
                "issue": "Delivery delays",
                "evidence": "32% of negative feedback mentions delivery",
                "recommended_action": "Improve logistics monitoring",
                "priority": "High",
                "expected_impact": "Reduced negative feedback volume",
            }
        ]
    }
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion(json.dumps(payload))
    with patch("services.llm_service._get_client", return_value=fake_client):
        result = llm_service.generate_recommendations({"top_complaints": ["Delivery: 32%"]})

    assert len(result) == 1
    assert result[0]["issue"] == "Delivery delays"
    assert result[0]["priority"] == "High"

    # Verify json_object mode was requested.
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_generate_recommendations_raises_on_invalid_json():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion("not valid json {{{")
    with patch("services.llm_service._get_client", return_value=fake_client):
        with pytest.raises(LLMServiceError, match="invalid JSON"):
            llm_service.generate_recommendations({"total_feedback": 100})


def test_generate_recommendations_raises_on_missing_key():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion(json.dumps({"wrong_key": []}))
    with patch("services.llm_service._get_client", return_value=fake_client):
        with pytest.raises(LLMServiceError, match="missing the expected"):
            llm_service.generate_recommendations({"total_feedback": 100})


def test_generate_recommendations_filters_malformed_entries():
    payload = {
        "recommendations": [
            {"issue": "Good one", "evidence": "e", "recommended_action": "a", "priority": "High", "expected_impact": "i"},
            {"issue": "Missing fields"},  # malformed, should be dropped
        ]
    }
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion(json.dumps(payload))
    with patch("services.llm_service._get_client", return_value=fake_client):
        result = llm_service.generate_recommendations({"total_feedback": 100})
    assert len(result) == 1
    assert result[0]["issue"] == "Good one"


def test_generate_recommendations_caps_at_five():
    payload = {
        "recommendations": [
            {"issue": f"Issue {i}", "evidence": "e", "recommended_action": "a", "priority": "Low", "expected_impact": "i"}
            for i in range(8)
        ]
    }
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion(json.dumps(payload))
    with patch("services.llm_service._get_client", return_value=fake_client):
        result = llm_service.generate_recommendations({"total_feedback": 100})
    assert len(result) == 5


def test_answer_question_returns_fallback_message_when_model_says_so():
    fallback = "I don't have enough information in the available feedback data to answer that accurately."
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion(fallback)
    with patch("services.llm_service._get_client", return_value=fake_client):
        result = llm_service.answer_question("What color do customers prefer?", {"total_feedback": 100})
    assert result == fallback


def test_answer_question_includes_context_in_prompt():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion("Delivery is the top complaint.")
    with patch("services.llm_service._get_client", return_value=fake_client):
        llm_service.answer_question("What's our biggest complaint?", {"top_complaint": "Delivery"})

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    combined_content = " ".join(m["content"] for m in call_kwargs["messages"])
    assert "top_complaint: Delivery" in combined_content
    assert "What's our biggest complaint?" in combined_content


def test_answer_question_formats_nested_product_dict_readably():
    # Regression test: a nested dict (e.g. assistant_context's
    # focused_product_stats) must be formatted as readable indented
    # lines, not dumped as a raw Python dict repr, which is harder for
    # the model to parse reliably and was part of why some answers
    # were inaccurate.
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_completion("The Smartwatch has mostly negative feedback.")
    context = {
        "focused_product_stats": {
            "product": "Smartwatch",
            "negative_pct": 80.0,
        }
    }
    with patch("services.llm_service._get_client", return_value=fake_client):
        llm_service.answer_question("What about the Smartwatch?", context)

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    combined_content = " ".join(m["content"] for m in call_kwargs["messages"])
    assert "{'product'" not in combined_content  # not a raw dict repr
    assert "product: Smartwatch" in combined_content
    assert "negative_pct: 80.0" in combined_content
