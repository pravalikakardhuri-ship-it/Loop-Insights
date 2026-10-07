"""
services/llm_service.py
--------------------------
Groq LLM integration. Handles:
    - AI Summary generation (spec section 20)
    - AI Recommendation generation (spec section 21)
    - Question answering (used by the LOOP Insight Assistant, Phase 8)

Core rule enforced throughout this file: the LLM is only ever given
numbers that were already calculated by Python/MySQL (via
services/analytics.py). It is never asked to estimate, guess, or "roughly
recall" a statistic -- only to explain, prioritize, and phrase in business
language numbers it's handed. Prompts explicitly instruct the model not to
invent figures beyond what's provided, and responses aren't taken on
faith: recommendation JSON is parsed defensively.

Note on testing: this environment has no network access to
api.groq.com, so none of these functions have been tested against the
real Groq API. Tests instead mock `_get_client()` to verify the request
shape and response parsing. Try it with your own GROQ_API_KEY once you
have the app running.
"""

import json
from functools import lru_cache

from config import config


class LLMServiceError(RuntimeError):
    """Raised for any Groq/LLM failure -- missing key, API error, timeout,
    or unparseable response. Callers should catch this and show a friendly
    message rather than crash."""


@lru_cache(maxsize=1)
def _get_client():
    if not config.GROQ_API_KEY:
        raise LLMServiceError(
            "GROQ_API_KEY is not set. Add it to your .env file to use AI Summary, "
            "AI Recommendations, or the LOOP Insight Assistant."
        )
    from groq import Groq

    # Explicit generous timeout + retries. The SDK's own default (60s read
    # timeout, 2 retries) is already reasonable, but a slow or restrictive
    # network (college WiFi, corporate proxy) can still exceed it -- being
    # explicit here also means the number is documented and tunable in one
    # place rather than relying on a library default that could change.
    return Groq(api_key=config.GROQ_API_KEY, timeout=90.0, max_retries=3)


def _chat(messages: list[dict], *, json_mode: bool = False, temperature: float = 0.3, max_tokens: int = 1200) -> str:
    """Low-level call to the Groq chat completion endpoint. Wraps every
    failure mode (missing key, network/API error, timeout) into
    LLMServiceError so callers never have to deal with groq's exception
    types directly.
    """
    client = _get_client()
    kwargs = dict(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - any groq/httpx exception becomes a friendly error
        error_text = str(exc).lower()
        if "timed out" in error_text or "timeout" in error_text:
            raise LLMServiceError(
                "Groq API request timed out. This is almost always a network issue, not "
                "an app bug -- common causes: restrictive WiFi (college/office networks "
                "sometimes block or throttle external APIs), an active VPN, or a slow/"
                "unstable connection. Try: switching networks (e.g. mobile hotspot) to "
                "confirm, disabling any VPN, or just retrying -- then use Settings -> "
                "Test Groq Connection to verify it's working."
            ) from exc
        raise LLMServiceError(f"Groq API request failed: {exc}") from exc

    if not completion.choices:
        raise LLMServiceError("Groq API returned no response.")

    content = completion.choices[0].message.content
    if not content or not content.strip():
        raise LLMServiceError("Groq API returned an empty response.")
    return content


# ---------------------------------------------------------------------------
# AI Summary (spec section 20)
# ---------------------------------------------------------------------------

def _format_analytics_for_prompt(analytics: dict, indent: str = "") -> str:
    """Turn a KPI/analytics dict into a compact, unambiguous text block for
    the prompt -- deliberately plain so the model can't mistake formatting
    for additional data. Handles nested lists and dicts (e.g. a
    product-specific stats block) by indenting them, rather than falling
    back to a raw Python repr that's harder for the model to parse
    reliably.
    """
    lines = []
    for key, value in analytics.items():
        if isinstance(value, list):
            lines.append(f"{indent}{key}:")
            for item in value:
                lines.append(f"{indent}  - {item}")
        elif isinstance(value, dict):
            lines.append(f"{indent}{key}:")
            lines.append(_format_analytics_for_prompt(value, indent + "  "))
        else:
            lines.append(f"{indent}{key}: {value}")
    return "\n".join(lines)


def generate_summary(analytics: dict) -> str:
    """Generate a business-friendly summary from already-computed
    analytics (spec section 20). `analytics` should come from
    services.analytics (e.g. compute_kpis + top complaints), never from
    raw feedback text -- the model summarizes numbers it's given, it
    doesn't re-derive them.
    """
    stats_block = _format_analytics_for_prompt(analytics)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a business analyst summarizing customer feedback analytics for "
                "a non-technical audience. You will be given pre-calculated statistics -- "
                "use ONLY the numbers provided. Do not invent, estimate, or round in a way "
                "that changes their meaning. Do not mention percentages or counts that "
                "were not given to you. Write 3-5 sentences in plain business language."
            ),
        },
        {
            "role": "user",
            "content": f"Here are the customer feedback analytics:\n\n{stats_block}\n\n"
                       "Write a short business-friendly summary of what these numbers show.",
        },
    ]
    return _chat(messages, temperature=0.3, max_tokens=400).strip()


# ---------------------------------------------------------------------------
# AI Recommendations (spec section 21)
# ---------------------------------------------------------------------------

_RECOMMENDATION_SCHEMA_INSTRUCTIONS = """
Respond with ONLY a JSON object (no markdown, no explanation) in this exact shape:
{
  "recommendations": [
    {
      "issue": "short issue name",
      "evidence": "the specific statistic from the data that supports this",
      "recommended_action": "a concrete action the business should take",
      "priority": "High" | "Medium" | "Low",
      "expected_impact": "a short, honest statement of expected impact"
    }
  ]
}
Generate at most 5 recommendations, ordered by priority (High first). Every
"evidence" value MUST be a statistic that was given to you in the data -- do
not invent numbers.
"""


def generate_recommendations(analytics: dict) -> list[dict]:
    """Generate prioritized, evidence-based recommendations from
    already-computed analytics (spec section 21). Returns a list of dicts
    with keys: issue, evidence, recommended_action, priority, expected_impact.

    Raises LLMServiceError if the API call fails OR if the response can't
    be parsed as valid JSON in the expected shape -- callers should show a
    friendly message rather than silently display garbage.
    """
    stats_block = _format_analytics_for_prompt(analytics)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a business analyst producing actionable recommendations from "
                "customer feedback analytics. You will be given pre-calculated statistics -- "
                "every recommendation's evidence must cite one of those exact statistics. "
                "Never invent a number that wasn't provided." + _RECOMMENDATION_SCHEMA_INSTRUCTIONS
            ),
        },
        {
            "role": "user",
            "content": f"Here are the customer feedback analytics:\n\n{stats_block}\n\n"
                       "Generate the recommendations JSON now.",
        },
    ]
    raw = _chat(messages, json_mode=True, temperature=0.2, max_tokens=1200)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(f"Groq returned invalid JSON for recommendations: {exc}") from exc

    recommendations = parsed.get("recommendations")
    if not isinstance(recommendations, list):
        raise LLMServiceError("Groq's recommendation response was missing the expected 'recommendations' list.")

    required_keys = {"issue", "evidence", "recommended_action", "priority", "expected_impact"}
    valid = [r for r in recommendations if isinstance(r, dict) and required_keys.issubset(r.keys())]
    if not valid:
        raise LLMServiceError("Groq's recommendation response didn't match the expected format.")

    return valid[:5]


# ---------------------------------------------------------------------------
# Question Answering (used by Phase 8 - LOOP Insight Assistant)
# ---------------------------------------------------------------------------

def answer_question(question: str, context: dict) -> str:
    """Answer a user's question using only the provided context (retrieved
    analytics/data). Built now so Phase 8 (LOOP Insight Assistant) can use
    it directly -- the "don't invent numbers, say so if data is missing"
    behavior is enforced here at the prompt level so every caller gets it
    for free.
    """
    context_block = _format_analytics_for_prompt(context) if context else "(no data available)"
    messages = [
        {
            "role": "system",
            "content": (
                "You are the LOOP Insight Assistant, answering questions about customer "
                "feedback analytics using ONLY the data provided below. Do not invent "
                "numbers, products, or complaints that aren't in the data. If the data "
                "provided doesn't contain enough information to answer the question, "
                "respond exactly with: \"I don't have enough information in the available "
                "feedback data to answer that accurately.\""
            ),
        },
        {
            "role": "user",
            "content": f"Data:\n{context_block}\n\nQuestion: {question}",
        },
    ]
    return _chat(messages, temperature=0.2, max_tokens=700).strip()


def reload_client() -> None:
    """Clear the cached Groq client (call after the API key changes)."""
    _get_client.cache_clear()
