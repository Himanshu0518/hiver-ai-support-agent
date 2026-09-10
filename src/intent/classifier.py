"""
Intent Classifier with LLM fallback chain.

Supports three tiers:
  1. Gemini (primary)  – structured output via Pydantic
  2. Groq  (fallback)  – structured output via Pydantic
  3. Keyword matcher    – no LLM needed
"""
import logging
import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

log = logging.getLogger(__name__)

from src.config import (
    gemini_available,
    groq_available,
    get_llm,
    get_groq_llm,
)
from src.models import (
    ClassificationResult,
    IntentEnum,
    classify_result_from_dict,
)

# ── Intent definitions (system prompt) ────────────────────────────
INTENT_DEFINITIONS = """\
1. late_delivery       – Package delayed beyond expected date.
2. missing_package     – Marked delivered but not received / lost / stolen.
3. wrong_item          – Received incorrect or different item.
4. refund_issue        – Refund inquiry, missing refund, or processing issue.
5. payment_issue       – Double charge, failed payment, or billing error.
6. order_cancellation  – Wants to cancel or reports cancellation problems.
7. return_item         – Wants to return, needs label, or return-policy question.
8. account_issue       – Login, password reset, or account-access problem.
9. technical_issue     – Website, app, or service technical problem.
10. product_inquiry    – Availability, restock, or general product question.
11. gift_card          – Gift-card balance, redeem, or related issue.
12. promotion          – Coupon, discount, deal, or promotional offer.
13. complaint          – Dissatisfaction or service complaint.
14. general_inquiry    – Anything that doesn't fit the categories above."""

# ── Lazy chains (built once per provider, then reused) ────────────
_gemini_chain = None
_groq_chain = None


def _build_chain(llm):
    """Build a LangChain chain that returns a ClassificationResult."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise customer-support intent classifier for AmazonHelp.\n\n"
         "Classify the customer message into EXACTLY ONE of these intents:\n\n"
         "{intent_definitions}\n\n"
         "Return a structured classification."),
        ("user",
         'Classify this customer message:\n\n"{message}"'),
    ])
    return prompt | llm.with_structured_output(ClassificationResult)


def _get_gemini_chain():
    global _gemini_chain
    if _gemini_chain is None:
        _gemini_chain = _build_chain(get_llm())
    return _gemini_chain


def _get_groq_chain():
    global _groq_chain
    if _groq_chain is None:
        _groq_chain = _build_chain(get_groq_llm())
    return _groq_chain


# ── Public API ────────────────────────────────────────────────────
def classify_with_llm(message: str) -> ClassificationResult:
    """Classify using the best available LLM (Gemini → Groq).

    Falls back to keyword matching if no LLM is reachable.
    """
    # Try Gemini first
    if gemini_available():
        try:
            chain = _get_gemini_chain()
            result: ClassificationResult = chain.invoke({
                "intent_definitions": INTENT_DEFINITIONS,
                "message": message,
            })
            return result
        except Exception as e:
            log.warning("Gemini classification error: %s — falling back to Groq", e)

    # Try Groq second
    if groq_available():
        try:
            chain = _get_groq_chain()
            result: ClassificationResult = chain.invoke({
                "intent_definitions": INTENT_DEFINITIONS,
                "message": message,
            })
            return result
        except Exception as e:
            log.warning("Groq classification error: %s — falling back to keywords", e)

    # Keyword fallback
    return classify_with_keywords(message)


def classify_with_llm_and_provider(message: str) -> tuple[ClassificationResult, str]:
    """Classify and return (result, actual_provider_used).

    Provider is one of: 'gemini', 'groq', 'keyword'.
    """
    if gemini_available():
        try:
            chain = _get_gemini_chain()
            result: ClassificationResult = chain.invoke({
                "intent_definitions": INTENT_DEFINITIONS,
                "message": message,
            })
            return result, "gemini"
        except Exception as e:
            log.warning("Gemini classification error: %s — falling back to Groq", e)

    if groq_available():
        try:
            chain = _get_groq_chain()
            result: ClassificationResult = chain.invoke({
                "intent_definitions": INTENT_DEFINITIONS,
                "message": message,
            })
            return result, "groq"
        except Exception as e:
            log.warning("Groq classification error: %s — falling back to keywords", e)

    result = classify_with_keywords(message)
    return result, "keyword"


# ── Keep old names for backward compat ────────────────────────────
def classify_with_gemini(message: str) -> ClassificationResult:
    """Classify using the best available LLM (backward-compat alias)."""
    return classify_with_llm(message)


# ── Keyword fallback ──────────────────────────────────────────────
KEYWORD_MAP: Dict[str, List[str]] = {
    "late_delivery": [
        "late", "delayed", "delay", "not arrived", "hasn't arrived",
        "where is", "when will", "overdue", "still waiting",
        "expected delivery", "delivery date",
    ],
    "missing_package": [
        "missing", "never received", "not received", "didn't receive",
        "lost", "stolen", "not delivered", "package missing",
    ],
    "wrong_item": [
        "wrong item", "wrong product", "incorrect item",
        "received wrong", "different item", "not what i ordered",
    ],
    "refund_issue": [
        "refund", "money back", "reimburse", "credited back",
        "refund status",
    ],
    "payment_issue": [
        "charged twice", "double charge", "overcharged", "payment",
        "billing", "transaction", "payment failed",
    ],
    "order_cancellation": [
        "cancel", "cancellation", "cancel order", "stop order",
    ],
    "return_item": [
        "return", "send back", "returning", "return label",
        "return shipping",
    ],
    "account_issue": [
        "account", "password", "login", "sign in",
        "can't access", "locked out", "suspended",
    ],
    "technical_issue": [
        "app", "website", "not working", "error", "bug", "crash",
        "technical", "loading",
    ],
    "product_inquiry": [
        "out of stock", "available", "restock", "in stock",
        "availability",
    ],
    "gift_card": [
        "gift card", "gift certificate", "redeem", "balance",
    ],
    "promotion": [
        "promo", "coupon", "discount", "deal", "offer",
        "promotion", "code",
    ],
    "complaint": [
        "unacceptable", "terrible", "worst", "disgusted",
        "furious", "angry", "frustrated",
    ],
}


def classify_with_keywords(message: str) -> ClassificationResult:
    """Fallback keyword classifier (no LLM needed)."""
    if not isinstance(message, str) or not message.strip():
        return ClassificationResult(
            intent=IntentEnum.GENERAL_INQUIRY,
            confidence=0.3,
            reasoning="Empty or invalid message",
        )

    text_lower = message.lower()
    scores: Dict[str, int] = {}
    for intent, keywords in KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score

    if scores:
        best = max(scores, key=scores.get)
        return ClassificationResult(
            intent=IntentEnum(best),
            confidence=min(0.9, 0.5 + scores[best] * 0.1),
            reasoning="Keyword match",
        )

    return ClassificationResult(
        intent=IntentEnum.GENERAL_INQUIRY,
        confidence=0.3,
        reasoning="No keyword match",
    )


def classify_batch(messages: List[str]) -> List[ClassificationResult]:
    """Classify a batch of messages using the best available strategy."""
    return [classify_with_llm(msg) for msg in messages]
