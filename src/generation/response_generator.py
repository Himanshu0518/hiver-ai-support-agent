"""
Response Generator with LLM fallback chain.

Tiers:
  1. Gemini (primary)  – structured output via Pydantic
  2. Groq  (fallback)  – structured output via Pydantic
  3. Template fallback  – no LLM needed
"""
import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    gemini_available,
    groq_available,
    get_llm,
    get_groq_llm,
)
from src.models import GenerationResult

# ── Lazy chains ───────────────────────────────────────────────────
_gemini_chain = None
_groq_chain = None

_SYSTEM_PROMPT = """\
You are a customer support response assistant for AmazonHelp.

Your task: draft a reply to a customer message, grounded in how AmazonHelp
has historically handled similar issues.

RULES:
1. Base your response STRICTLY on the historical cases provided as evidence.
2. Do NOT invent refunds, compensation, specific policies, delivery dates,
   guarantees, or account actions.
3. If the issue requires account-specific investigation or payment
   verification, recommend escalation.
4. Keep responses professional, empathetic, and actionable.
5. Mirror the tone and style of the historical AmazonHelp responses."""

_USER_TEMPLATE = """\
CUSTOMER MESSAGE:
"{customer_message}"

PREDICTED INTENT: {intent}

HISTORICAL AMAZONHELP CASES (evidence):
{evidence_text}

Draft your grounded response. Return structured output."""

_USER_PROMPT_TEMPLATE = """\
CUSTOMER MESSAGE:
"{customer_message}"

PREDICTED INTENT: {intent}

HISTORICAL AMAZONHELP CASES (evidence):
{evidence_text}

Draft your grounded response."""


def _build_chain(llm):
    """Build a LangChain chain that returns a GenerationResult."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("user", _USER_PROMPT_TEMPLATE),
    ])
    return prompt | llm.with_structured_output(GenerationResult)


def _get_gemini_chain():
    global _gemini_chain
    if _gemini_chain is None:
        _gemini_chain = _build_chain(get_llm(temperature=0.3, max_output_tokens=800))
    return _gemini_chain


def _get_groq_chain():
    global _groq_chain
    if _groq_chain is None:
        _groq_chain = _build_chain(get_groq_llm(temperature=0.3, max_output_tokens=800))
    return _groq_chain


# ── Evidence formatting ───────────────────────────────────────────
def _format_evidence(cases: List[Dict] | list) -> str:
    """Format retrieved cases into evidence text for the LLM."""
    if not cases:
        return "No relevant historical cases found."

    parts: list[str] = []
    for i, c in enumerate(cases, 1):
        if hasattr(c, "model_dump"):
            c = c.model_dump()
        case_id = c.get("case_id", f"Unknown_{i}")
        problem = str(c.get("customer_problem", "N/A"))[:300]
        resolution = str(c.get("resolution", "N/A"))[:300]
        quality = c.get("resolution_quality", "unknown")
        score = c.get("rerank_score", c.get("similarity_score", 0))
        parts.append(
            f"Case {case_id} (Quality: {quality}, Relevance: {score:.3f}):\n"
            f"  Customer problem: {problem}\n"
            f"  AmazonHelp resolution: {resolution}"
        )
    return "\n\n".join(parts)


# ── Public API ────────────────────────────────────────────────────
def generate(customer_message: str, intent: str, cases: List[Dict] | list) -> GenerationResult:
    """Generate a grounded response using the best available LLM.

    Falls back through Gemini → Groq → templates.
    """
    # Try Gemini
    if gemini_available():
        try:
            chain = _get_gemini_chain()
            result: GenerationResult = chain.invoke({
                "customer_message": customer_message,
                "intent": intent,
                "evidence_text": _format_evidence(cases),
            })
            return result
        except Exception as e:
            print(f"Gemini generation error: {e} — falling back to Groq")

    # Try Groq
    if groq_available():
        try:
            chain = _get_groq_chain()
            result: GenerationResult = chain.invoke({
                "customer_message": customer_message,
                "intent": intent,
                "evidence_text": _format_evidence(cases),
            })
            return result
        except Exception as e:
            print(f"Groq generation error: {e} — falling back to templates")

    # Template fallback
    return generate_template(customer_message, intent, cases)


# ── Backward-compat aliases ──────────────────────────────────────
def generate_with_gemini(customer_message: str, intent: str, cases: List[Dict] | list) -> GenerationResult:
    """Generate using best available LLM (backward-compat alias)."""
    return generate(customer_message, intent, cases)


# ── Template fallback ─────────────────────────────────────────────
TEMPLATES = {
    "late_delivery": "We're sorry for the delay. Please check your order status for the latest updates. If the package is still delayed, our support team can help investigate further.",
    "missing_package": "We understand your concern. Please verify the delivery address on your order. If the package shows as delivered but you haven't received it, please contact our support team for assistance.",
    "wrong_item": "We apologize for the mix-up. Please contact our support team so we can arrange a replacement or return for the incorrect item.",
    "refund_issue": "We understand your concern about the refund. Please allow the standard processing time. If you haven't received it, please contact our support team for verification.",
    "payment_issue": "We understand this is concerning. Please contact our support team directly so we can verify and resolve the payment issue.",
    "order_cancellation": "We can help with that. Please check your order status to see if cancellation is still possible. If you need further assistance, please contact our support team.",
    "return_item": "You can initiate a return through your Amazon account. Please check the return policy for your item. If you need help, our support team is available.",
    "account_issue": "We understand the urgency. Please contact our support team directly so they can verify your identity and help restore access to your account.",
    "technical_issue": "We're sorry for the inconvenience. Please try clearing your browser cache or restarting the app. If the issue persists, please contact our support team.",
    "product_inquiry": "Thank you for your interest. Product availability can change frequently. Please check the product page for the most current information.",
    "gift_card": "Please check your gift card balance in your Amazon account. If you're having issues redeeming it, our support team can assist.",
    "promotion": "Promotional offers have specific terms and conditions. Please check the promotion details page for validity and applicable terms.",
    "complaint": "We're sorry to hear about your experience. Please contact our support team directly so we can investigate and address your concerns.",
    "general_inquiry": "Thank you for reaching out. Please let us know how we can help you today.",
}

_ESCALATION_INTENTS = {"payment_issue", "refund_issue", "account_issue", "complaint"}


def generate_template(
    customer_message: str, intent: str, cases: List[Dict] | list
) -> GenerationResult:
    """Fallback template-based generation (no LLM)."""
    response = TEMPLATES.get(intent, TEMPLATES["general_inquiry"])
    evidence_ids = [
        c.get("case_id", "") if isinstance(c, dict) else getattr(c, "case_id", "")
        for c in cases[:3]
    ]
    return GenerationResult(
        response=response,
        evidence_case_ids=[cid for cid in evidence_ids if cid],
        escalate=intent in _ESCALATION_INTENTS,
        reason=f"Template response for {intent}",
    )
