"""
Pydantic models for structured data across the project.

These models replace raw dicts for type safety, validation, and
structured LLM output parsing.
"""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


# ── Intents ────────────────────────────────────────────────────────
VALID_INTENTS = [
    "late_delivery",
    "missing_package",
    "wrong_item",
    "refund_issue",
    "payment_issue",
    "order_cancellation",
    "return_item",
    "account_issue",
    "technical_issue",
    "product_inquiry",
    "gift_card",
    "promotion",
    "complaint",
    "general_inquiry",
]


class IntentEnum(str, Enum):
    """Enumeration of all valid intents."""

    LATE_DELIVERY = "late_delivery"
    MISSING_PACKAGE = "missing_package"
    WRONG_ITEM = "wrong_item"
    REFUND_ISSUE = "refund_issue"
    PAYMENT_ISSUE = "payment_issue"
    ORDER_CANCELLATION = "order_cancellation"
    RETURN_ITEM = "return_item"
    ACCOUNT_ISSUE = "account_issue"
    TECHNICAL_ISSUE = "technical_issue"
    PRODUCT_INQUIRY = "product_inquiry"
    GIFT_CARD = "gift_card"
    PROMOTION = "promotion"
    COMPLAINT = "complaint"
    GENERAL_INQUIRY = "general_inquiry"


# ── Classification ─────────────────────────────────────────────────
class ClassificationResult(BaseModel):
    """Structured output from the intent classifier LLM."""

    intent: IntentEnum = Field(description="Predicted customer intent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="", description="One-sentence explanation")


# ── Retrieval ──────────────────────────────────────────────────────
class RetrievedCase(BaseModel):
    """A single case returned by the FAISS retriever."""

    case_id: str = ""
    conversation_id: str = ""
    intent: str = ""
    resolution_quality: str = ""
    turn_count: int = 0
    similarity_score: float = 0.0
    customer_problem: str = ""
    resolution: str = ""
    amazon_responses: str = ""

    model_config = {"coerce_numbers_to_str": True}


class RerankedCase(BaseModel):
    """A retrieved case after intent-aware reranking."""

    case_id: str = ""
    conversation_id: str = ""
    intent: str = ""
    resolution_quality: str = ""
    turn_count: int = 0
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    semantic_score: float = 0.0
    intent_score: float = 0.0
    quality_score: float = 0.0
    customer_problem: str = ""
    resolution: str = ""
    amazon_responses: str = ""

    model_config = {"coerce_numbers_to_str": True}

    @classmethod
    def from_retrieved(cls, case: RetrievedCase | dict, **scores) -> "RerankedCase":
        """Build from a RetrievedCase (or raw dict) plus reranking scores."""
        data = case.model_dump() if isinstance(case, BaseModel) else dict(case)
        data.update(scores)
        return cls(**data)


# ── Routing / Escalation ──────────────────────────────────────────
class RoutingAction(str, Enum):
    """Whether a message should be auto-handled or escalated."""

    AUTO_HANDLE = "auto_handle"
    ESCALATE = "escalate"


class RoutingResult(BaseModel):
    """Decision from the router: auto-handle or escalate."""

    action: RoutingAction
    risk_score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    intent: str = ""
    confidence: float = 0.0
    has_evidence: bool = False


# ── Response Generation ───────────────────────────────────────────
class GenerationResult(BaseModel):
    """Structured output from the response generator LLM."""

    response: str = Field(description="Drafted customer-facing reply")
    evidence_case_ids: List[str] = Field(default_factory=list)
    escalate: bool = Field(default=False)
    reason: str = Field(default="", description="Escalation decision rationale")


# ── LLM Provider ──────────────────────────────────────────────────
class LLMProvider(str, Enum):
    """Supported LLM backends."""

    GEMINI = "gemini"
    GROQ = "groq"
    TEMPLATE = "template"


# ── Convenience ───────────────────────────────────────────────────
def classify_result_from_dict(d: dict) -> ClassificationResult:
    """Safely parse a dict into a ClassificationResult, coercing bad values."""
    intent = d.get("intent", "general_inquiry")
    if intent not in VALID_INTENTS:
        intent = "general_inquiry"
    return ClassificationResult(
        intent=IntentEnum(intent),
        confidence=float(d.get("confidence", 0.3)),
        reasoning=d.get("reasoning", ""),
    )


def routing_result_from_dict(d: dict) -> RoutingResult:
    """Safely parse a dict into a RoutingResult."""
    action = d.get("action", "auto_handle")
    if action not in ("auto_handle", "escalate"):
        action = "auto_handle"
    return RoutingResult(
        action=RoutingAction(action),
        risk_score=float(d.get("risk_score", 0.0)),
        reason=d.get("reason", ""),
        intent=d.get("intent", ""),
        confidence=float(d.get("confidence", 0.0)),
        has_evidence=bool(d.get("has_evidence", False)),
    )


def generation_result_from_dict(d: dict) -> GenerationResult:
    """Safely parse a dict into a GenerationResult."""
    return GenerationResult(
        response=d.get("response", ""),
        evidence_case_ids=d.get("evidence_case_ids", []),
        escalate=bool(d.get("escalate", False)),
        reason=d.get("reason", ""),
    )
