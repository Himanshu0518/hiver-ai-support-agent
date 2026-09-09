"""
Routing and Escalation Module.
Decides whether to auto-handle or escalate to a human agent.
"""
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models import RoutingAction, RoutingResult


# ── Risk configuration ────────────────────────────────────────────
INTENT_RISK_SCORES: dict[str, float] = {
    "payment_issue": 0.9,
    "refund_issue": 0.8,
    "account_issue": 0.85,
    "complaint": 0.7,
    "wrong_item": 0.5,
    "return_item": 0.4,
    "order_cancellation": 0.3,
    "late_delivery": 0.2,
    "missing_package": 0.3,
    "product_inquiry": 0.1,
    "gift_card": 0.2,
    "promotion": 0.1,
    "technical_issue": 0.3,
    "general_inquiry": 0.1,
}

HIGH_RISK_INTENTS = {"payment_issue", "refund_issue", "account_issue", "complaint"}

ACCOUNT_KEYWORDS = [
    "my account", "my order number", "my email", "my phone",
    "my address", "my password", "locked out", "cannot access",
]


class Router:
    """Routes messages to auto-handle or escalation."""

    def __init__(self, escalation_threshold: float = 0.5):
        self.escalation_threshold = escalation_threshold

    def compute_risk_score(
        self,
        intent: str,
        confidence: float,
        cases: list,
        response_quality: float = 0.5,
    ) -> float:
        """Compute a 0-1 risk score."""
        intent_risk = INTENT_RISK_SCORES.get(intent, 0.5)
        confidence_risk = 1.0 - confidence

        if not cases:
            retrieval_risk = 0.8
        else:
            avg_sim = sum(
                c.get("rerank_score", c.get("similarity_score", 0)) if isinstance(c, dict)
                else getattr(c, "rerank_score", getattr(c, "similarity_score", 0))
                for c in cases
            ) / len(cases)
            retrieval_risk = 1.0 - min(avg_sim, 1.0)

        risk = (
            0.40 * intent_risk
            + 0.20 * confidence_risk
            + 0.25 * retrieval_risk
            + 0.15 * 0.0  # account_risk placeholder
        )
        return min(risk, 1.0)

    def route(
        self,
        intent: str,
        confidence: float,
        cases: list,
        customer_message: str = "",
        response_quality: float = 0.5,
        reranked_cases: list | None = None,
    ) -> RoutingResult:
        """Make the auto_handle / escalate decision."""
        # Account-specific language check
        account_specific = False
        if customer_message:
            msg_lower = customer_message.lower()
            account_specific = any(kw in msg_lower for kw in ACCOUNT_KEYWORDS)

        risk_score = self.compute_risk_score(intent, confidence, cases, response_quality)
        if account_specific:
            risk_score = min(risk_score + 0.2, 1.0)

        escalate = risk_score >= self.escalation_threshold

        # Build reasons
        reasons: list[str] = []
        if intent in HIGH_RISK_INTENTS:
            reasons.append(f"High-risk intent: {intent}")
        if confidence < 0.5:
            reasons.append(f"Low classification confidence: {confidence:.2f}")
        if not cases:
            reasons.append("No relevant historical evidence found")
        elif response_quality < 0.3:
            reasons.append("Poor quality of retrieved evidence")
        if account_specific:
            reasons.append("Message contains account-specific information")
        if not reasons:
            tag = "exceeds" if escalate else "below"
            reasons.append(f"Combined risk score ({risk_score:.2f}) {tag} threshold")

        return RoutingResult(
            action=RoutingAction.ESCALATE if escalate else RoutingAction.AUTO_HANDLE,
            risk_score=risk_score,
            reason="; ".join(reasons),
            intent=intent,
            confidence=confidence,
            has_evidence=len(cases) > 0,
        )
