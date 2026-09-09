"""
Reranker Module.
Reranks retrieved cases using intent awareness and quality signals.
"""
from typing import List

from src.models import RerankedCase


# Intent similarity groups (partial matches score 0.5)
_INTENT_GROUPS: list[set[str]] = [
    {"late_delivery", "missing_package", "wrong_item"},
    {"refund_issue", "payment_issue"},
    {"order_cancellation", "return_item"},
    {"account_issue", "technical_issue"},
]

# Resolution quality → numeric score
_QUALITY_SCORES: dict[str, float] = {
    "strong": 1.0,
    "medium": 0.6,
    "weak": 0.3,
}


class Reranker:
    """Reranks retrieved cases with intent awareness."""

    def __init__(
        self,
        semantic_weight: float = 0.60,
        intent_weight: float = 0.25,
        quality_weight: float = 0.15,
    ):
        self.semantic_weight = semantic_weight
        self.intent_weight = intent_weight
        self.quality_weight = quality_weight

    def _intent_score(self, query_intent: str, case_intent: str) -> float:
        """1.0 for exact match, 0.5 for same group, 0.0 otherwise."""
        if query_intent == case_intent:
            return 1.0
        for group in _INTENT_GROUPS:
            if query_intent in group and case_intent in group:
                return 0.5
        return 0.0

    def rerank(
        self,
        query: str,
        retrieved_cases: List,
        query_intent: str,
        top_k: int = 5,
    ) -> list[RerankedCase]:
        """Rerank *retrieved_cases* and return the top *top_k*."""
        if not retrieved_cases:
            return []

        reranked: list[RerankedCase] = []
        for case in retrieved_cases:
            # Accept both Pydantic models and raw dicts
            if isinstance(case, dict):
                c = case
            elif isinstance(case, RerankedCase):
                c = case.model_dump()
            else:
                c = case.model_dump()

            semantic_score = float(c.get("similarity_score", 0.0))
            intent_sc = self._intent_score(query_intent, c.get("intent", ""))
            quality_sc = _QUALITY_SCORES.get(c.get("resolution_quality", "weak"), 0.3)

            final = (
                self.semantic_weight * semantic_score
                + self.intent_weight * intent_sc
                + self.quality_weight * quality_sc
            )

            reranked.append(RerankedCase(
                case_id=c.get("case_id", ""),
                conversation_id=c.get("conversation_id", ""),
                intent=c.get("intent", ""),
                resolution_quality=c.get("resolution_quality", ""),
                turn_count=int(c.get("turn_count", 0)),
                similarity_score=semantic_score,
                rerank_score=final,
                semantic_score=semantic_score,
                intent_score=intent_sc,
                quality_score=quality_sc,
                customer_problem=c.get("customer_problem", ""),
                resolution=c.get("resolution", ""),
                amazon_responses=c.get("amazon_responses", ""),
            ))

        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_k]
