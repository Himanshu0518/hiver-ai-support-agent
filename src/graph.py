"""
LangGraph AI Support Agent Pipeline.

State machine:
  classify → retrieve → rerank → route → generate / escalate
"""
import logging
import os
import sys
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger(__name__)

from src.intent.classifier import classify_with_llm_and_provider, classify_with_keywords
from src.retrieval.reranker import Reranker
from src.generation.response_generator import generate_and_provider
from src.routing.router import Router
from src.models import (
    ClassificationResult,
    RoutingResult,
    GenerationResult,
    RerankedCase,
    RoutingAction,
)


# ================================================================
# 1.  STATE  (TypedDict required by LangGraph)
# ================================================================

class AgentState(TypedDict):
    """Shared state across every node in the graph."""

    # Input
    customer_message: str

    # Classification
    intent: str
    confidence: float
    classification_reasoning: str

    # Retrieval
    retrieved_cases: List[Dict]
    reranked_cases: List[Dict]

    # Routing
    action: str            # "auto_handle" | "escalate"
    risk_score: float
    routing_reason: str

    # Generation
    response: str
    evidence_case_ids: List[str]
    escalate: bool
    generation_reason: str

    # Provider tracking
    used_llm: str          # "gemini" | "groq" | "keyword" | ""


# ================================================================
# 2.  NODES
# ================================================================

def classify_node(state: AgentState) -> dict:
    """Classify the customer message into an intent."""
    message = state["customer_message"]

    result, used_llm = classify_with_llm_and_provider(message)

    return {
        "intent": result.intent.value,
        "confidence": result.confidence,
        "classification_reasoning": result.reasoning,
        "used_llm": used_llm,
    }


def retrieve_node(state: AgentState) -> dict:
    """Retrieve similar historical cases via FAISS."""
    message = state["customer_message"]

    try:
        from src.retrieval.retrieve import Retriever
        retriever = Retriever()
        retriever.load()
        cases = retriever.retrieve(message, top_k=10)
    except Exception as e:
        log.warning("Retrieval error: %s", e)
        cases = []

    return {"retrieved_cases": cases}


def rerank_node(state: AgentState) -> dict:
    """Rerank cases with intent awareness."""
    reranker = Reranker()
    reranked: list[RerankedCase] = reranker.rerank(
        query=state["customer_message"],
        retrieved_cases=state["retrieved_cases"],
        query_intent=state["intent"],
        top_k=5,
    )
    # LangGraph state needs plain dicts
    return {"reranked_cases": [r.model_dump() for r in reranked]}


def route_node(state: AgentState) -> dict:
    """Decide auto_handle vs escalate."""
    router = Router()
    result: RoutingResult = router.route(
        intent=state["intent"],
        confidence=state["confidence"],
        cases=state["reranked_cases"],
        customer_message=state["customer_message"],
    )
    return {
        "action": result.action.value,
        "risk_score": result.risk_score,
        "routing_reason": result.reason,
    }


def generate_node(state: AgentState) -> dict:
    """Generate a grounded response (or skip if escalating)."""
    if state["action"] == RoutingAction.ESCALATE.value:
        return {
            "response": "",
            "evidence_case_ids": [],
            "escalate": True,
            "generation_reason": "Escalated to human agent",
            "used_llm": state.get("used_llm", "keyword"),
        }

    result, gen_provider = generate_and_provider(
        customer_message=state["customer_message"],
        intent=state["intent"],
        cases=state["reranked_cases"],
    )

    return {
        "response": result.response,
        "evidence_case_ids": result.evidence_case_ids,
        "escalate": False,  # Router decided auto_handle — authoritative
        "generation_reason": result.reason,
        "used_llm": gen_provider,
    }


def escalate_node(state: AgentState) -> dict:
    """Format the escalation response."""
    return {
        "response": (
            f"This message has been escalated to a human agent.\n"
            f"Reason: {state['routing_reason']}\n"
            f"Intent: {state['intent']} (confidence: {state['confidence']:.2f})"
        ),
        "escalate": True,
    }


# ================================================================
# 3.  ROUTING FUNCTION
# ================================================================

def should_generate_or_escalate(state: AgentState) -> str:
    """Branch after route_node."""
    if state["action"] == RoutingAction.ESCALATE.value:
        return "escalate"
    return "generate"


# ================================================================
# 4.  GRAPH
# ================================================================

def build_graph() -> StateGraph:
    """Build the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("route", route_node)
    graph.add_node("generate", generate_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "route")

    graph.add_conditional_edges(
        "route",
        should_generate_or_escalate,
        {
            "generate": "generate",
            "escalate": "escalate",
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# ================================================================
# 5.  PUBLIC API
# ================================================================

_app = None


def get_app():
    """Get or build the compiled graph."""
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_pipeline(customer_message: str) -> dict:
    """Run the full pipeline on a customer message."""
    app = get_app()
    result = app.invoke({"customer_message": customer_message})
    return result


# ================================================================
# 6.  CLI
# ================================================================

def display_results(results: dict):
    """Pretty-print pipeline results."""
    import json

    log.info("=" * 60)
    log.info("PIPELINE RESULTS  (LangGraph + Gemini/Groq)")
    log.info("=" * 60)

    log.info("\nIntent:")
    log.info("  %s  (confidence: %.2f)", results.get('intent', '?'), results.get('confidence', 0))
    log.info("  Reasoning: %s", results.get('classification_reasoning', 'N/A'))
    log.info("  Provider: %s", results.get('used_llm', '?'))

    log.info("\nRetrieved cases:")
    for c in results.get("reranked_cases", [])[:5]:
        cid = c.get("case_id", "?") if isinstance(c, dict) else getattr(c, "case_id", "?")
        score = c.get("rerank_score", 0) if isinstance(c, dict) else getattr(c, "rerank_score", 0)
        qual = c.get("resolution_quality", "?") if isinstance(c, dict) else getattr(c, "resolution_quality", "?")
        log.info("  %s  score=%.3f  quality=%s", cid, score, qual)

    log.info("\nDecision:  %s", results.get('action', '?').upper())
    log.info("  Risk: %.3f", results.get('risk_score', 0))
    log.info("  Reason: %s", results.get('routing_reason', 'N/A'))

    log.info("\nResponse:")
    log.info("  %s", results.get('response', ''))

    # Build structured JSON output
    json_output = {
        "customer_message": results.get("customer_message", ""),
        "intent": results.get("intent", ""),
        "confidence": round(results.get("confidence", 0), 4),
        "classification_reasoning": results.get("classification_reasoning", ""),
        "action": results.get("action", ""),
        "risk_score": round(results.get("risk_score", 0), 4),
        "routing_reason": results.get("routing_reason", ""),
        "escalate": results.get("escalate", False),
        "response": results.get("response", ""),
        "evidence_case_ids": results.get("evidence_case_ids", []),
        "used_llm": results.get("used_llm", ""),
    }
    log.info("\n" + "=" * 60)
    log.info("STRUCTURED JSON RESPONSE")
    log.info("=" * 60)
    log.info("\n%s", json.dumps(json_output, indent=2, ensure_ascii=False))
    log.info("")


def main():
    """Interactive CLI."""
    log.info("=" * 60)
    log.info("AMAZONHELP AI SUPPORT AGENT  (Gemini / Groq fallback)")
    log.info("=" * 60)
    log.info("\nEnter a customer message (or 'quit' to exit):\n")

    while True:
        try:
            message = input("Customer message: ").strip()
        except (EOFError, KeyboardInterrupt):
            log.info("\nGoodbye!")
            break

        if not message or message.lower() in ("quit", "exit", "q"):
            log.info("Goodbye!")
            break

        results = run_pipeline(message)
        display_results(results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        results = run_pipeline(message)
        display_results(results)
    else:
        main()
