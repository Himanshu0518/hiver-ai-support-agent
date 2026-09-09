"""
LangGraph AI Support Agent Pipeline.

State machine:
  classify → retrieve → rerank → route → generate / escalate
"""
import os
import sys
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.intent.classifier import classify_with_llm, classify_with_keywords
from src.retrieval.reranker import Reranker
from src.generation.response_generator import generate
from src.routing.router import Router
from src.models import (
    ClassificationResult,
    RoutingResult,
    GenerationResult,
    RerankedCase,
    RoutingAction,
)
from src.config import llm_provider_name


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

    try:
        result: ClassificationResult = classify_with_llm(message)
        used_llm = llm_provider_name()
    except Exception:
        result = classify_with_keywords(message)
        used_llm = "keyword"

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
        print(f"Retrieval error: {e}")
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
        }

    result: GenerationResult = generate(
        customer_message=state["customer_message"],
        intent=state["intent"],
        cases=state["reranked_cases"],
    )
    return {
        "response": result.response,
        "evidence_case_ids": result.evidence_case_ids,
        "escalate": result.escalate,
        "generation_reason": result.reason,
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
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS  (LangGraph + Gemini/Groq)")
    print("=" * 60)

    print(f"\nIntent:")
    print(f"  {results.get('intent', '?')}  (confidence: {results.get('confidence', 0):.2f})")
    print(f"  Reasoning: {results.get('classification_reasoning', 'N/A')}")
    print(f"  Provider: {results.get('used_llm', '?')}")

    print(f"\nRetrieved cases:")
    for c in results.get("reranked_cases", [])[:5]:
        cid = c.get("case_id", "?") if isinstance(c, dict) else getattr(c, "case_id", "?")
        score = c.get("rerank_score", 0) if isinstance(c, dict) else getattr(c, "rerank_score", 0)
        qual = c.get("resolution_quality", "?") if isinstance(c, dict) else getattr(c, "resolution_quality", "?")
        print(f"  {cid}  score={score:.3f}  quality={qual}")

    print(f"\nDecision:  {results.get('action', '?').upper()}")
    print(f"  Risk: {results.get('risk_score', 0):.3f}")
    print(f"  Reason: {results.get('routing_reason', 'N/A')}")

    print(f"\nResponse:")
    print(f"  {results.get('response', '')}")
    print()


def main():
    """Interactive CLI."""
    print("=" * 60)
    print("AMAZONHELP AI SUPPORT AGENT  (Gemini / Groq fallback)")
    print("=" * 60)
    print("\nEnter a customer message (or 'quit' to exit):\n")

    while True:
        try:
            message = input("Customer message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not message or message.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
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
