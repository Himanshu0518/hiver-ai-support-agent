"""
Full Evaluation Harness — Gemini/Groq + LangGraph powered.
Run: python -m src.evaluation.run
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.intent.classifier import classify_with_llm, classify_with_keywords
from src.intent.baseline import MajorityClassifier
from src.retrieval.reranker import Reranker
from src.generation.response_generator import generate
from src.routing.router import Router
from src.graph import get_app
from src.models import (
    ClassificationResult,
    RoutingResult,
    GenerationResult,
    RoutingAction,
)
from src.config import llm_provider_name


# ================================================================
#  Intent Classification
# ================================================================

def evaluate_intent_classification(golden_path: str):
    """Compare majority, keyword, and LLM classifiers."""
    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION")
    print("=" * 60)

    golden = pd.read_csv(golden_path)
    X = golden["customer_message"].fillna("").astype(str).values
    y_true = golden["intent"].values
    results = {}

    # --- Majority baseline ---
    print("\n--- Majority Class Baseline ---")
    maj = MajorityClassifier()
    maj.fit(X, y_true)
    y_pred = maj.predict(X)
    results["majority"] = _metrics(y_true, y_pred, "majority")

    # --- Keyword ---
    print("\n--- Keyword Classifier ---")
    y_pred = [classify_with_keywords(m).intent.value for m in X]
    results["keyword"] = _metrics(y_true, y_pred, "keyword")

    # --- LLM (Gemini or Groq) ---
    provider = llm_provider_name()
    print(f"\n--- LLM Classifier ({provider}) ---")
    y_pred = []
    for m in X:
        r: ClassificationResult = classify_with_llm(m)
        y_pred.append(r.intent.value)
    results["llm"] = _metrics(y_true, y_pred, provider)

    return results, y_true, y_pred


def _metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1m:.4f}")
    print(f"  Weighted:  {f1w:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w}


# ================================================================
#  Retrieval
# ================================================================

def evaluate_retrieval(golden_path: str):
    print("\n" + "=" * 60)
    print("RETRIEVAL")
    print("=" * 60)

    golden = pd.read_csv(golden_path)

    try:
        from src.retrieval.retrieve import Retriever
        retriever = Retriever()
        retriever.load()
    except Exception as e:
        print(f"Retriever unavailable: {e}")
        return {"recall_at_5": 0, "mrr": 0}

    reranker = Reranker()
    r1 = r3 = r5 = mrr = n = 0

    for _, row in golden.iterrows():
        q = str(row.get("customer_message", ""))
        qi = row.get("intent", "general_inquiry")
        if len(q) < 5:
            continue

        cases = retriever.retrieve(q, top_k=10)
        reranked = reranker.rerank(q, cases, qi, top_k=5)
        intents = [c.intent for c in reranked]

        if qi in intents:
            rank = intents.index(qi) + 1
            r1 += rank <= 1
            r3 += rank <= 3
            r5 += rank <= 5
            mrr += 1.0 / rank
        n += 1

    if n:
        r1 /= n; r3 /= n; r5 /= n; mrr /= n

    print(f"  Evaluated {n} queries")
    print(f"  Recall@1: {r1:.4f}")
    print(f"  Recall@3: {r3:.4f}")
    print(f"  Recall@5: {r5:.4f}")
    print(f"  MRR:      {mrr:.4f}")
    return {"recall_at_1": r1, "recall_at_3": r3, "recall_at_5": r5, "mrr": mrr}


# ================================================================
#  Response quality  (LLM-as-judge)
# ================================================================

JUDGE_PROMPT = """Rate this customer support response on 5 dimensions (1-5 each).

CUSTOMER MESSAGE: {message}
INTENT: {intent}
GENERATED RESPONSE: {response}
EVIDENCE USED: {evidence}

Score each dimension 1-5:
1. groundedness   — is the response supported by the evidence?
2. correctness    — does it address the customer's issue?
3. helpfulness    — does the customer know what to do next?
4. safety         — does it avoid unsupported promises?
5. escalation     — was escalation recommended when necessary?

Return ONLY a JSON object:
{{"groundedness": N, "correctness": N, "helpfulness": N, "safety": N, "escalation": N}}"""


def evaluate_response_quality(golden_path: str, n_samples: int = 30):
    """Evaluate response quality with LLM-as-judge."""
    print("\n" + "=" * 60)
    print("RESPONSE QUALITY  (LLM-as-judge)")
    print("=" * 60)

    golden = pd.read_csv(golden_path)
    sample = golden.sample(n=min(n_samples, len(golden)), random_state=42)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from src.config import get_best_available_llm

    judge_chain = ChatPromptTemplate.from_messages([
        ("system", "You are a strict evaluation judge. Return only valid JSON."),
        ("user", JUDGE_PROMPT),
    ]) | get_best_available_llm(temperature=0.0, max_output_tokens=300) | JsonOutputParser()

    scores = {"groundedness": [], "correctness": [], "helpfulness": [], "safety": [], "escalation": []}
    ok = 0

    for _, row in sample.iterrows():
        msg = str(row.get("customer_message", ""))
        intent = row.get("intent", "general_inquiry")
        if len(msg) < 5:
            continue

        try:
            from src.retrieval.retrieve import Retriever
            retriever = Retriever()
            retriever.load()
            cases = retriever.retrieve(msg, top_k=5)
        except Exception:
            cases = []

        reranker = Reranker()
        reranked = reranker.rerank(msg, cases, intent, top_k=3)
        reranked_dicts = [r.model_dump() for r in reranked]

        gen: GenerationResult = generate(msg, intent, reranked_dicts)

        try:
            judge = judge_chain.invoke({
                "message": msg, "intent": intent,
                "response": gen.response,
                "evidence": json.dumps(reranked_dicts[:2], default=str)[:500],
            })
            for dim in scores:
                scores[dim].append(float(judge.get(dim, 3)))
            ok += 1
        except Exception as e:
            print(f"  Judge error: {e}")

    results = {dim: np.mean(vals) if vals else 0 for dim, vals in scores.items()}
    for dim, val in results.items():
        print(f"  {dim:15s}: {val:.2f}")
    print(f"  ({ok}/{len(sample)} judged)")
    return results


# ================================================================
#  Routing
# ================================================================

def evaluate_routing(golden_path: str):
    print("\n" + "=" * 60)
    print("ROUTING")
    print("=" * 60)

    golden = pd.read_csv(golden_path)
    router = Router()

    true_actions = golden["expected_action"].values
    pred_actions = []
    for _, row in golden.iterrows():
        r: RoutingResult = router.route(
            intent=row.get("intent", "general_inquiry"),
            confidence=0.8,
            cases=[],
            customer_message=str(row.get("customer_message", "")),
        )
        pred_actions.append(r.action.value)

    acc = accuracy_score(true_actions, pred_actions)
    cm = confusion_matrix(true_actions, pred_actions, labels=["auto_handle", "escalate"])

    false_auto = sum(1 for t, p in zip(true_actions, pred_actions) if t == "escalate" and p == "auto_handle")
    total_esc = sum(1 for t in true_actions if t == "escalate")
    false_auto_rate = false_auto / total_esc if total_esc else 0

    print(f"  Accuracy:              {acc:.4f}")
    print(f"  False Auto-Handle Rate: {false_auto_rate:.4f}")
    print(f"  Confusion matrix:")
    print(f"    auto_handle  escalate")
    print(f"    {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"    {cm[1][0]:>5}  {cm[1][1]:>5}")

    return {"accuracy": acc, "false_auto_handle_rate": false_auto_rate}


# ================================================================
#  Full pipeline (LangGraph end-to-end)
# ================================================================

def evaluate_full_pipeline(golden_path: str, n_samples: int = 20):
    """Run full LangGraph pipeline on a sample and show examples."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE (LangGraph) — sample runs")
    print("=" * 60)

    golden = pd.read_csv(golden_path)
    sample = golden.sample(n=min(n_samples, len(golden)), random_state=123)
    app = get_app()

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        msg = str(row.get("customer_message", ""))
        if len(msg) < 5:
            continue
        print(f"\n--- Example {i} ---")
        print(f"  Message: {msg[:100]}...")
        result = app.invoke({"customer_message": msg})
        print(f"  Intent:    {result.get('intent')}")
        print(f"  Action:    {result.get('action')}")
        print(f"  Provider:  {result.get('used_llm', '?')}")
        print(f"  Response:  {str(result.get('response', ''))[:120]}...")


# ================================================================
#  Main
# ================================================================

def main():
    print("=" * 60)
    print("  AMAZONHELP AI SUPPORT AGENT — FULL EVALUATION")
    print(f"  (Provider: {llm_provider_name()} + LangGraph)")
    print("=" * 60)

    golden_path = "data/evaluation/golden_set.csv"
    if not os.path.exists(golden_path):
        print(f"Golden set not found at {golden_path}")
        return

    t0 = time.time()

    intent_res, _, _ = evaluate_intent_classification(golden_path)
    retrieval_res = evaluate_retrieval(golden_path)
    response_res = evaluate_response_quality(golden_path)
    routing_res = evaluate_routing(golden_path)
    evaluate_full_pipeline(golden_path)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    provider = llm_provider_name()
    print(f"  Provider:            {provider}")
    print(f"  Intent ({provider}):   Acc={intent_res['llm']['accuracy']:.4f}  F1={intent_res['llm']['macro_f1']:.4f}")
    print(f"  Retrieval:           MRR={retrieval_res['mrr']:.4f}  R@5={retrieval_res['recall_at_5']:.4f}")
    print(f"  Response:            Grounded={response_res.get('groundedness', 0):.2f}  Safe={response_res.get('safety', 0):.2f}")
    print(f"  Routing:             Acc={routing_res['accuracy']:.4f}  FalseAuto={routing_res['false_auto_handle_rate']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    all_results = {
        "provider": provider,
        "intent_classification": intent_res,
        "retrieval": retrieval_res,
        "response_quality": response_res,
        "routing": routing_res,
        "elapsed_seconds": elapsed,
    }
    out = "artifacts/evaluation_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
