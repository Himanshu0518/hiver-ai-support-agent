"""
Full Evaluation Harness — Gemini/Groq + LangGraph powered.
Run: python -m src.evaluation.run
"""
import logging
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

log = logging.getLogger(__name__)


# ================================================================
#  Intent Classification
# ================================================================

def evaluate_intent_classification(golden_path: str):
    """Compare majority, keyword, and LLM classifiers."""
    log.info("\n" + "=" * 60)
    log.info("INTENT CLASSIFICATION")
    log.info("=" * 60)

    golden = pd.read_csv(golden_path)
    X = golden["customer_message"].fillna("").astype(str).values
    y_true = golden["intent"].values
    results = {}

    # --- Majority baseline ---
    log.info("\n--- Majority Class Baseline ---")
    maj = MajorityClassifier()
    maj.fit(X, y_true)
    y_pred = maj.predict(X)
    results["majority"] = _metrics(y_true, y_pred, "majority")

    # --- Keyword ---
    log.info("\n--- Keyword Classifier ---")
    y_pred = [classify_with_keywords(m).intent.value for m in X]
    results["keyword"] = _metrics(y_true, y_pred, "keyword")

    # --- LLM (Gemini or Groq) ---
    provider = llm_provider_name()
    log.info("\n--- LLM Classifier (%s) ---", provider)
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
    log.info("  Accuracy:  %.4f", acc)
    log.info("  Macro F1:  %.4f", f1m)
    log.info("  Weighted:  %.4f", f1w)
    log.info("%s", classification_report(y_true, y_pred, zero_division=0))
    return {"accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w}


# ================================================================
#  Retrieval
# ================================================================

def evaluate_retrieval(golden_path: str):
    log.info("\n" + "=" * 60)
    log.info("RETRIEVAL")
    log.info("=" * 60)

    golden = pd.read_csv(golden_path)

    try:
        from src.retrieval.retrieve import Retriever
        retriever = Retriever()
        retriever.load()
    except Exception as e:
        log.warning("Retriever unavailable: %s", e)
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

    log.info("  Evaluated %s queries", n)
    log.info("  Recall@1: %.4f", r1)
    log.info("  Recall@3: %.4f", r3)
    log.info("  Recall@5: %.4f", r5)
    log.info("  MRR:      %.4f", mrr)
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

# Scoring guidelines shown to human annotators
HUMAN_LABELING_GUIDE = """
SCORING GUIDELINES (1-5 scale for each dimension):

1. groundedness (is the response supported by the evidence?):
   5 = Fully grounded: every claim traces to evidence cases
   4 = Mostly grounded: 1-2 minor unsupported details
   3 = Partially grounded: core advice is supported but some parts are generic
   2 = Weakly grounded: response ignores evidence, uses only generic templates
   1 = Not grounded: contradicts evidence or fabricates specifics

2. correctness (does it address the customer's issue?):
   5 = Perfectly addresses the stated issue
   4 = Addresses issue with minor irrelevant content
   3 = Partially addresses; misses a sub-issue
   2 = Barely addresses; mostly off-topic
   1 = Completely misses the issue

3. helpfulness (does the customer know what to do next?):
   5 = Clear, actionable next steps
   4 = Actionable but missing one step
   3 = Vague but directionally useful
   2 = No clear next step
   1 = No help at all

4. safety (does it avoid unsupported promises?):
   5 = Zero promises; explicitly says what it cannot do
   4 = Safe but could be more explicit about limits
   3 = Mostly safe; one slightly overpromising phrase
   2 = Makes one unsupported promise (refund, date, guarantee)
   1 = Makes multiple unsupported promises

5. escalation (was escalation recommended when necessary?):
   Score for whether escalation was handled correctly:
   5 = Correctly escalated high-risk issue OR correctly auto-handled low-risk
   4 = Correct decision with weak justification
   3 = Correct decision but no justification
   2 = Should have escalated but auto-handled (or vice versa)
   1 = Dangerously wrong decision
"""

# Human + LLM judge comparison prompt (for agreement study)
HUMAN_JUDGE_PROMPT = """You are helping evaluate a customer support AI system.

Below is a CUSTOMER MESSAGE, the AI'S GENERATED RESPONSE, and the HISTORICAL EVIDENCE
the AI used. Your job is to score the response on 5 dimensions.

CUSTOMER MESSAGE:
{message}

INTENT: {intent}

AI GENERATED RESPONSE:
{response}

HISTORICAL EVIDENCE (what the AI saw):
{evidence}

"{HUMAN_LABELING_GUIDE}"

Please score each dimension from 1 to 5. Return ONLY a JSON object:
{{"groundedness": N, "correctness": N, "helpfulness": N, "safety": N, "escalation": N, "notes": "short note about your scoring"}}"""


def evaluate_response_quality(golden_path: str, n_samples: int = 30):
    """Evaluate response quality with LLM-as-judge."""
    log.info("\n" + "=" * 60)
    log.info("RESPONSE QUALITY  (LLM-as-judge)")
    log.info("=" * 60)

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
            log.warning("  Judge error: %s", e)

    results = {dim: np.mean(vals) if vals else 0 for dim, vals in scores.items()}
    for dim, val in results.items():
        log.info("  %-15s: %.2f", dim, val)
    log.info("  (%s/%s judged)", ok, len(sample))
    return results


# ================================================================
#  Routing
# ================================================================

def evaluate_routing(golden_path: str):
    log.info("\n" + "=" * 60)
    log.info("ROUTING")
    log.info("=" * 60)

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

    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(true_actions, pred_actions)

    false_auto = sum(1 for t, p in zip(true_actions, pred_actions) if t == "escalate" and p == "auto_handle")
    total_esc = sum(1 for t in true_actions if t == "escalate")
    false_auto_rate = false_auto / total_esc if total_esc else 0

    log.info("  Accuracy:              %.4f", acc)
    log.info("  False Auto-Handle Rate: %.4f", false_auto_rate)
    log.info("  Cohen's Kappa:         %.4f", kappa)
    log.info("  Confusion matrix:")
    log.info("    auto_handle  escalate")
    log.info("    %5d  %5d", cm[0][0], cm[0][1])
    log.info("    %5d  %5d", cm[1][0], cm[1][1])

    return {"accuracy": acc, "false_auto_handle_rate": false_auto_rate, "kappa": kappa}


# ================================================================
#  Full pipeline (LangGraph end-to-end)
# ================================================================

def evaluate_full_pipeline(golden_path: str, n_samples: int = 20):
    """Run full LangGraph pipeline on a sample and show examples."""
    log.info("\n" + "=" * 60)
    log.info("FULL PIPELINE (LangGraph) — sample runs")
    log.info("=" * 60)

    golden = pd.read_csv(golden_path)
    sample = golden.sample(n=min(n_samples, len(golden)), random_state=123)
    app = get_app()

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        msg = str(row.get("customer_message", ""))
        if len(msg) < 5:
            continue
        log.info("\n--- Example %s ---", i)
        log.info("  Message: %s...", msg[:100])
        result = app.invoke({"customer_message": msg})
        log.info("  Intent:    %s", result.get('intent'))
        log.info("  Action:    %s", result.get('action'))
        log.info("  Provider:  %s", result.get('used_llm', '?'))
        log.info("  Response:  %s...", str(result.get('response', ''))[:120])


# ================================================================
#  Main
# ================================================================

def main():
    log.info("=" * 60)
    log.info("  AMAZONHELP AI SUPPORT AGENT — FULL EVALUATION")
    log.info("  (Provider: %s + LangGraph)", llm_provider_name())
    log.info("=" * 60)

    golden_path = "data/evaluation/golden_set.csv"
    if not os.path.exists(golden_path):
        log.error("Golden set not found at %s", golden_path)
        return

    t0 = time.time()

    intent_res, _, _ = evaluate_intent_classification(golden_path)
    retrieval_res = evaluate_retrieval(golden_path)
    response_res = evaluate_response_quality(golden_path)
    routing_res = evaluate_routing(golden_path)
    evaluate_full_pipeline(golden_path)

    elapsed = time.time() - t0

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    provider = llm_provider_name()
    log.info("  Provider:            %s", provider)
    log.info("  Intent (%s):   Acc=%.4f  F1=%.4f", provider, intent_res['llm']['accuracy'], intent_res['llm']['macro_f1'])
    log.info("  Retrieval:           MRR=%.4f  R@5=%.4f", retrieval_res['mrr'], retrieval_res['recall_at_5'])
    log.info("  Response:            Grounded=%.2f  Safe=%.2f", response_res.get('groundedness', 0), response_res.get('safety', 0))
    log.info("  Routing:             Acc=%.4f  FalseAuto=%.4f  Kappa=%.4f", routing_res['accuracy'], routing_res['false_auto_handle_rate'], routing_res.get('kappa', 0))
    log.info("  Time: %.1fs", elapsed)

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
    log.info("\nSaved to %s", out)


if __name__ == "__main__":
    main()
