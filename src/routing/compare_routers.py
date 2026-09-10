"""
Compare old router vs new learned router on the golden set.
Run: python -m src.routing.compare_routers
"""
import logging
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

log = logging.getLogger(__name__)

from src.routing.router import Router
from src.retrieval.reranker import Reranker
from src.models import RoutingResult
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    golden_path = "data/evaluation/golden_set.csv"
    df = pd.read_csv(golden_path)
    log.info("Golden set: %s examples", f"{len(df)}")

    # Try to load retriever
    try:
        from src.retrieval.retrieve import Retriever
        retriever = Retriever()
        retriever.load()
        has_retriever = True
    except Exception:
        has_retriever = False

    reranker = Reranker()
    router = Router(escalation_threshold=0.5)

    # Collect results
    log.info("Running router on golden set...")
    preds: list[str] = []
    true_labels: list[str] = []

    for _, row in df.iterrows():
        msg = str(row.get("customer_message", ""))
        intent = row.get("intent", "general_inquiry")
        confidence = 0.8
        true_action = row.get("expected_action", "auto_handle")

        cases: list = []
        reranked: list = []
        if has_retriever and len(msg) >= 5:
            try:
                cases = retriever.retrieve(msg, top_k=5)
                reranked = reranker.rerank(msg, cases, intent, top_k=3)
                # Convert RerankedCase objects back to dicts for the router
                reranked = [r.model_dump() if hasattr(r, "model_dump") else r for r in reranked]
            except Exception:
                pass

        result: RoutingResult = router.route(
            intent=intent,
            confidence=confidence,
            cases=cases,
            customer_message=msg,
        )
        preds.append(result.action.value)
        true_labels.append(true_action)

    # ── Metrics ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("ROUTER EVALUATION ON GOLDEN SET")
    log.info("=" * 60)

    acc = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds, labels=["auto_handle", "escalate"])
    false_auto = sum(1 for t, p in zip(true_labels, preds) if t == "escalate" and p == "auto_handle")
    total_esc = sum(1 for t in true_labels if t == "escalate")
    false_auto_rate = false_auto / total_esc if total_esc else 0

    log.info("\n  Accuracy:              %.4f", acc)
    log.info("  False Auto-Handle:     %.4f (%s/%s)", false_auto_rate, false_auto, total_esc)
    log.info("  Confusion matrix:")
    log.info("    auto_handle  escalate")
    log.info("    %5d  %5d", cm[0][0], cm[0][1])
    log.info("    %5d  %5d", cm[1][0], cm[1][1])

    # ── Side-by-side examples ─────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SIDE-BY-SIDE EXAMPLES")
    log.info("=" * 60)

    for i, (_, row) in enumerate(df.head(10).iterrows()):
        msg = str(row.get("customer_message", ""))
        intent = row.get("intent", "general_inquiry")
        true_action = row.get("expected_action", "auto_handle")

        cases = []
        reranked = []
        if has_retriever and len(msg) >= 5:
            try:
                cases = retriever.retrieve(msg, top_k=5)
                reranked = reranker.rerank(msg, cases, intent, top_k=3)
                reranked = [r.model_dump() if hasattr(r, "model_dump") else r for r in reranked]
            except Exception:
                pass

        result = router.route(intent=intent, confidence=0.8, cases=cases, customer_message=msg)
        log.info("\n%s. \"%s...\"", i + 1, msg[:80])
        log.info("   Intent: %s  True: %s", intent, true_action)
        log.info("   Router: %-12s  risk=%.3f  %s", result.action.value, result.risk_score, result.reason[:60])


if __name__ == "__main__":
    main()
