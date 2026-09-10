"""
Pipeline CLI — thin wrapper around the LangGraph graph.
Run: python -m src.pipeline "My package is late"
"""
import logging
import sys
from src.graph import run_pipeline, display_results

log = logging.getLogger(__name__)


def main():
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        results = run_pipeline(message)
        display_results(results)
    else:
        log.info("Usage: python -m src.pipeline \"<customer message>\"")
        log.info("  or run: python -m src.graph   (interactive mode)")


if __name__ == "__main__":
    main()
