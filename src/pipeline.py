"""
Pipeline CLI — thin wrapper around the LangGraph graph.
Run: python -m src.pipeline "My package is late"
"""
import sys
from src.graph import run_pipeline, display_results


def main():
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        results = run_pipeline(message)
        display_results(results)
    else:
        print("Usage: python -m src.pipeline \"<customer message>\"")
        print("  or run: python -m src.graph   (interactive mode)")


if __name__ == "__main__":
    main()
