"""
Build processed dataset: PII sanitization + quality filtering.
Run: python -m src.preprocessing.build_processed
"""
import logging
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

log = logging.getLogger(__name__)

from src.preprocessing.pii import sanitize_text
from src.preprocessing.quality import filter_conversations


def main():
    log.info("=" * 60)
    log.info("BUILDING PROCESSED DATASET")
    log.info("=" * 60)

    # Load raw conversations
    raw_path = "data/raw/amazonhelp_conversations.csv"
    log.info("\nLoading raw conversations from %s...", raw_path)
    conversations = pd.read_csv(raw_path, low_memory=False)
    log.info("Loaded %s conversations", f"{len(conversations):,}")

    # Step 1: PII Sanitization
    log.info("\n--- Step 1: PII Sanitization ---")
    text_cols = ['conversation', 'customer_problem', 'amazon_responses']
    for col in text_cols:
        if col in conversations.columns:
            log.info("  Sanitizing %s...", col)
            conversations[col] = conversations[col].apply(sanitize_text)
    log.info("  PII sanitization complete.")

    # Step 2: Quality Filtering
    log.info("\n--- Step 2: Quality Filtering ---")
    filtered, stats = filter_conversations(conversations)
    log.info("  Total: %s", f"{stats['total']:,}")
    log.info("  Kept: %s (%.1f%%)", f"{stats['kept']:,}", stats['kept']/stats['total']*100)
    log.info("  Removed: %s", f"{stats['removed']:,}")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
        log.info("    %s: %s", reason, f"{count:,}")

    # Save processed conversations
    output_path = "data/processed/filtered_conversations.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered.to_csv(output_path, index=False)
    log.info("\nSaved %s filtered conversations to %s", f"{len(filtered):,}", output_path)

    # Also save a sampled subset for faster development
    sample_size = min(5000, len(filtered))
    sampled = filtered.sample(n=sample_size, random_state=42)
    sample_path = "data/processed/filtered_conversations_sample.csv"
    sampled.to_csv(sample_path, index=False)
    log.info("Saved %s sampled conversations to %s", f"{sample_size:,}", sample_path)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
