"""
Build processed dataset: PII sanitization + quality filtering.
Run: python -m src.preprocessing.build_processed
"""
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.pii import sanitize_text
from src.preprocessing.quality import filter_conversations


def main():
    print("=" * 60)
    print("BUILDING PROCESSED DATASET")
    print("=" * 60)
    
    # Load raw conversations
    raw_path = "data/raw/amazonhelp_conversations.csv"
    print(f"\nLoading raw conversations from {raw_path}...")
    conversations = pd.read_csv(raw_path, low_memory=False)
    print(f"Loaded {len(conversations):,} conversations")
    
    # Step 1: PII Sanitization
    print("\n--- Step 1: PII Sanitization ---")
    text_cols = ['conversation', 'customer_problem', 'amazon_responses']
    for col in text_cols:
        if col in conversations.columns:
            print(f"  Sanitizing {col}...")
            conversations[col] = conversations[col].apply(sanitize_text)
    print("  PII sanitization complete.")
    
    # Step 2: Quality Filtering
    print("\n--- Step 2: Quality Filtering ---")
    filtered, stats = filter_conversations(conversations)
    print(f"  Total: {stats['total']:,}")
    print(f"  Kept: {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"  Removed: {stats['removed']:,}")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count:,}")
    
    # Save processed conversations
    output_path = "data/processed/filtered_conversations.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"\nSaved {len(filtered):,} filtered conversations to {output_path}")
    
    # Also save a sampled subset for faster development
    sample_size = min(5000, len(filtered))
    sampled = filtered.sample(n=sample_size, random_state=42)
    sample_path = "data/processed/filtered_conversations_sample.csv"
    sampled.to_csv(sample_path, index=False)
    print(f"Saved {sample_size:,} sampled conversations to {sample_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
