"""
Intent Taxonomy Builder.
Explores customer problems to define intent categories from data.
Run: python -m src.intent.taxonomy
"""
import pandas as pd
import re
import os
import sys
import json
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Keyword patterns for initial intent discovery
INTENT_KEYWORDS = {
    'late_delivery': ['late', 'delayed', 'delay', 'not arrived', 'hasn\'t arrived', 'hasn\'t come', 'where is', 'when will', 'overdue', 'still waiting', 'expected delivery', 'delivery date', 'past due'],
    'missing_package': ['missing', 'never received', 'not received', 'didn\'t receive', 'didnt receive', 'lost', 'stolen', 'wrong address', 'delivered to wrong', 'not delivered', 'package missing', 'never got'],
    'wrong_item': ['wrong item', 'wrong product', 'incorrect item', 'received wrong', 'different item', 'not what i ordered', 'not what i ordered', 'mistake', 'incorrect'],
    'refund_issue': ['refund', 'money back', 'charge back', 'reimburse', 'credited back', 'haven\'t received refund', 'where is my refund', 'refund status'],
    'payment_issue': ['charged twice', 'double charge', 'overcharged', 'payment', 'billing', 'charged', 'credit card', 'transaction', 'payment failed', 'payment error'],
    'order_cancellation': ['cancel', 'cancellation', 'cancel order', 'stop order', 'don\'t want anymore', 'remove order'],
    'return_item': ['return', 'send back', 'returning', 'return label', 'return shipping', 'return policy'],
    'account_issue': ['account', 'password', 'login', 'sign in', 'can\'t access', 'locked out', 'suspended', 'banned', 'blocked', 'profile'],
    'technical_issue': ['app', 'website', 'not working', 'error', 'bug', 'crash', 'freeze', 'technical', 'loading', 'broken'],
    'product_availability': ['out of stock', 'available', 'restock', 'when will', 'in stock', 'availability'],
    'gift_card': ['gift card', 'gift certificate', 'redeem', 'balance'],
    'promotion': ['promo', 'coupon', 'discount', 'deal', 'offer', 'promotion', 'code'],
    'general_inquiry': ['question', 'help', 'info', 'information', 'how do i', 'how to', 'can i'],
    'complaint': ['unacceptable', 'terrible', 'worst', 'disgusted', 'furious', 'angry', 'frustrated', 'ridiculous', 'pathetic', 'horrible'],
}


def classify_by_keywords(text):
    """Classify text using keyword matching."""
    if not isinstance(text, str):
        return 'general_inquiry'
    
    text_lower = text.lower()
    scores = {}
    
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score
    
    if scores:
        return max(scores, key=scores.get)
    return 'general_inquiry'


def explore_intents(df, sample_size=2000):
    """Explore intent distribution in the dataset."""
    print("=" * 60)
    print("INTENT EXPLORATION")
    print("=" * 60)
    
    # Sample for exploration
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Classify each customer problem
    intents = sample['customer_problem'].apply(classify_by_keywords)
    
    # Distribution
    print("\n--- Intent Distribution (keyword-based) ---")
    intent_counts = intents.value_counts()
    for intent, count in intent_counts.items():
        pct = count / len(intents) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")
    
    # Show examples for each intent
    print("\n--- Examples per Intent ---")
    for intent in intent_counts.index[:12]:  # Top 12 intents
        examples = sample[intents == intent]['customer_problem'].head(3)
        print(f"\n  [{intent}]")
        for ex in examples:
            try:
                print(f"    - {str(ex)[:100]}")
            except UnicodeEncodeError:
                print(f"    - (unicode, {len(str(ex))} chars)")
    
    return intents, intent_counts


def main():
    """Build intent taxonomy from data."""
    # Load filtered conversations
    path = "data/processed/filtered_conversations_sample.csv"
    print(f"Loading from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} conversations")
    
    # Explore intents
    intents, distribution = explore_intents(df)
    
    # Define final taxonomy
    taxonomy = {
        "intents": [
            {
                "intent": "late_delivery",
                "definition": "Customer reports that an expected package has not arrived or is delayed beyond the expected delivery date.",
                "examples": [],
                "confusing_with": ["missing_package", "order_cancellation"]
            },
            {
                "intent": "missing_package",
                "definition": "Customer reports that a package was marked as delivered but was not received, or is lost/stolen.",
                "examples": [],
                "confusing_with": ["late_delivery", "wrong_item"]
            },
            {
                "intent": "wrong_item",
                "definition": "Customer received an incorrect or different item than what was ordered.",
                "examples": [],
                "confusing_with": ["missing_package", "return_item"]
            },
            {
                "intent": "refund_issue",
                "definition": "Customer is inquiring about a refund, hasn't received an expected refund, or has issues with refund processing.",
                "examples": [],
                "confusing_with": ["payment_issue", "return_item"]
            },
            {
                "intent": "payment_issue",
                "definition": "Customer reports payment problems such as double charges, failed payments, or billing errors.",
                "examples": [],
                "confusing_with": ["refund_issue", "account_issue"]
            },
            {
                "intent": "order_cancellation",
                "definition": "Customer wants to cancel an order or reports issues with order cancellation.",
                "examples": [],
                "confusing_with": ["return_item", "late_delivery"]
            },
            {
                "intent": "return_item",
                "definition": "Customer wants to return an item, needs a return label, or has questions about return policy.",
                "examples": [],
                "confusing_with": ["refund_issue", "wrong_item"]
            },
            {
                "intent": "account_issue",
                "definition": "Customer has problems with their Amazon account, including login issues, password resets, or account access.",
                "examples": [],
                "confusing_with": ["technical_issue", "payment_issue"]
            },
            {
                "intent": "technical_issue",
                "definition": "Customer reports technical problems with the Amazon website, app, or services.",
                "examples": [],
                "confusing_with": ["account_issue", "general_inquiry"]
            },
            {
                "intent": "product_inquiry",
                "definition": "Customer asks about product availability, restocking, or general product questions.",
                "examples": [],
                "confusing_with": ["general_inquiry", "late_delivery"]
            },
            {
                "intent": "gift_card",
                "definition": "Customer has questions or issues related to gift cards or gift certificates.",
                "examples": [],
                "confusing_with": ["payment_issue", "general_inquiry"]
            },
            {
                "intent": "general_inquiry",
                "definition": "General questions or requests that don't fit into other specific categories.",
                "examples": [],
                "confusing_with": []
            },
        ]
    }
    
    # Save taxonomy
    output_path = "data/processed/intent_taxonomy.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    print(f"\nSaved taxonomy to {output_path}")
    
    # Assign intents to the full filtered dataset
    print("\nAssigning intents to all conversations...")
    all_path = "data/processed/filtered_conversations.csv"
    all_df = pd.read_csv(all_path, low_memory=False)
    all_df['intent'] = all_df['customer_problem'].apply(classify_by_keywords)
    
    # Save with intents
    all_df.to_csv(all_path, index=False)
    print(f"Updated {all_path} with intent labels")
    
    # Show distribution on full dataset
    print("\n--- Full Dataset Intent Distribution ---")
    full_dist = all_df['intent'].value_counts()
    for intent, count in full_dist.items():
        pct = count / len(all_df) * 100
        print(f"  {intent}: {count:,} ({pct:.1f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
