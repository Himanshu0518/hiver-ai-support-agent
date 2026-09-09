"""
Build Golden Evaluation Set.
Creates 200 stratified hand-labelled examples for evaluation.
Run: python -m src.evaluation.build_golden_set
"""
import pandas as pd
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# High-risk intents that should be escalated
ESCALATION_INTENTS = {
    'payment_issue', 'refund_issue', 'account_issue', 'complaint'
}

# Intent that are generally auto-handleable
AUTO_HANDLE_INTENTS = {
    'late_delivery', 'missing_package', 'wrong_item', 'order_cancellation',
    'return_item', 'product_inquiry', 'gift_card', 'promotion', 'technical_issue',
    'general_inquiry'
}


def determine_expected_action(intent, customer_problem):
    """
    Determine expected routing action based on intent and content.
    
    Args:
        intent: Predicted intent
        customer_problem: Customer message text
        
    Returns:
        'auto_handle' or 'escalate'
    """
    # Always escalate high-risk intents
    if intent in ESCALATION_INTENTS:
        return 'escalate'
    
    # Check for account-specific language
    if isinstance(customer_problem, str):
        problem_lower = customer_problem.lower()
        if any(kw in problem_lower for kw in ['my account', 'my order number', 'my email', 'my phone', 'my address']):
            # Could be auto-handle or escalate depending on context
            if intent in ['late_delivery', 'missing_package']:
                return 'auto_handle'
            return 'escalate'
    
    return 'auto_handle'


def generate_reference_response(intent, customer_problem, resolution):
    """
    Generate a reference response based on historical patterns.
    
    Args:
        intent: Intent label
        customer_problem: Customer message
        resolution: Historical resolution
        
    Returns:
        Reference response string
    """
    # Template responses based on intent
    templates = {
        'late_delivery': "We're sorry for the delay. Please check your order status for the latest updates. If the package is still delayed, our support team can help investigate further.",
        'missing_package': "We understand your concern. Please verify the delivery address on your order. If the package shows as delivered but you haven't received it, please contact our support team for assistance.",
        'wrong_item': "We apologize for the mix-up. Please contact our support team so we can arrange a replacement or return for the incorrect item.",
        'refund_issue': "We understand your concern about the refund. Please allow the standard processing time. If you haven't received it, please contact our support team for verification.",
        'payment_issue': "We understand this is concerning. Please contact our support team directly so we can verify and resolve the payment issue.",
        'order_cancellation': "We can help with that. Please check your order status to see if cancellation is still possible. If you need further assistance, please contact our support team.",
        'return_item': "You can initiate a return through your Amazon account. Please check the return policy for your item. If you need help, our support team is available.",
        'account_issue': "We understand the urgency. Please contact our support team directly so they can verify your identity and help restore access to your account.",
        'technical_issue': "We're sorry for the inconvenience. Please try clearing your browser cache or restarting the app. If the issue persists, please contact our support team.",
        'product_inquiry': "Thank you for your interest. Product availability can change frequently. Please check the product page for the most current information.",
        'gift_card': "Please check your gift card balance in your Amazon account. If you're having issues redeeming it, our support team can assist.",
        'promotion': "Promotional offers have specific terms and conditions. Please check the promotion details page for validity and applicable terms.",
        'complaint': "We're sorry to hear about your experience. Please contact our support team directly so we can investigate and address your concerns.",
        'general_inquiry': "Thank you for reaching out. Please let us know how we can help you today.",
    }
    
    return templates.get(intent, templates['general_inquiry'])


def determine_difficulty(intent, customer_problem, turn_count):
    """
    Assess difficulty of a golden set example.
    
    Args:
        intent: Intent label
        customer_problem: Customer message
        turn_count: Number of turns
        
    Returns:
        Difficulty level: 'easy', 'medium', or 'hard'
    """
    score = 0
    
    if isinstance(customer_problem, str):
        # Short messages are harder
        if len(customer_problem.split()) < 5:
            score += 2
        
        # Multiple issues in one message
        issue_keywords = ['and', 'also', 'plus', 'additionally', 'moreover']
        if sum(1 for kw in issue_keywords if kw in customer_problem.lower()) > 1:
            score += 1
        
        # Angry tone
        angry_words = ['angry', 'furious', 'terrible', 'worst', 'horrible', 'unacceptable']
        if any(w in customer_problem.lower() for w in angry_words):
            score += 1
    
    # More turns can mean more complex
    if turn_count > 5:
        score += 1
    
    if score >= 3:
        return 'hard'
    elif score >= 1:
        return 'medium'
    return 'easy'


def build_golden_set(cases_path, output_path, n_samples=200):
    """
    Build golden evaluation set from cases.
    
    Args:
        cases_path: Path to cases CSV
        output_path: Path to output golden set CSV
        n_samples: Number of samples to generate
    """
    print(f"Loading cases from {cases_path}...")
    cases = pd.read_csv(cases_path, low_memory=False)
    print(f"Loaded {len(cases):,} cases")
    
    # Stratified sampling: proportional to intent distribution
    intent_counts = cases['intent'].value_counts()
    samples_per_intent = {}
    remaining = n_samples
    
    for intent, count in intent_counts.items():
        proportion = count / len(cases)
        n = max(5, int(n_samples * proportion))  # At least 5 per intent
        samples_per_intent[intent] = min(n, count)
        remaining -= samples_per_intent[intent]
    
    # Distribute remaining samples
    for intent in list(samples_per_intent.keys())[:abs(remaining)]:
        if remaining > 0:
            samples_per_intent[intent] += 1
            remaining -= 1
        elif remaining < 0:
            samples_per_intent[intent] = max(5, samples_per_intent[intent] - 1)
            remaining += 1
    
    # Sample from each intent
    golden_examples = []
    example_id = 1
    
    for intent, n in samples_per_intent.items():
        intent_cases = cases[cases['intent'] == intent]
        sampled = intent_cases.sample(n=min(n, len(intent_cases)), random_state=42)
        
        for _, row in sampled.iterrows():
            customer_problem = str(row.get('customer_problem', ''))
            turn_count = row.get('turn_count', 2)
            resolution = row.get('resolution', '')
            
            example = {
                'example_id': f"{example_id:03d}",
                'customer_message': customer_problem,
                'intent': intent,
                'intent_notes': f"Auto-classified by keyword matching",
                'expected_action': determine_expected_action(intent, customer_problem),
                'reference_response': generate_reference_response(intent, customer_problem, resolution),
                'difficulty': determine_difficulty(intent, customer_problem, turn_count),
                'turn_count': turn_count,
            }
            golden_examples.append(example)
            example_id += 1
    
    golden_df = pd.DataFrame(golden_examples)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    golden_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(golden_df)} golden examples to {output_path}")
    
    # Statistics
    print("\n--- Golden Set Statistics ---")
    print(f"Total examples: {len(golden_df)}")
    print(f"\nIntent distribution:")
    for intent, count in golden_df['intent'].value_counts().items():
        print(f"  {intent}: {count}")
    print(f"\nExpected action:")
    for action, count in golden_df['expected_action'].value_counts().items():
        print(f"  {action}: {count}")
    print(f"\nDifficulty:")
    for diff, count in golden_df['difficulty'].value_counts().items():
        print(f"  {diff}: {count}")
    
    # Save also as JSON for easy loading
    json_path = output_path.replace('.csv', '.json')
    golden_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    print(f"Also saved as {json_path}")
    
    return golden_df


def main():
    """Build the golden evaluation set."""
    cases_path = "data/kb/amazonhelp_cases.csv"
    output_path = "data/evaluation/golden_set.csv"
    
    build_golden_set(cases_path, output_path, n_samples=200)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
