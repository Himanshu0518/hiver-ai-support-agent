"""
Conversation Quality Filtering for AmazonHelp conversations.
Filters out low-quality conversations that shouldn't become RAG cases.
"""
import pandas as pd


# Quality rules
QUALITY_CONFIG = {
    'min_turn_count': 2,
    'max_turn_count': 50,
    'min_problem_length': 5,  # characters
    'min_response_length': 5,
    'spam_keywords': ['buy now', 'click here', 'free money', 'winner', 'congratulations'],
    'compliment_keywords': ['thank you so much', 'great job', 'awesome service', 'love you'],
}


def is_spam(text):
    """Check if text is spam."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in QUALITY_CONFIG['spam_keywords'])


def is_compliment(text):
    """Check if text is a pure compliment."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower().strip()
    return any(kw in text_lower for kw in QUALITY_CONFIG['compliment_keywords'])


def has_useful_support_interaction(conversation_text):
    """Check if conversation contains useful support interaction."""
    if not isinstance(conversation_text, str):
        return False
    
    # Look for support-related keywords
    support_keywords = [
        'help', 'assist', 'support', 'contact', 'link', 'order',
        'package', 'delivery', 'refund', 'return', 'replace',
        'track', 'status', 'account', 'issue', 'problem',
        'sorry', 'apologize', 'investigate', 'resolve'
    ]
    
    text_lower = conversation_text.lower()
    return any(kw in text_lower for kw in support_keywords)


def is_valid_conversation(row):
    """
    Check if a conversation meets quality criteria.
    
    Args:
        row: pandas Series with conversation data
        
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    # Check turn count
    turn_count = row.get('turn_count', 0)
    if pd.isna(turn_count) or turn_count < QUALITY_CONFIG['min_turn_count']:
        return False, 'insufficient_turns'
    if turn_count > QUALITY_CONFIG['max_turn_count']:
        return False, 'too_many_turns'
    
    # Check customer problem
    problem = str(row.get('customer_problem', ''))
    if pd.isna(row.get('customer_problem')) or len(problem.strip()) < QUALITY_CONFIG['min_problem_length']:
        return False, 'no_customer_problem'
    
    # Check for spam
    if is_spam(problem):
        return False, 'spam'
    
    # Check for pure compliments
    if is_compliment(problem):
        return False, 'pure_compliment'
    
    # Check for useful support interaction in conversation
    conversation = str(row.get('conversation', ''))
    if not has_useful_support_interaction(conversation):
        return False, 'no_support_interaction'
    
    # Check amazon responses exist
    amazon_responses = str(row.get('amazon_responses', ''))
    if pd.isna(row.get('amazon_responses')) or len(amazon_responses.strip()) < QUALITY_CONFIG['min_response_length']:
        return False, 'no_amazon_response'
    
    return True, 'valid'


def filter_conversations(conversations):
    """
    Filter conversations based on quality criteria.
    
    Args:
        conversations: pandas DataFrame of conversations
        
    Returns:
        tuple: (filtered_df, stats_dict)
    """
    stats = {
        'total': len(conversations),
        'kept': 0,
        'removed': 0,
        'reasons': {}
    }
    
    results = []
    for idx, row in conversations.iterrows():
        is_valid, reason = is_valid_conversation(row)
        
        if is_valid:
            stats['kept'] += 1
            results.append(True)
        else:
            stats['removed'] += 1
            stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
            results.append(False)
    
    mask = pd.Series(results, index=conversations.index)
    filtered = conversations[mask].copy()
    
    return filtered, stats


def main():
    """Run quality filtering."""
    print("Loading conversations...")
    conv_path = "data/raw/amazonhelp_conversations.csv"
    conversations = pd.read_csv(conv_path, low_memory=False)
    
    print(f"Total conversations: {len(conversations):,}")
    
    filtered, stats = filter_conversations(conversations)
    
    print(f"\nQuality Filtering Results:")
    print(f"  Kept: {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"  Removed: {stats['removed']:,} ({stats['removed']/stats['total']*100:.1f}%)")
    print(f"\nRemoval reasons:")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}")
    
    # Save filtered conversations
    output_path = "data/processed/filtered_conversations.csv"
    filtered.to_csv(output_path, index=False)
    print(f"\nSaved filtered conversations to {output_path}")


if __name__ == "__main__":
    main()
