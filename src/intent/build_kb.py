"""
Build Historical Resolution Knowledge Base.
Extracts resolution cases from conversations for RAG retrieval.
Run: python -m src.intent.build_kb
"""
import logging
import pandas as pd
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

log = logging.getLogger(__name__)


def extract_amazon_actions(amazon_responses):
    """
    Extract what Amazon actually did from their responses.
    
    Args:
        amazon_responses: String containing Amazon's responses
        
    Returns:
        String summarizing Amazon's actions
    """
    if not isinstance(amazon_responses, str):
        return "No response recorded."
    
    actions = []
    responses = amazon_responses.strip().split('\n')
    
    for resp in responses:
        resp = resp.strip()
        if not resp:
            continue
        
        # Remove @mentions
        resp_clean = re.sub(r'@\w+', '', resp).strip()
        
        if len(resp_clean) < 5:
            continue
        
        # Detect action types
        if any(kw in resp_clean.lower() for kw in ['please contact', 'please call', 'please email', 'please dm', 'please message']):
            actions.append("Directed customer to contact support.")
        elif any(kw in resp_clean.lower() for kw in ['check your order', 'track your', 'order status', 'check the status']):
            actions.append("Directed customer to check order status.")
        elif any(kw in resp_clean.lower() for kw in ['sorry', 'apologize', 'apology', 'regret']):
            actions.append("Acknowledged the issue and apologized.")
        elif any(kw in resp_clean.lower() for kw in ['refund', 'credited', 'money back']):
            actions.append("Processed or initiated refund.")
        elif any(kw in resp_clean.lower() for kw in ['replace', 'replacement', 'send a new']):
            actions.append("Offered replacement.")
        elif any(kw in resp_clean.lower() for kw in ['cancel', 'cancelled']):
            actions.append("Assisted with cancellation.")
        elif any(kw in resp_clean.lower() for kw in ['link', 'http', 'url']):
            actions.append("Provided support link.")
        elif len(resp_clean) > 20:
            actions.append(f"Provided support guidance.")
    
    if not actions:
        return "Provided general support response."
    
    # Deduplicate while preserving order
    seen = set()
    unique_actions = []
    for action in actions:
        if action not in seen:
            seen.add(action)
            unique_actions.append(action)
    
    return " ".join(unique_actions)


def assess_resolution_quality(conversation_text, amazon_responses, turn_count):
    """
    Assess the quality of a resolution based on conversation content.
    
    Args:
        conversation_text: Full conversation text
        amazon_responses: Amazon's responses
        turn_count: Number of turns in conversation
        
    Returns:
        Quality rating: 'strong', 'medium', or 'weak'
    """
    score = 0
    
    # Check if Amazon provided specific actions
    if isinstance(amazon_responses, str):
        resp_lower = amazon_responses.lower()
        
        # Strong indicators
        if any(kw in resp_lower for kw in ['refund', 'replacement', 'cancel', 'credited']):
            score += 3
        if any(kw in resp_lower for kw in ['contact', 'call', 'email', 'dm']):
            score += 2
        if any(kw in resp_lower for kw in ['check', 'track', 'status', 'link']):
            score += 1
        if any(kw in resp_lower for kw in ['sorry', 'apologize']):
            score += 1
        
        # Response length as proxy for detail
        if len(amazon_responses) > 200:
            score += 1
        elif len(amazon_responses) > 100:
            score += 0.5
    
    # Turn count: more turns can mean more thorough handling
    if turn_count >= 4:
        score += 1
    
    # Determine quality
    if score >= 4:
        return 'strong'
    elif score >= 2:
        return 'medium'
    else:
        return 'weak'


def build_cases(input_path, output_path):
    """
    Build resolution cases from filtered conversations.
    
    Args:
        input_path: Path to filtered conversations CSV
        output_path: Path to output cases CSV
    """
    log.info("Loading conversations from %s...", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    log.info("Loaded %s conversations", f"{len(df):,}")

    cases = []
    for idx, row in df.iterrows():
        case_id = f"C{idx:06d}"

        # Extract customer problem
        customer_problem = str(row.get('customer_problem', ''))
        if pd.isna(row.get('customer_problem')):
            customer_problem = ''

        # Extract Amazon actions
        amazon_responses = str(row.get('amazon_responses', ''))
        resolution = extract_amazon_actions(amazon_responses)

        # Assess quality
        quality = assess_resolution_quality(
            str(row.get('conversation', '')),
            amazon_responses,
            row.get('turn_count', 2)
        )

        case = {
            'case_id': case_id,
            'conversation_id': row.get('conversation_id', ''),
            'customer_problem': customer_problem,
            'conversation': str(row.get('conversation', '')),
            'amazon_responses': amazon_responses,
            'resolution': resolution,
            'intent': row.get('intent', 'general_inquiry'),
            'resolution_quality': quality,
            'turn_count': row.get('turn_count', 2),
        }
        cases.append(case)

    cases_df = pd.DataFrame(cases)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cases_df.to_csv(output_path, index=False)
    log.info("Saved %s cases to %s", f"{len(cases_df):,}", output_path)

    # Statistics
    log.info("\n--- Case Statistics ---")
    log.info("Total cases: %s", f"{len(cases_df):,}")
    log.info("\nIntent distribution:")
    for intent, count in cases_df['intent'].value_counts().items():
        log.info("  %s: %s", intent, f"{count:,}")
    log.info("\nResolution quality:")
    for quality, count in cases_df['resolution_quality'].value_counts().items():
        log.info("  %s: %s", quality, f"{count:,}")
    
    return cases_df


def main():
    """Build the full knowledge base."""
    input_path = "data/processed/filtered_conversations.csv"
    output_path = "data/kb/amazonhelp_cases.csv"
    
    build_cases(input_path, output_path)
    
    log.info("\nDone!")


if __name__ == "__main__":
    main()
