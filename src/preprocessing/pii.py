"""
PII Sanitization for AmazonHelp conversations.
Replaces sensitive information with placeholders.
"""
import re


# PII patterns
PATTERNS = {
    'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'PHONE': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    'ORDER_ID': re.compile(r'\b\d{3}-\d{7}-\d{7}\b'),
    'TRACKING_ID': re.compile(r'\b[A-Z]{2}\d{9}US\b'),
    'URL': re.compile(r'https?://t\.co/\w+|https?://[^\s<>\"\'\)]+'),
    'AMAZON_ID': re.compile(r'@\d{5,}'),
    'CREDIT_CARD': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'ZIP_CODE': re.compile(r'\b\d{5}(-\d{4})?\b'),
}

REPLACEMENTS = {
    'EMAIL': '[EMAIL]',
    'PHONE': '[PHONE]',
    'ORDER_ID': '[ORDER_ID]',
    'TRACKING_ID': '[TRACKING_ID]',
    'URL': '[URL]',
    'AMAZON_ID': '[USER_ID]',
    'CREDIT_CARD': '[CARD]',
    'SSN': '[SSN]',
    'ZIP_CODE': '[ZIP]',
}


def sanitize_text(text):
    """
    Sanitize PII from text.
    
    Args:
        text: Input string
        
    Returns:
        Sanitized string with PII replaced by placeholders
    """
    if not isinstance(text, str):
        return text
    
    result = text
    for pii_type, pattern in PATTERNS.items():
        result = pattern.sub(REPLACEMENTS[pii_type], result)
    
    return result


def sanitize_dataframe(df, text_columns=None):
    """
    Sanitize PII in all text columns of a DataFrame.
    
    Args:
        df: pandas DataFrame
        text_columns: List of column names to sanitize. If None, auto-detect string columns.
        
    Returns:
        DataFrame with sanitized text
    """
    if text_columns is None:
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    df = df.copy()
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(sanitize_text)
    
    return df


def main():
    """Test sanitization on sample data."""
    test_cases = [
        "My order 404-1234567-1234567 hasn't arrived.",
        "Contact me at john@example.com or call 555-123-4567",
        "Tracking: 1Z999AA10123456784",
        "Check this link: https://amazon.com/order/12345",
        "I was charged $99.99 on my card 4111-1111-1111-1111",
        "@12345678 help me please!",
    ]
    
    print("PII Sanitization Test:")
    print("=" * 60)
    for text in test_cases:
        sanitized = sanitize_text(text)
        print(f"Original:  {text}")
        print(f"Sanitized: {sanitized}")
        print()


if __name__ == "__main__":
    main()
