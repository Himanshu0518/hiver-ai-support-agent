"""
Data analysis for AmazonHelp conversations dataset.
Run: python -m src.preprocessing.analyze
"""
import logging
import pandas as pd
import os
from collections import Counter

log = logging.getLogger(__name__)


def load_data():
    """Load conversations and tweets CSVs."""
    conv_path = os.path.join("data", "raw", "amazonhelp_conversations.csv")
    tweets_path = os.path.join("data", "raw", "amazonhelp_tweets.csv")

    log.info("Loading conversations from %s...", conv_path)
    conversations = pd.read_csv(conv_path, low_memory=False)
    log.info("Loading tweets from %s...", tweets_path)
    tweets = pd.read_csv(tweets_path, low_memory=False)
    
    return conversations, tweets


def analyze_conversations(conversations):
    """Analyze conversation-level statistics."""
    log.info("\n" + "=" * 60)
    log.info("CONVERSATION ANALYSIS")
    log.info("=" * 60)

    log.info("\nTotal conversations: %s", f"{len(conversations):,}")
    log.info("Columns: %s", list(conversations.columns))
    
    # Turn count distribution
    if 'turn_count' in conversations.columns:
        log.info("\n--- Turn Count Distribution ---")
        tc = conversations['turn_count']
        log.info("Mean: %.2f", tc.mean())
        log.info("Median: %.1f", tc.median())
        log.info("Std: %.2f", tc.std())
        log.info("Min: %s", tc.min())
        log.info("Max: %s", tc.max())

        log.info("\nTurn count value counts:")
        vc = tc.value_counts().sort_index()
        for count, freq in vc.head(20).items():
            pct = freq / len(conversations) * 100
            log.info("  %s: %s (%.1f%%)", count, f"{freq:,}", pct)

        # 2-turn vs multi-turn
        two_turn = (tc == 2).sum()
        multi_turn = (tc > 2).sum()
        log.info("\n2-turn conversations: %s (%.1f%%)", f"{two_turn:,}", two_turn/len(conversations)*100)
        log.info("Multi-turn (>2): %s (%.1f%%)", f"{multi_turn:,}", multi_turn/len(conversations)*100)
    
    # Missing values
    log.info("\n--- Missing Values ---")
    for col in conversations.columns:
        missing = conversations[col].isna().sum()
        if missing > 0:
            log.info("  %s: %s (%.1f%%)", col, f"{missing:,}", missing/len(conversations)*100)
    
    # Customer problem analysis
    if 'customer_problem' in conversations.columns:
        log.info("\n--- Customer Problem Samples ---")
        problems = conversations['customer_problem'].dropna()
        log.info("Non-null customer problems: %s", f"{len(problems):,}")

        # Word length distribution
        word_counts = problems.str.split().str.len()
        log.info("Avg words per problem: %.1f", word_counts.mean())
        log.info("Median words per problem: %.1f", word_counts.median())

        # Sample problems
        log.info("\nSample customer problems:")
        for i, prob in enumerate(problems.sample(min(10, len(problems)), random_state=42).values):
            try:
                log.info("  %s. %s", i + 1, str(prob)[:120])
            except UnicodeEncodeError:
                log.info("  %s. (unicode text, %s chars)", i + 1, len(str(prob)))
    
    # Amazon responses analysis
    if 'amazon_responses' in conversations.columns:
        log.info("\n--- Amazon Responses ---")
        responses = conversations['amazon_responses'].dropna()
        log.info("Non-null amazon responses: %s", f"{len(responses):,}")

        # Response length
        resp_lengths = responses.str.len()
        log.info("Avg response length: %.0f chars", resp_lengths.mean())
        log.info("Median response length: %.0f chars", resp_lengths.median())
    
    return conversations


def analyze_tweets(tweets):
    """Analyze tweet-level statistics."""
    log.info("\n" + "=" * 60)
    log.info("TWEET ANALYSIS")
    log.info("=" * 60)

    log.info("\nTotal tweets: %s", f"{len(tweets):,}")
    log.info("Columns: %s", list(tweets.columns))
    
    # Inbound vs outbound
    if 'inbound' in tweets.columns:
        inbound = tweets['inbound'].sum()
        outbound = (~tweets['inbound']).sum()
        log.info("\nInbound (customer) tweets: %s (%.1f%%)", f"{inbound:,}", inbound/len(tweets)*100)
        log.info("Outbound (agent) tweets: %s (%.1f%%)", f"{outbound:,}", outbound/len(tweets)*100)
    
    # Missing values
    log.info("\n--- Missing Values ---")
    for col in tweets.columns:
        missing = tweets[col].isna().sum()
        if missing > 0:
            log.info("  %s: %s (%.1f%%)", col, f"{missing:,}", missing/len(tweets)*100)
    
    # Text length
    if 'text' in tweets.columns:
        text_len = tweets['text'].str.len()
        log.info("\n--- Text Length ---")
        log.info("Mean: %.0f chars", text_len.mean())
        log.info("Median: %.0f chars", text_len.median())
        log.info("Max: %.0f chars", text_len.max())
    
    # Unique authors
    if 'author_id' in tweets.columns:
        log.info("\nUnique authors: %s", f"{tweets['author_id'].nunique():,}")
        top_authors = tweets['author_id'].value_counts().head(10)
        log.info("Top authors:")
        for author, count in top_authors.items():
            log.info("  %s: %s", author, f"{count:,}")
    
    # Conversation coverage
    if 'conversation_id' in tweets.columns:
        conv_ids = tweets['conversation_id'].nunique()
        log.info("\nUnique conversations in tweets: %s", f"{conv_ids:,}")
    
    return tweets


def find_common_patterns(texts, top_n=20):
    """Find common word patterns in text."""
    # Simple word frequency
    all_words = []
    for text in texts.dropna():
        words = str(text).lower().split()
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    log.info("\nTop %s words:", top_n)
    for word, count in word_freq.most_common(top_n):
        try:
            log.info("  %s: %s", word, f"{count:,}")
        except UnicodeEncodeError:
            log.info("  (unicode word): %s", f"{count:,}")


def main():
    """Run full data analysis."""
    conversations, tweets = load_data()
    
    conversations = analyze_conversations(conversations)
    tweets = analyze_tweets(tweets)
    
    # Pattern analysis on customer problems
    if 'customer_problem' in conversations.columns:    log.info("\n" + "=" * 60)
    log.info("CUSTOMER PROBLEM PATTERNS")
    log.info("=" * 60)
    find_common_patterns(conversations['customer_problem'])

    log.info("\n" + "=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
