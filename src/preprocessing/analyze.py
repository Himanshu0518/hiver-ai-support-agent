"""
Data analysis for AmazonHelp conversations dataset.
Run: python -m src.preprocessing.analyze
"""
import pandas as pd
import os
from collections import Counter


def load_data():
    """Load conversations and tweets CSVs."""
    conv_path = os.path.join("data", "raw", "amazonhelp_conversations.csv")
    tweets_path = os.path.join("data", "raw", "amazonhelp_tweets.csv")
    
    print(f"Loading conversations from {conv_path}...")
    conversations = pd.read_csv(conv_path, low_memory=False)
    print(f"Loading tweets from {tweets_path}...")
    tweets = pd.read_csv(tweets_path, low_memory=False)
    
    return conversations, tweets


def analyze_conversations(conversations):
    """Analyze conversation-level statistics."""
    print("\n" + "=" * 60)
    print("CONVERSATION ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal conversations: {len(conversations):,}")
    print(f"Columns: {list(conversations.columns)}")
    
    # Turn count distribution
    if 'turn_count' in conversations.columns:
        print(f"\n--- Turn Count Distribution ---")
        tc = conversations['turn_count']
        print(f"Mean: {tc.mean():.2f}")
        print(f"Median: {tc.median():.1f}")
        print(f"Std: {tc.std():.2f}")
        print(f"Min: {tc.min()}")
        print(f"Max: {tc.max()}")
        
        print(f"\nTurn count value counts:")
        vc = tc.value_counts().sort_index()
        for count, freq in vc.head(20).items():
            pct = freq / len(conversations) * 100
            print(f"  {count}: {freq:,} ({pct:.1f}%)")
        
        # 2-turn vs multi-turn
        two_turn = (tc == 2).sum()
        multi_turn = (tc > 2).sum()
        print(f"\n2-turn conversations: {two_turn:,} ({two_turn/len(conversations)*100:.1f}%)")
        print(f"Multi-turn (>2): {multi_turn:,} ({multi_turn/len(conversations)*100:.1f}%)")
    
    # Missing values
    print(f"\n--- Missing Values ---")
    for col in conversations.columns:
        missing = conversations[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing:,} ({missing/len(conversations)*100:.1f}%)")
    
    # Customer problem analysis
    if 'customer_problem' in conversations.columns:
        print(f"\n--- Customer Problem Samples ---")
        problems = conversations['customer_problem'].dropna()
        print(f"Non-null customer problems: {len(problems):,}")
        
        # Word length distribution
        word_counts = problems.str.split().str.len()
        print(f"Avg words per problem: {word_counts.mean():.1f}")
        print(f"Median words per problem: {word_counts.median():.1f}")
        
        # Sample problems
        print(f"\nSample customer problems:")
        for i, prob in enumerate(problems.sample(min(10, len(problems)), random_state=42).values):
            try:
                print(f"  {i+1}. {str(prob)[:120]}")
            except UnicodeEncodeError:
                print(f"  {i+1}. (unicode text, {len(str(prob))} chars)")
    
    # Amazon responses analysis
    if 'amazon_responses' in conversations.columns:
        print(f"\n--- Amazon Responses ---")
        responses = conversations['amazon_responses'].dropna()
        print(f"Non-null amazon responses: {len(responses):,}")
        
        # Response length
        resp_lengths = responses.str.len()
        print(f"Avg response length: {resp_lengths.mean():.0f} chars")
        print(f"Median response length: {resp_lengths.median():.0f} chars")
    
    return conversations


def analyze_tweets(tweets):
    """Analyze tweet-level statistics."""
    print("\n" + "=" * 60)
    print("TWEET ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal tweets: {len(tweets):,}")
    print(f"Columns: {list(tweets.columns)}")
    
    # Inbound vs outbound
    if 'inbound' in tweets.columns:
        inbound = tweets['inbound'].sum()
        outbound = (~tweets['inbound']).sum()
        print(f"\nInbound (customer) tweets: {inbound:,} ({inbound/len(tweets)*100:.1f}%)")
        print(f"Outbound (agent) tweets: {outbound:,} ({outbound/len(tweets)*100:.1f}%)")
    
    # Missing values
    print(f"\n--- Missing Values ---")
    for col in tweets.columns:
        missing = tweets[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing:,} ({missing/len(tweets)*100:.1f}%)")
    
    # Text length
    if 'text' in tweets.columns:
        text_len = tweets['text'].str.len()
        print(f"\n--- Text Length ---")
        print(f"Mean: {text_len.mean():.0f} chars")
        print(f"Median: {text_len.median():.0f} chars")
        print(f"Max: {text_len.max():.0f} chars")
    
    # Unique authors
    if 'author_id' in tweets.columns:
        print(f"\nUnique authors: {tweets['author_id'].nunique():,}")
        top_authors = tweets['author_id'].value_counts().head(10)
        print(f"Top authors:")
        for author, count in top_authors.items():
            print(f"  {author}: {count:,}")
    
    # Conversation coverage
    if 'conversation_id' in tweets.columns:
        conv_ids = tweets['conversation_id'].nunique()
        print(f"\nUnique conversations in tweets: {conv_ids:,}")
    
    return tweets


def find_common_patterns(texts, top_n=20):
    """Find common word patterns in text."""
    # Simple word frequency
    all_words = []
    for text in texts.dropna():
        words = str(text).lower().split()
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    print(f"\nTop {top_n} words:")
    for word, count in word_freq.most_common(top_n):
        try:
            print(f"  {word}: {count:,}")
        except UnicodeEncodeError:
            print(f"  (unicode word): {count:,}")


def main():
    """Run full data analysis."""
    conversations, tweets = load_data()
    
    conversations = analyze_conversations(conversations)
    tweets = analyze_tweets(tweets)
    
    # Pattern analysis on customer problems
    if 'customer_problem' in conversations.columns:
        print("\n" + "=" * 60)
        print("CUSTOMER PROBLEM PATTERNS")
        print("=" * 60)
        find_common_patterns(conversations['customer_problem'])
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
