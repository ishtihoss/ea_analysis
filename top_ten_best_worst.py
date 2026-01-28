import pandas as pd

def analyze_top_posts(csv_path: str = "ea_forum_sentiment_roberta.csv"):
    """Analyze and display top positive and negative posts by comments and views."""
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Display settings for pandas
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    
    # Parse timestamps and show date range
    # Note: Timestamps are timezone-naive (no TZ offset in source data, e.g. "2026-01-21T09:30:00")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    print("=" * 100)
    print("DATA RANGE")
    print("=" * 100)
    print(f"Earliest post: {min_date.strftime('%Y-%m-%d %H:%M:%S')} (timezone unknown)")
    print(f"Latest post:   {max_date.strftime('%Y-%m-%d %H:%M:%S')} (timezone unknown)")
    print(f"Total posts:   {len(df):,}")
    print()
    
    # Columns to display
    display_cols = ['title', 'author', 'category', 'timestamp', 'compound_score', 'confidence', 'likes', 'comments', 'views', 'body_preview']
    
    # Filter negative posts (sentiment == 'Negative')
    negative_df = df[df['sentiment'] == 'Negative'].copy()
    negative_df = negative_df.sort_values(by='comments', ascending=False)
    top_negative = negative_df.head(10)
    
    # Print results
    print("=" * 100)
    print("TOP 10 NEGATIVE POSTS (by comments)")
    print("=" * 100)
    
    for idx, (_, row) in enumerate(top_negative.iterrows(), 1):
        print(f"\n{'─' * 80}")
        print(f"#{idx}")
        print(f"Title:       {row['title']}")
        print(f"Author:      {row['author']}")
        print(f"Category:    {row['category']}")
        print(f"Timestamp:   {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Score:       {row['compound_score']:.4f}")
        print(f"Confidence:  {row['confidence']:.4f}")
        print(f"Views:       {row['views']:,}")
        print(f"Likes:       {row['likes']:,}")
        print(f"Comments:    {row['comments']:,}")
        print(f"Body:")
        print(f"  {row['body_preview']}")
    



if __name__ == "__main__":
    analyze_top_posts()