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
    
    # Filter for FC 26 Technical Issues category AND Negative sentiment
    technical_issues_df = df[df['category'] == 'FC 26 Technical Issues']
    negative_df = technical_issues_df[technical_issues_df['sentiment'] == 'Negative'].copy()
    
    print("=" * 100)
    print("CATEGORY FILTER: FC 26 Technical Issues")
    print("=" * 100)
    print(f"Total posts in category:    {len(technical_issues_df):,}")
    print(f"Negative posts in category: {len(negative_df):,}")
    print()
    
    # Keywords from wordcloud (common words in negative FC 26 Technical Issues posts)
    # Excluding stopwords: issue, play, fc, problem, bug, match, playing, time, fix, still, every, back, help, etc.
    keywords = [
        "crash", "lag", "freeze", "error", "disconnect", "connection",
        "loading", "screen", "fps", "stuttering", "controller", "audio",
        "graphics", "update", "patch", "PC", "console", "PS5", "Xbox",
        "menu", "stuck", "black", "kick", "timeout", "latency"
    ]
    
    # Count negative posts for each keyword
    keyword_counts = []
    for keyword in keywords:
        mask = (
            negative_df['title'].str.contains(keyword, case=False, na=False) |
            negative_df['body_preview'].str.contains(keyword, case=False, na=False)
        )
        count = mask.sum()
        keyword_counts.append((keyword, count))
    
    # Sort keywords by count in descending order (most negative posts first)
    keyword_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Print results in sorted order
    for keyword, _ in keyword_counts:
        print_keyword_results(negative_df, keyword)


def print_keyword_results(negative_df: pd.DataFrame, keyword: str):
    """Print top 3 negative posts containing a specific keyword."""
    
    # Case-insensitive search in title and body_preview
    mask = (
        negative_df['title'].str.contains(keyword, case=False, na=False) |
        negative_df['body_preview'].str.contains(keyword, case=False, na=False)
    )
    
    filtered_df = negative_df[mask].copy()
    filtered_df = filtered_df.sort_values(by='comments', ascending=False)
    top_results = filtered_df.head(3)
    
    print("=" * 100)
    print(f"TOP 3 NEGATIVE POSTS CONTAINING \"{keyword}\" (by comments)")
    print(f"Found {len(filtered_df):,} negative posts containing \"{keyword}\"")
    print("=" * 100)
    
    if len(top_results) == 0:
        print("\nNo posts found matching this criteria.\n")
        return
    
    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
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
    
    print("\n\n")


if __name__ == "__main__":
    analyze_top_posts()
