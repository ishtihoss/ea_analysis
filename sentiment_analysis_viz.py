# %%
# Visualization 1: Overall Sentiment Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}

# Pie chart
ax1 = axes[0]
sentiment_counts = df['sentiment'].value_counts()
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
        colors=[colors.get(l, '#95a5a6') for l in sentiment_counts.index],
        explode=[0.03] * len(sentiment_counts))
ax1.set_title('Overall Sentiment Distribution\n(RoBERTa Twitter-trained)', fontsize=14, fontweight='bold')

# Bar chart with counts
ax2 = axes[1]
bars = ax2.bar(sentiment_counts.index, sentiment_counts.values,
               color=[colors.get(l, '#95a5a6') for l in sentiment_counts.index])
ax2.set_xlabel('Sentiment', fontsize=12)
ax2.set_ylabel('Number of Posts', fontsize=12)
ax2.set_title('Sentiment Counts', fontsize=14)

# Add count labels on bars
for bar, count in zip(bars, sentiment_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sentiment_counts.values)*0.01,
             f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualization 2a: Category Post Counts
if 'category' in df.columns:
    plt.figure(figsize=(12, 8))
    
    # Get category counts and sort
    category_counts = df['category'].value_counts()
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(category_counts)), category_counts.values, color='steelblue')
    plt.yticks(range(len(category_counts)), category_counts.index)
    
    # Add count labels on the bars
    for i, (count, bar) in enumerate(zip(category_counts.values, bars)):
        plt.text(count + max(category_counts.values) * 0.01, i, f'{count:,}', 
                 va='center', fontsize=10)
    
    plt.xlabel('Number of Posts', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.title(f'Posts per Category (Total: {len(df):,} posts)', fontsize=14, fontweight='bold')
    plt.xlim(0, max(category_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig('posts_per_category.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "-" * 40)
    print("📊 POSTS PER CATEGORY:")
    print("-" * 40)
    for cat, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count:,} posts ({pct:.1f}%)")

# %%
# Visualization 2b: Sentiment by Category
if 'category' in df.columns:
    plt.figure(figsize=(14, 8))
    
    # Get category counts for labels
    category_counts = df['category'].value_counts()
    
    # Calculate sentiment percentages by category
    category_sentiment = pd.crosstab(
        df['category'], 
        df['sentiment'], 
        normalize='index'
    ) * 100
    
    # Sort by negativity
    if 'Negative' in category_sentiment.columns:
        category_sentiment = category_sentiment.sort_values('Negative', ascending=True)
    
    # Create labels with counts
    new_labels = [f"{cat} (n={category_counts[cat]:,})" for cat in category_sentiment.index]
    category_sentiment.index = new_labels
    
    # Plot
    category_sentiment.plot(
        kind='barh',
        stacked=True,
        color=[colors.get(c, '#95a5a6') for c in category_sentiment.columns],
        figsize=(14, 8)
    )
    
    plt.title('Sentiment Distribution by Forum Category (RoBERTa Analysis)', fontsize=14)
    plt.xlabel('Percentage of Posts')
    plt.ylabel('Category')
    plt.legend(title='Sentiment', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig('sentiment_by_category.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
# Visualization 3: Compound Score Distribution
plt.figure(figsize=(12, 5))
sns.histplot(df['compound_score'], bins=50, kde=True, color='steelblue')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.axvline(x=df['compound_score'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["compound_score"].mean():.3f}')
plt.title('Distribution of Sentiment Scores (RoBERTa)', fontsize=14)
plt.xlabel('Compound Score (-1 = Very Negative, +1 = Very Positive)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('compound_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Visualization 4: Confidence Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confidence by sentiment
ax1 = axes[0]
sns.boxplot(data=df, x='sentiment', y='confidence', 
            palette=colors, ax=ax1, order=['Negative', 'Neutral', 'Positive'])
ax1.set_title('Model Confidence by Sentiment', fontsize=12)
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Confidence Score')

# Sentiment vs Engagement
ax2 = axes[1]
if 'comments' in df.columns:
    sns.scatterplot(data=df, x='compound_score', y='comments', 
                    hue='sentiment', palette=colors, alpha=0.6, ax=ax2)
    ax2.set_title('Sentiment vs Comment Engagement', fontsize=12)
    ax2.set_xlabel('Sentiment Score')
    ax2.set_ylabel('Number of Comments')
    ax2.legend(title='Sentiment')

plt.tight_layout()
plt.savefig('confidence_and_engagement.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Correlation Analysis
print("\n" + "=" * 60)
print("📈 CORRELATION ANALYSIS")
print("=" * 60)

numeric_cols = ['compound_score', 'confidence']
if 'views' in df.columns:
    numeric_cols.append('views')
if 'likes' in df.columns:
    numeric_cols.append('likes')
if 'comments' in df.columns:
    numeric_cols.append('comments')

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, fmt='.3f',
            square=True, linewidths=0.5)
plt.title('Correlation Matrix: Sentiment vs Engagement', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Word Analysis for Negative Posts
try:
    from wordcloud import WordCloud, STOPWORDS
    
    # Custom stopwords for EA forums
    custom_stopwords = STOPWORDS.union({
        'ea', 'game', 'games', 'player', 'players', 'team', 'account',
        'will', 'get', 'got', 'one', 'also', 'would', 'could', 'even',
        'like', 'just', 'now', 'know', 'dont', "don't", 'im', "i'm"
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Negative word cloud
    negative_text = ' '.join(df[df['sentiment'] == 'Negative']['full_text'].dropna())
    if negative_text:
        wc_neg = WordCloud(width=800, height=400, background_color='white',
                          stopwords=custom_stopwords, colormap='Reds',
                          max_words=100).generate(negative_text)
        axes[0].imshow(wc_neg, interpolation='bilinear')
        axes[0].set_title('🔴 Common Words in NEGATIVE Posts', fontsize=14)
        axes[0].axis('off')
    
    # Positive word cloud
    positive_text = ' '.join(df[df['sentiment'] == 'Positive']['full_text'].dropna())
    if positive_text:
        wc_pos = WordCloud(width=800, height=400, background_color='white',
                          stopwords=custom_stopwords, colormap='Greens',
                          max_words=100).generate(positive_text)
        axes[1].imshow(wc_pos, interpolation='bilinear')
        axes[1].set_title('🟢 Common Words in POSITIVE Posts', fontsize=14)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('wordclouds.png', dpi=150, bbox_inches='tight')
    plt.show()
    
except ImportError:
    print("WordCloud not installed. Run: pip install wordcloud")

# %%
# Export results
output_columns = [
    'title', 'author', 'category', 'timestamp',
    'sentiment', 'confidence', 'compound_score',
    'views', 'likes', 'comments',
    'body_preview'
]

# Only include columns that exist
output_columns = [col for col in output_columns if col in df.columns]
output_df = df[output_columns].copy()

output_df.to_csv('ea_forum_sentiment_roberta.csv', index=False)
print("✅ Results exported to 'ea_forum_sentiment_roberta.csv'")

# %%
# Final Summary Report
print("\n" + "=" * 60)
print("📊 FINAL SENTIMENT ANALYSIS REPORT")
print("=" * 60)

total = len(df)
neg_count = (df['sentiment'] == 'Negative').sum()
pos_count = (df['sentiment'] == 'Positive').sum()
neu_count = (df['sentiment'] == 'Neutral').sum()

# Determine overall sentiment
if df['compound_score'].mean() < -0.1:
    overall = "NEGATIVE 😠"
elif df['compound_score'].mean() > 0.1:
    overall = "POSITIVE 😊"
else:
    overall = "MIXED 😐"

# Determine engagement trend
if 'comments' in df.columns:
    high_engagement = df[df['comments'] > df['comments'].median()]
    engagement_trend = "more negative" if high_engagement['compound_score'].mean() < 0 else "more positive"
else:
    engagement_trend = "N/A"

print(f"""
📈 OVERVIEW
{'─' * 40}
Total Posts Analyzed: {total:,}
Model: RoBERTa (Twitter-trained)
Average Sentiment Score: {df['compound_score'].mean():.3f}
Average Model Confidence: {df['confidence'].mean():.3f}

📊 SENTIMENT BREAKDOWN
{'─' * 40}
🟢 Positive: {pos_count:,} ({pos_count/total*100:.1f}%)
🟡 Neutral:  {neu_count:,} ({neu_count/total*100:.1f}%)
🔴 Negative: {neg_count:,} ({neg_count/total*100:.1f}%)

🎯 KEY INSIGHTS
{'─' * 40}
• Overall forum sentiment: {overall}
• Most discussed issues appear in negative posts
• High-engagement posts tend to be {engagement_trend}

📁 FILES GENERATED
{'─' * 40}
• ea_forum_sentiment_roberta.csv (full results)
• sentiment_distribution.png
• posts_per_category.png
• sentiment_by_category.png
• compound_score_distribution.png
• confidence_and_engagement.png
• correlation_matrix.png
• wordclouds.png
""")