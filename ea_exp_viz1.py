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
# Visualization 2a: Category Post Counts (Heatmap Style)
if 'category' in df.columns:
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    plt.figure(figsize=(12, 8))
    
    # Get category counts and sort descending (already default for value_counts)
    category_counts = df['category'].value_counts()
    
    # Reverse order so highest count appears at the top of horizontal bar chart
    category_counts = category_counts.iloc[::-1]
    
    # Create heatmap colors based on counts (YlOrRd = Yellow-Orange-Red, hot colors for high values)
    norm = Normalize(vmin=category_counts.min(), vmax=category_counts.max())
    cmap = plt.cm.YlOrRd
    bar_colors = [cmap(norm(val)) for val in category_counts.values]
    
    # Create horizontal bar chart with heatmap colors
    bars = plt.barh(range(len(category_counts)), category_counts.values, color=bar_colors)
    plt.yticks(range(len(category_counts)), category_counts.index)
    
    # Add count and percentage labels on the bars
    total_posts = len(df)
    for i, (count, bar) in enumerate(zip(category_counts.values, bars)):
        pct = count / total_posts * 100
        plt.text(count + max(category_counts.values) * 0.01, i, f'{count:,} ({pct:.1f}%)', 
                 va='center', fontsize=10)
    
    # Add colorbar to show the scale
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
    cbar.set_label('Post Count (Hotter = More Posts)', fontsize=10)
    
    plt.xlabel('Number of Posts', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.title(f'Posts per Category (Total: {len(df):,} posts)', fontsize=14, fontweight='bold')
    plt.xlim(0, max(category_counts.values) * 1.20)  # Extra space for percentage labels
    plt.tight_layout()
    plt.savefig('posts_per_category.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary (use original descending order for readability)
    category_counts_print = df['category'].value_counts()
    print("\n" + "-" * 40)
    print("📊 POSTS PER CATEGORY:")
    print("-" * 40)
    for cat, count in category_counts_print.items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count:,} posts ({pct:.1f}%)")

# %%
# Visualization 2b: Raw Sentiment Counts by Category
if 'category' in df.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get raw counts of each sentiment per category
    sentiment_by_cat = pd.crosstab(df['category'], df['sentiment'])
    
    # Reorder columns to Positive, Neutral, Negative
    col_order = [c for c in ['Positive', 'Neutral', 'Negative'] if c in sentiment_by_cat.columns]
    sentiment_by_cat = sentiment_by_cat[col_order]
    
    # Sort by total posts (descending) then reverse for horizontal bar
    sentiment_by_cat['_total'] = sentiment_by_cat.sum(axis=1)
    sentiment_by_cat = sentiment_by_cat.sort_values('_total', ascending=True)
    sentiment_by_cat = sentiment_by_cat.drop('_total', axis=1)
    
    # Create grouped horizontal bar chart
    y_pos = range(len(sentiment_by_cat))
    bar_height = 0.25
    
    for i, sentiment in enumerate(col_order):
        offset = (i - len(col_order)/2 + 0.5) * bar_height
        bars = ax.barh([y + offset for y in y_pos], sentiment_by_cat[sentiment], 
                       height=bar_height, label=sentiment, 
                       color=colors.get(sentiment, '#95a5a6'))
        
        # Add count labels
        for j, (val, bar) in enumerate(zip(sentiment_by_cat[sentiment], bars)):
            if val > 0:
                ax.text(val + max(sentiment_by_cat.max()) * 0.01, 
                       y_pos[j] + offset, f'{val:,}', 
                       va='center', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sentiment_by_cat.index)
    ax.set_xlabel('Number of Posts', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title('Raw Sentiment Counts by Category', fontsize=14, fontweight='bold')
    ax.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlim(0, sentiment_by_cat.max().max() * 1.15)
    
    plt.tight_layout()
    plt.savefig('sentiment_counts_by_category.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "-" * 60)
    print("📊 RAW SENTIMENT COUNTS BY CATEGORY:")
    print("-" * 60)
    summary = pd.crosstab(df['category'], df['sentiment'])
    summary = summary.reindex(columns=[c for c in ['Positive', 'Neutral', 'Negative'] if c in summary.columns])
    summary['Total'] = summary.sum(axis=1)
    summary = summary.sort_values('Total', ascending=False)
    print(summary.to_string())

# %%
# Visualization 2c: Sentiment by Category (Percentage Stacked)
if 'category' in df.columns:
    plt.figure(figsize=(14, 8))
    
    # Get category counts for labels (descending order by count)
    category_counts = df['category'].value_counts()
    
    # Calculate sentiment percentages by category
    category_sentiment = pd.crosstab(
        df['category'], 
        df['sentiment'], 
        normalize='index'
    ) * 100
    
    # Sort by category count (descending) - reindex to match category_counts order
    # Then reverse so highest count appears at top of horizontal bar chart
    category_sentiment = category_sentiment.reindex(category_counts.index)
    category_sentiment = category_sentiment.iloc[::-1]
    
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
• posts_per_category.png (heatmap style)
• sentiment_counts_by_category.png (raw counts)
• sentiment_by_category.png (percentages)
• compound_score_distribution.png
• confidence_and_engagement.png
• correlation_matrix.png
""")