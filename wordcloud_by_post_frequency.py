from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import re

# Load your data
df = pd.read_csv("ea_forum_sentiment_roberta.csv")

# Check what columns exist
print("Available columns:", df.columns.tolist())

custom_stopwords = STOPWORDS.union({
    'ea', 'game', 'games', 'player', 'players', 'team', 'account',
    'will', 'get', 'got', 'one', 'also', 'would', 'could', 'even',
    'like', 'just', 'now', 'know', 'dont', "don't", 'im', "i'm",
    'issue', 'play', 'fc', 'problem', 'bug', 'match', 'playing', 'time', 
    's', 'fix', 'still', 'every', 'back', 'please', 'thank', 'help',
    't', 'u', 've', 're', 'll', 'd', 'm'  # Common contractions
})

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return f"hsl(0, 80%, {np.random.randint(25, 50)}%)"

def count_words_by_post_frequency(texts, stopwords):
    """
    Count how many POSTS contain each word (not total occurrences).
    Each word is counted at most ONCE per post.
    """
    word_post_counts = Counter()
    
    for text in texts:
        if pd.isna(text):
            continue
        # Tokenize: lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        # Get unique words in this post (each word counted once per post)
        unique_words = set(words)
        # Remove stopwords
        unique_words = {w for w in unique_words if w not in stopwords}
        # Update counts
        word_post_counts.update(unique_words)
    
    return dict(word_post_counts)

# Filter negative posts
negative_df = df[df['sentiment'] == 'Negative'].copy()

# Combine title and body_preview to create full_text
# (Use the same columns your original word cloud used)
negative_df['full_text'] = (
    negative_df['title'].fillna('') + ' ' + negative_df['body_preview'].fillna('')
)
negative_texts = negative_df['full_text']

# Count words by POST FREQUENCY (not total occurrences)
word_frequencies = count_words_by_post_frequency(negative_texts, custom_stopwords)

# Generate word cloud from frequencies
plt.figure(figsize=(10, 6))
wc_neg = WordCloud(
    width=800, 
    height=400, 
    background_color='white',
    max_words=50,
    color_func=red_color_func
).generate_from_frequencies(word_frequencies)  # <-- KEY CHANGE

plt.imshow(wc_neg, interpolation='bilinear')
plt.title('🔴 Common Words in NEGATIVE Posts (by post frequency)', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

# Print top 20 words to verify
print("\nTop 20 words by number of posts containing them:")
print("-" * 40)
for word, count in sorted(word_frequencies.items(), key=lambda x: -x[1])[:20]:
    print(f"{word:20s} {count:5d} posts")