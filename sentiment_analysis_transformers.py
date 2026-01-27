# %% [markdown]
# # Sentiment Analysis with RoBERTa Transformer
# Using Twitter-trained RoBERTa model for EA Forum Posts

# %%
# Install required packages (run once)
# !pip install pandas transformers torch accelerate matplotlib seaborn wordcloud tqdm

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using: {'GPU 🚀' if device == 0 else 'CPU'}")

# %%
# Load the data
df = pd.read_csv('ea_forum_posts.csv')
print(f"Loaded {len(df)} posts")

# Combine title and body for better context
df['full_text'] = df['title'].fillna('') + '. ' + df['body_preview'].fillna('')
df['full_text'] = df['full_text'].str.strip()

df.head()

# %% [markdown]
# ## RoBERTa Model
# Using Cardiff NLP's RoBERTa trained on Twitter data - excellent for informal text and forums

# %%
# Initialize RoBERTa sentiment analysis pipeline
print("Loading RoBERTa model (this may take a minute)...")

sentiment_roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device,
    truncation=True,
    max_length=512
)

print("✅ RoBERTa model loaded!")

# %%
# Function to safely analyze sentiment with error handling
def analyze_sentiment(text, analyzer):
    """Analyze sentiment with error handling for long/problematic texts"""
    if pd.isna(text) or not text.strip():
        return {'label': 'neutral', 'score': 0.0}
    
    try:
        # Truncate very long texts
        text = str(text)[:1500]
        result = analyzer(text)[0]
        return result
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return {'label': 'neutral', 'score': 0.0}

# %%
# Analyze with RoBERTa
print("Analyzing sentiment with RoBERTa...")
tqdm.pandas(desc="RoBERTa Analysis")

roberta_results = df['full_text'].progress_apply(
    lambda x: analyze_sentiment(x, sentiment_roberta)
)

df['roberta_label'] = roberta_results.apply(lambda x: x['label'])
df['roberta_score'] = roberta_results.apply(lambda x: x['score'])

# Map labels to standardized format
label_map_roberta = {
    'positive': 'Positive',
    'negative': 'Negative', 
    'neutral': 'Neutral'
}
df['sentiment'] = df['roberta_label'].map(label_map_roberta)
df['confidence'] = df['roberta_score']

# Create compound score (-1 to 1)
def calculate_compound(row):
    label = row['sentiment']
    score = row['confidence']
    if label == 'Positive':
        return score
    elif label == 'Negative':
        return -score
    return 0

df['compound_score'] = df.apply(calculate_compound, axis=1)

# %%
# Display Results Summary
print("=" * 60)
print("ROBERTA SENTIMENT ANALYSIS RESULTS")
print("=" * 60)

print(f"\n📊 Total Posts Analyzed: {len(df)}")

print("\n" + "-" * 40)
print("Sentiment Distribution:")
print("-" * 40)
print(df['sentiment'].value_counts())

print(f"\nPercentages:")
print(df['sentiment'].value_counts(normalize=True).mul(100).round(1))

print(f"\nAverage Compound Score: {df['compound_score'].mean():.3f}")
print(f"Average Confidence: {df['confidence'].mean():.3f}")


