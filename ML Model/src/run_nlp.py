import torch
import pandas as pd

from fetch_news import fetch_articles, extract_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT model & tokenizer once at import time
# (so you don't re-download or re-load for every function call)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def analyze_text_finbert(text):
    """
    Given a single text (string), return sentiment details using FinBERT.
    Returns a dict with keys: 
      { 'positive', 'neutral', 'negative', 'sentiment_score' }
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # scores[0] = positive, scores[1] = neutral, scores[2] = negative
    positive = scores[0].item()
    neutral = scores[1].item()
    negative = scores[2].item()

    # Single sentiment score: positive - negative (range roughly [-1, +1])
    sentiment_score = positive - negative
    
    return {
        'positive': positive,
        'neutral': neutral,
        'negative': negative,
        'sentiment_score': sentiment_score
    }

def generate_sentiment_df(ticker, start_date, end_date):
    """
    Fetch articles via fetch_articles, run FinBERT sentiment on each article,
    and return a DataFrame with columns:
      ['published_date', 'title', 'description', 
       'positive', 'neutral', 'negative', 'sentiment_score'].

    You can group or average by published_date if you want daily sentiment.
    """

    # 1) Fetch articles
    articles = fetch_articles(ticker, start_date, end_date)  # returns a list of article dicts

    # 2) Build a list of dicts containing text + metadata
    records = []
    for article in articles:
        published_date = article.get('published_utc', '')  # e.g. "2024-01-05T12:30:00Z"
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title} {description}"

        # 3) Run sentiment analysis
        scores = analyze_text_finbert(text)

        record = {
            'published_date': published_date,
            'title': title,
            'description': description,
            'positive': scores['positive'],
            'neutral': scores['neutral'],
            'negative': scores['negative'],
            'sentiment_score': scores['sentiment_score']
        }
        records.append(record)

    # 4) Create DataFrame
    df = pd.DataFrame(records)

    # Convert published_date to datetime if you want to group by date
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    # You might drop rows with invalid date
    df.dropna(subset=['published_date'], inplace=True)

    # 5) Sort by date
    df.sort_values(by='published_date', inplace=True)

    return df

# Optional: If you want a separate function to produce a daily average:
def generate_daily_sentiment(ticker, start_date, end_date):
    """
    Same as above, but returns a DataFrame of daily average sentiment_score.
    Columns: [date, avg_sentiment]
    """
    df_full = generate_sentiment_df(ticker, start_date, end_date)
    # Create a column for date only
    df_full['date'] = df_full['published_date'].dt.date

    df_daily = (df_full.groupby('date')['sentiment_score']
                .mean()
                .reset_index()
                .rename(columns={'sentiment_score': 'avg_sentiment'}))

    return df_daily