import torch
import pandas as pd
from datetime import datetime, timedelta

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
    articles = fetch_articles(ticker, start_date, end_date)
    print(articles)  # returns a list of article dicts

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
    Fetch articles, compute daily average sentiment, then calculate a 5-day rolling average.
    Returns a DataFrame with columns: [Date, avg_sentiment]
    """
    # Fetch all sentiment data for the given period
    df_full = generate_sentiment_df(ticker, start_date, end_date)
    
    # Debug: Print out the raw fetched data to check date range
    print("Fetched articles (raw):")
    print(df_full[['published_date', 'sentiment_score']].head(10))
    
    # Convert published_date to a date-only column
    df_full['Date'] = pd.to_datetime(df_full['published_date'], errors='coerce').dt.date

    # Compute the daily average sentiment score from all articles on that day
    df_daily = (
        df_full.groupby('Date')['sentiment_score']
        .mean()
        .reset_index()
        .rename(columns={'sentiment_score': 'daily_sentiment'})
    )
    
    # Convert 'Date' back to datetime for proper sorting and rolling window calculations
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily.sort_values('Date', inplace=True)
    
    # Calculate a rolling average over the last 5 days (including the current day)
    df_daily['avg_sentiment'] = df_daily['daily_sentiment'].rolling(window=5, min_periods=1).mean()
    
    # Return only the Date and the computed rolling average
    return df_daily[['Date', 'avg_sentiment']]


def generate_next_day_rolling_sentiments(ticker, start_date, end_date, window=5):
    """
    Generate a DataFrame where for each day (starting when a full window is available)
    the "next_day_sentiment" is computed as the average of the raw daily sentiment scores
    from the previous `window` days.
    
    For example:
      - If January 1–5 have raw sentiment values that average to 0.75, then January 6's
        next_day_sentiment will be 0.75.
      - January 7's next_day_sentiment will be the average of January 2–6 (using the raw data,
        not any rolling average that was computed for January 6).
    
    Parameters:
      ticker: Stock ticker symbol.
      start_date: Start date as a string in 'YYYY-MM-DD' format.
      end_date: End date as a string in 'YYYY-MM-DD' format.
      window: Number of days to use for computing the next day sentiment (default is 5).
      
    Returns:
      A DataFrame with columns [Date, next_day_sentiment].
    """
    # Fetch all articles for the period
    articles = fetch_articles(ticker, start_date, end_date)
    
    records = []
    for article in articles:
        published_utc = article.get('published_utc', '')
        try:
            pub_date = pd.to_datetime(published_utc).date()
        except Exception:
            continue
        
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title} {description}"
        sentiment_data = analyze_text_finbert(text)
        score = sentiment_data.get('sentiment_score', 0)
        
        records.append({
            'published_date': pub_date,
            'sentiment_score': score
        })
    
    if not records:
        print("No articles found in this period.")
        return None

    # Create a DataFrame with raw daily sentiment values (averaging all articles for each day)
    df = pd.DataFrame(records)
    daily_sentiment = (
        df.groupby('published_date')['sentiment_score']
          .mean()
          .reset_index()
          .rename(columns={'published_date': 'Date', 'sentiment_score': 'daily_sentiment'})
    )
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    
    # Create a complete date range DataFrame
    full_dates = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date)
    })
    
    # Merge to ensure every day in the range is present (fill days with no articles with 0)
    merged = full_dates.merge(daily_sentiment, on='Date', how='left')
    merged['daily_sentiment'] = merged['daily_sentiment'].fillna(0)
    
    # Compute a rolling average on the raw daily sentiment with the specified window.
    # This rolling average gives the average over a block of 5 days.
    # Then shift it forward by 1 day so that for day T, we use the previous 5 days (T-5 to T-1).
    merged['next_day_sentiment'] = merged['daily_sentiment'] \
        .rolling(window=window, min_periods=window).mean().shift(1)
    
    # Drop rows where we don't have a full 5-day history
    result = merged.dropna(subset=['next_day_sentiment']).reset_index(drop=True)
    
    return result[['Date', 'next_day_sentiment']]