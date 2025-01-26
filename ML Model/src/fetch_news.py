import requests
import pandas as pd

# Your Polygon.io API key
API_KEY = 'BkBAQi6j6q46125xaQkX7TjBXn3es4Fi'

def fetch_articles(ticker, start_date, end_date):
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.gte={start_date}&published_utc.lte={end_date}&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('results', [])
    return articles

def extract_text(articles):
    text_data = []
    for article in articles:
        text_data.append(article.get('title', '') + " " + article.get('description', ''))
    return text_data
