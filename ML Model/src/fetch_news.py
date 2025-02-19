import requests
import pandas as pd

# Your Polygon.io API key
API_KEY = 'BkBAQi6j6q46125xaQkX7TjBXn3es4Fi'

def fetch_articles(ticker, start_date, end_date):
    base_url = (
        f"https://api.polygon.io/v2/reference/news?"
        f"ticker={ticker}&published_utc.gte={start_date}&published_utc.lte={end_date}"
        f"&limit=1000&apiKey={API_KEY}"
    )
    all_articles = []
    url = base_url

    while url:
        response = requests.get(url)
        data = response.json()

        # Append articles from the current page
        articles = data.get('results', [])
        all_articles.extend(articles)
        
        # Retrieve the next_url from the response
        next_url = data.get('next_url')
        if next_url:
            # If the API key isn't present in the next_url, append it
            if "apiKey=" not in next_url:
                next_url += f"&apiKey={API_KEY}"
        url = next_url  # This will be None when there are no further pages

    # Print total count of articles fetched
    print("Total number of articles fetched:", len(all_articles))
    
    return all_articles

def extract_text(articles):
    text_data = []
    for article in articles:
        text_data.append(article.get('title', '') + " " + article.get('description', ''))
    return text_data
