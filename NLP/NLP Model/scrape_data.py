import requests
import json

API_KEY = 'ZZTG2ZFOL261GFXG'
BASE_URL = 'https://www.alphavantage.co/query'

params = {
    'function': 'NEWS_SENTIMENT',
    'tickers': 'TSLA',  # Ticker we are targeting for 
    'limit': 1000,       # Maximum number of articles we want to search for 
    'apikey': API_KEY
}

def fetch_data():
    """Fetches data from the API and saves it as a JSON file."""
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        # Save JSON data to a file
        with open('tsla_news.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("Data saved to tsla_news.json")
        return data  # Return the data for use in other scripts
    else:
        print(f"Error: Unable to fetch data (status code: {response.status_code})")
        return None

if __name__ == "__main__":
    # Fetch and display data only when running this script directly
    data = fetch_data()
    if data:
        for article in data.get('feed', []):
            print(f"Headline: {article['title']}")
            print(f"Published: {article['time_published']}")
            print(f"Sentiment: {article['overall_sentiment']}")
            print(f"Source: {article['source']}")
            print("-" * 40)
