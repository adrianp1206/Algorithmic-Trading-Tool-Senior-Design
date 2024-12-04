import pandas as pd
import json
from scrape_data import fetch_data  # Import the fetch_data function

# Fetch the data
data = fetch_data()

if data:  # Ensure the data is not None
    # Extract the 'feed' section and convert it to a DataFrame
    articles = pd.DataFrame(data.get('feed', []))
    # Save the DataFrame to a CSV file
    articles.to_csv('tsla_news_sentiment.csv', index=False)
    print("Data saved to tsla_news_sentiment.csv")
else:
    print("No data to convert to CSV.")
