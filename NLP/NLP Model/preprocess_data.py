import pandas as pd
import ast  

def preprocess_csv(file_path):
    """Loads and preprocesses the CSV file."""
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Handle nested fields (e.g., ticker_sentiment) 
        def extract_ticker_sentiment(ticker_sentiment, target_ticker='TSLA'):
            """Extracts sentiment data for the target ticker."""
            try:
                sentiments = ast.literal_eval(ticker_sentiment)  # Convert stringified list/dict to Python object
                ticker_data = {}
                # Extract sentiment for all tickers
                for sentiment in sentiments:
                    ticker = sentiment.get('ticker')
                    if ticker:  # Ensure 'ticker' key is present
                        ticker_data[ticker] = {
                            'relevance_score': sentiment.get('relevance_score', None),
                            'sentiment_score': sentiment.get('ticker_sentiment_score', None),
                            'sentiment_label': sentiment.get('ticker_sentiment_label', None)
                        }
                return ticker_data
            except (ValueError, SyntaxError):
                return None

        # Extract all sentiment data for each row
        df['ticker_sentiment_data'] = df['ticker_sentiment'].apply(lambda x: extract_ticker_sentiment(x))

        # Filter rows where at least one ticker sentiment is not null
        df = df[df['ticker_sentiment_data'].notnull()]

        # Now extract the relevant data for TSLA (or any other ticker)
        df['relevance_score'] = df['ticker_sentiment_data'].apply(lambda x: x.get('TSLA', {}).get('relevance_score', None))
        df['sentiment_score'] = df['ticker_sentiment_data'].apply(lambda x: x.get('TSLA', {}).get('sentiment_score', None))
        df['sentiment_label'] = df['ticker_sentiment_data'].apply(lambda x: x.get('TSLA', {}).get('sentiment_label', None))

        # Optionally, extract other tickers' sentiment if needed (for example, for AAPL)
        # df['relevance_score_aapl'] = df['ticker_sentiment_data'].apply(lambda x: x.get('AAPL', {}).get('relevance_score', None))
        # df['sentiment_score_aapl'] = df['ticker_sentiment_data'].apply(lambda x: x.get('AAPL', {}).get('sentiment_score', None))
        # df['sentiment_label_aapl'] = df['ticker_sentiment_data'].apply(lambda x: x.get('AAPL', {}).get('sentiment_label', None))

        # Inspecting time_published values before conversion (optional debug)
        # print(df['time_published'].head(20))  

        # Convert time_published to a datetime format with error handling
        df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S', errors='coerce')

        # Check if there are any invalid dates
        invalid_dates = df[df['time_published'].isna()]
        if not invalid_dates.empty:
            print("Found invalid date formats:")
            print(invalid_dates[['time_published']])

        # Keep only the columns needed for analysis
        #'overall_sentiment_score', 'overall_sentiment_label', took them off for now cause we don't probably need it 
        cleaned_df = df[
            ['title', 'time_published', 
             'relevance_score', 'sentiment_score', 'sentiment_label', 'topics']
        ]

        # Save the cleaned data to a new CSV file
        cleaned_df.to_csv('tsla_news_cleaned.csv', index=False)
        print("Cleaned data saved to tsla_news_cleaned.csv")
        return cleaned_df

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    # Specify the input CSV file path
    input_csv_file = 'tsla_news_sentiment.csv'
    # Preprocess the CSV file
    preprocess_csv(input_csv_file)
