import yfinance as yf
import pandas as pd 
import os

def fetch_tsla_data(start_date='2015-01-01', end_date='2024-01-01'):
    """
    Fetch historical stock data for TSLA from Yahoo Finance.
    
    Args:
    start_date: str, The start date for the data fetch.
    end_date: str, The end date for the data fetch.
    
    Returns:
    tsla_data: DataFrame, The historical stock data for TSLA.
    """
    tsla_data = yf.download('TSLA', start='2008-01-01', end='2021-12-31')
    
    # Fetch additional fundamental data
    tsla = yf.Ticker('TSLA')
    fundamentals = {
        "DE Ratio": tsla.info.get("debtToEquity"),
        "Return on Equity": tsla.info.get("returnOnEquity"),
        "Price/Book": tsla.info.get("priceToBook"),
        "Profit Margin": tsla.info.get("profitMargins"),
        "Diluted EPS": tsla.info.get("trailingEps"),
        "Beta": tsla.info.get("beta")
    }

    # Add fundamentals as static values for the whole period if no historical data is available
    for key, value in fundamentals.items():
        tsla_data[key] = value

    return tsla_data

def calculate_technical_indicators(df):
    """
    Calculate common technical indicators using TA-Lib.
    
    Args:
    df: DataFrame, The stock data.
    
    Returns:
    df: DataFrame, The stock data with added indicators.
    """
    return df

def save_data_to_csv(df, filename='tsla_data.csv'):
    """
    Save the stock data to a CSV file.
    
    Args:
    df: DataFrame, The data to save.
    filename: str, The name of the file to save the data to.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename)
    print(f"Data saved to {filename}")

def load_data_from_csv(filename='tsla_data.csv'):
    """
    Load stock data from a CSV file.
    
    Args:
    filename: str, The name of the CSV file.
    
    Returns:
    df: DataFrame, The loaded data.
    """
    return pd.read_csv(filename)

import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def create_lstm_input(data, target_column='adj_close', lookback=20):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data.iloc[i-lookback:i].values)
        y.append(data.iloc[i][target_column]) 
    return np.array(X), np.array(y)

def preprocess_data(data):
    """
    Preprocess the stock data by handling missing values and scaling the data.
    
    Args:
    df: DataFrame, The raw stock data.
    
    Returns:
    df: DataFrame, The preprocessed data.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Adj Close', 'Volume', 'DE Ratio', 'Return on Equity', 
                                             'Price/Book', 'Profit Margin', 'Diluted EPS', 'Beta']])
    return pd.DataFrame(scaled_data, columns=['Adj Close', 'Volume', 'DE Ratio', 'Return on Equity', 
                                              'Price/Book', 'Profit Margin', 'Diluted EPS', 'Beta']), scaler

def fetch_data_up_to_last_week(ticker='TSLA', start_date='2023-01-01', end_date='2024-09-27'):
    """
    Fetch stock data up to the date just before last week.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    ticker_info = yf.Ticker(ticker)
    fundamentals = {
        'DE Ratio': ticker_info.info.get('debtToEquity', None),
        'Return on Equity': ticker_info.info.get('returnOnEquity', None),
        'Price/Book': ticker_info.info.get('priceToBook', None),
        'Profit Margin': ticker_info.info.get('profitMargins', None),
        'Diluted EPS': ticker_info.info.get('trailingEps', None),
        'Beta': ticker_info.info.get('beta', None)
    }
    
    # Create a DataFrame with the fundamental data and repeat it for each date
    fundamentals_df = pd.DataFrame([fundamentals] * len(stock_data), index=stock_data.index)
    
    # Merge stock price data with fundamental data
    combined_data = pd.concat([stock_data, fundamentals_df], axis=1)
    
    return combined_data

def fetch_last_week_data(ticker='TSLA', start_date='2024-09-30', end_date='2024-10-05'):
    """
    Fetch stock data for last week.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

