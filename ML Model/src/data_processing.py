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

def fetch_last_month_data(ticker='TSLA', start_date='2024-09-01', end_date='2024-10-01'):
    """
    Fetch stock data for last week.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def fetch_data_prior_to_last_month(ticker='TSLA', start_date='2008-01-01', last_month_start='2024-09-01'):
    """
    Fetch stock data up to the start of the last month and include fundamental indicators.
    """
    # Download historical stock price data
    stock_data = yf.download(ticker, start=start_date, end=last_month_start)
    
    # Fetch fundamental data
    ticker_info = yf.Ticker(ticker)
    fundamentals = {
        'DE Ratio': ticker_info.info.get('debtToEquity', None),
        'Return on Equity': ticker_info.info.get('returnOnEquity', None),
        'Price/Book': ticker_info.info.get('priceToBook', None),
        'Profit Margin': ticker_info.info.get('profitMargins', None),
        'Diluted EPS': ticker_info.info.get('trailingEps', None),
        'Beta': ticker_info.info.get('beta', None)
    }
    
    # Create a DataFrame for fundamental data, repeated for each date in stock_data
    fundamentals_df = pd.DataFrame([fundamentals] * len(stock_data), index=stock_data.index)
    
    # Concatenate stock price data with the fundamental indicators
    combined_data = pd.concat([stock_data, fundamentals_df], axis=1)
    
    return combined_data

import talib as ta

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for stock data.
    
    Args:
    df: DataFrame, The stock data.
    
    Returns:
    df: DataFrame, The stock data with added indicators.
    """
    # Historical Prices are already in df (Open, High, Low, Close, Volume)

    # Overlap Studies Indicators
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['Close'], timeperiod=20)
    df['DEMA'] = ta.DEMA(df['Close'], timeperiod=30)
    df['MIDPOINT'] = ta.MIDPOINT(df['Close'], timeperiod=14)
    df['MIDPRICE'] = ta.MIDPRICE(df['High'], df['Low'], timeperiod=14)
    df['SMA'] = ta.SMA(df['Close'], timeperiod=30)
    df['T3'] = ta.T3(df['Close'], timeperiod=5, vfactor=0.7)
    df['TEMA'] = ta.TEMA(df['Close'], timeperiod=30)
    df['TRIMA'] = ta.TRIMA(df['Close'], timeperiod=30)
    df['WMA'] = ta.WMA(df['Close'], timeperiod=30)

    # Momentum Indicators
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADXR'] = ta.ADXR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['APO'] = ta.APO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
    df['AROON_DOWN'], df['AROON_UP'] = ta.AROON(df['High'], df['Low'], timeperiod=14)
    df['AROONOSC'] = ta.AROONOSC(df['High'], df['Low'], timeperiod=14)
    df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CMO'] = ta.CMO(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MFI'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    df['MINUS_DI'] = ta.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DM'] = ta.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MOM'] = ta.MOM(df['Close'], timeperiod=10)
    df['PLUS_DI'] = ta.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = ta.PLUS_DM(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ROC'] = ta.ROC(df['Close'], timeperiod=10)
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['STOCH_slowk'], df['STOCH_slowd'] = ta.STOCH(df['High'], df['Low'], df['Close'], 
                                                      fastk_period=5, slowk_period=3, slowk_matype=0, 
                                                      slowd_period=3, slowd_matype=0)
    df['STOCH_fastk'], df['STOCH_fastd'] = ta.STOCHF(df['High'], df['Low'], df['Close'], 
                                                     fastk_period=5, fastd_period=3, fastd_matype=0)
    
    # Volatility Indicators
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['NATR'] = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TRANGE'] = ta.TRANGE(df['High'], df['Low'], df['Close'])

    # Volume Indicators
    df['AD'] = ta.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    df['ADOSC'] = ta.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])

    # Price Transform Indicators
    df['AVGPRICE'] = ta.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'])
    df['MEDPRICE'] = ta.MEDPRICE(df['High'], df['Low'])
    df['TYPPRICE'] = ta.TYPPRICE(df['High'], df['Low'], df['Close'])
    df['WCLPRICE'] = ta.WCLPRICE(df['High'], df['Low'], df['Close'])

    # Cycle Indicators
    df['HT_DCPERIOD'] = ta.HT_DCPERIOD(df['Close'])
    df['HT_DCPHASE'] = ta.HT_DCPHASE(df['Close'])
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = ta.HT_PHASOR(df['Close'])
    df['HT_SINE'], df['HT_LEADSINE'] = ta.HT_SINE(df['Close'])
    df['HT_TRENDMODE'] = ta.HT_TRENDMODE(df['Close'])
    
    return df


