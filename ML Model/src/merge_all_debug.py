import pandas as pd

from run_nlp import generate_daily_sentiment  # for NLP daily sentiment
from run_boost import xgboost_inference_df      # for XGBoost predictions
from run_lstm import generate_lstm_predictions    # for LSTM predictions

def normalize_date_column(df, date_col):
    """
    Convert the specified column to datetime and then normalize (set time to 00:00:00).
    Alternatively, you can convert to a Python date if preferred.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # Normalize so that all times are at midnight
    df[date_col] = df[date_col].dt.normalize()
    return df

def print_date_range(df, date_col, label):
    """Print the min and max dates for a given DataFrame column."""
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    print(f"{label} date range: {min_date} to {max_date}")

def main():
    # Define common parameters
    ticker = "TSLA"
    start_date = "2008-01-01"
    end_date = "2021-12-31"
    
    #########################
    # 1. NLP Sentiment Data #
    #########################
    print("Generating NLP sentiment data...")
    # Generate daily sentiment; expected columns: ['date', 'avg_sentiment']
    df_daily_sentiment = generate_daily_sentiment(ticker, start_date, end_date)
    
    # Rename the date column and normalize
    if 'date' in df_daily_sentiment.columns:
        df_daily_sentiment.rename(columns={'date': 'Date'}, inplace=True)
    df_daily_sentiment = normalize_date_column(df_daily_sentiment, 'Date')
    
    print("NLP Daily Sentiment (first 5 rows):")
    print(df_daily_sentiment.head())
    print_date_range(df_daily_sentiment, 'Date', "NLP Sentiment")
    
    ###############################
    # 2. XGBoost Prediction Data  #
    ###############################
    print("\nGenerating XGBoost prediction data...")
    best_features_TSLA = [
        'WCLPRICE', 'AROON_UP', 'MIDPOINT', 'TYPPRICE', 'MACD', 'BB_upper',
        'MACD_hist', 'T3', 'ADX', 'SMA', 'PLUS_DI', 'STOCH_fastk', 'MINUS_DM',
        'TEMA', 'ATR', 'STOCH_fastd', 'AROON_DOWN', 'BB_middle', 'NATR',
        'HT_LEADSINE', 'MFI', 'OBV', 'HT_PHASOR_inphase', 'STOCH_slowd'
    ]
    model_path_boost = '../tests/xgboost_TSLA.joblib'
    df_xgb = xgboost_inference_df(
        ticker=ticker,
        model_path=model_path_boost,
        start_date=start_date,
        end_date=end_date,
        feature_subset=best_features_TSLA
    )
    # Normalize the Date column for XGBoost DataFrame
    df_xgb = normalize_date_column(df_xgb, 'Date')
    
    print("XGBoost Predictions (first 5 rows):")
    print(df_xgb.head())
    print_date_range(df_xgb, 'Date', "XGBoost")
    
    ##########################
    # 3. LSTM Prediction Data#
    ##########################
    print("\nGenerating LSTM prediction data...")
    model_path_lstm = '../tests/lstm_TSLA_model.h5'
    df_lstm = generate_lstm_predictions(ticker, model_path_lstm, start_date=start_date, end_date=end_date)
    # Normalize the Date column for LSTM DataFrame
    df_lstm = normalize_date_column(df_lstm, 'Date')
    
    print("LSTM Predictions (first 5 rows):")
    print(df_lstm.head())
    print_date_range(df_lstm, 'Date', "LSTM")
    
    ##########################
    # 4. Merge All DataFrames #
    ##########################
    print("\nMerging data...")
    # Merge the XGBoost and LSTM predictions first on 'Date'
    df_merged = pd.merge(df_xgb, df_lstm, on='Date', how='inner', suffixes=('_xgb', '_lstm'))
    # Then merge the NLP sentiment data
    df_merged = pd.merge(df_merged, df_daily_sentiment, on='Date', how='inner')
    
    # Sort by date for clarity
    df_merged.sort_values(by='Date', inplace=True)
    
    print("\nMerged DataFrame (first 5 rows):")
    print(df_merged.head())
    print_date_range(df_merged, 'Date', "Merged")
    
    # Optionally, save the merged data for later use in your RL pipeline.
    output_file = "merged_rl_input.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"\nMerged data saved to {output_file}")