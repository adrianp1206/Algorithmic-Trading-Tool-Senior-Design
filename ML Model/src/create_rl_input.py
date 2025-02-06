import pandas as pd

from run_nlp import generate_daily_sentiment  # for NLP daily sentiment
from run_boost import xgboost_inference_df      # for XGBoost predictions
from run_lstm import generate_lstm_predictions    # for LSTM predictions

def create_input_TSLA():
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
    
    # Ensure the date column is in datetime format and rename to 'Date'
    if 'date' in df_daily_sentiment.columns:
        df_daily_sentiment.rename(columns={'date': 'Date'}, inplace=True)
    df_daily_sentiment['Date'] = pd.to_datetime(df_daily_sentiment['Date'])
    
    print("NLP Daily Sentiment (first 5 rows):")
    print(df_daily_sentiment.head())
    
    ###############################
    # 2. XGBoost Prediction Data  #
    ###############################
    print("\nGenerating XGBoost prediction data...")
    # For TSLA, use your predefined best features and model path
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
    # Ensure the Date column is in datetime format
    df_xgb['Date'] = pd.to_datetime(df_xgb['Date'])
    
    print("XGBoost Predictions (first 5 rows):")
    print(df_xgb.head())
    
    ##########################
    # 3. LSTM Prediction Data#
    ##########################
    print("\nGenerating LSTM prediction data...")
    model_path_lstm = '../tests/lstm_TSLA_model.h5'
    df_lstm = generate_lstm_predictions(ticker, model_path_lstm, start_date=start_date, end_date=end_date)
    # Assume df_lstm has a column 'Date' with date information.
    df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])
    
    print("LSTM Predictions (first 5 rows):")
    print(df_lstm.head())
    
    ##########################
    # 4. Merge All DataFrames #
    ##########################
    print("\nMerging data...")
    # Merge the XGBoost and LSTM predictions first on 'Date'
    df_merged = pd.merge(df_xgb, df_lstm, on='Date', how='inner')
    # Then merge the NLP sentiment data
    df_merged = pd.merge(df_merged, df_daily_sentiment, on='Date', how='inner')
    
    # Sort by date for clarity
    df_merged.sort_values(by='Date', inplace=True)
    
    print("Merged DataFrame (first 5 rows):")
    print(df_merged.head())
    
    # Optionally, save the merged data for later use in your RL pipeline.
    output_file = "merged_rl_input.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"\nMerged data saved to {output_file}")