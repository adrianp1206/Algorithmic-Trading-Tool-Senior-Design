import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from data_processing import fetch_stock_data, preprocess_data, create_lstm_input
from lstm_model import Attention

def generate_lstm_predictions(ticker, model_path, start_date, end_date):
    """
    Generate LSTM predictions for a given ticker and date range using a saved model.
    
    Returns a DataFrame with columns:
    [Date, Close, LSTM_Predicted]
    """
    # 1) Fetch the data (e.g., via your fetch_stock_data function)
    #    This should include 'Adj Close' in the DataFrame.
    df_raw = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    
    # If fetch_stock_data returns columns like ['Open','High','Low','Close','Adj Close','Volume',...],
    # make sure you have 'Adj Close' for your LSTM's target.

    # 2) Preprocess data (scaling) - returns (scaled_df, scaler)
    #    scaled_df typically has columns in the order your model expects
    scaled_df, scaler = preprocess_data(df_raw)
    # The output 'scaled_df' might have columns in this order (example):
    # ['Adj Close','Volume','DE Ratio','Return on Equity','Price/Book',
    #  'Profit Margin','Diluted EPS','Beta']
    
    # 3) Create sequences for LSTM (X, y)
    #    For example: create_lstm_input(scaled_data, target_column='Adj Close', lookback=20)
    #    But note: inside `scaled_df`, the column name might be exactly 'Adj Close' or index 0.
    X, y = create_lstm_input(scaled_df, target_column='Adj Close', lookback=20)
    
    # 4) Load the pre-trained model
    model = load_model(model_path, custom_objects={'Attention': Attention})

    # 5) Generate predictions
    y_pred_scaled = model.predict(X)
    # shape of y_pred_scaled: (num_samples, 1)

    # 6) Inverse transform predictions to original price scale
    #    We'll do a small trick: create an array with the same shape as scaled_df
    #    so we can apply the scaler's inverse_transform properly.
    #    Then we'll extract just the 'Adj Close' column from the inverse.
    
    # We need the same # of rows for inverse transform:
    # But X has shape (num_samples, lookback, num_features), so the # of samples is different from scaled_df.
    # We typically have "len(scaled_df) - lookback" rows of predictions.
    
    # Also, 'scaled_df' presumably has shape (num_days, num_features).
    # If we only want to invert 'Adj Close', we can do something like:
    
    num_samples = len(y_pred_scaled)
    
    # Create a placeholder with the same number of columns as scaled_df
    # (ex: 8 columns if your scaled_df has 8 features).
    # We'll fill only the first column (which we assume is 'Adj Close') with y_pred_scaled.
    # Everything else can be zero or any placeholder. Then apply inverse_transform.
    
    placeholder = np.zeros((num_samples, scaled_df.shape[1]))
    placeholder[:, 0] = y_pred_scaled.flatten()  # assume 'Adj Close' is in column index 0
    y_pred_inversed_all = scaler.inverse_transform(placeholder)
    # Now y_pred_inversed_all[:,0] is the predicted price in original scale
    
    y_pred_inversed = y_pred_inversed_all[:, 0]  # just that first column

    # Similarly, if you want the actual 'y' in original scale:
    placeholder_y = np.zeros((num_samples, scaled_df.shape[1]))
    placeholder_y[:, 0] = y[:num_samples]  # y has the same length as X
    y_inversed_all = scaler.inverse_transform(placeholder_y)
    y_inversed = y_inversed_all[:, 0]
    
    # 7) Align predictions with actual dates
    #    If you created input sequences with a lookback of 20,
    #    your first prediction corresponds to day index 20 in df_raw (the 21st row),
    #    your last prediction corresponds to the final row of df_raw.

    # We can get that subset of the original df_raw's dates:
    df_dates = df_raw.iloc[20:].index  # e.g. from row 20 to the end

    df_pred = pd.DataFrame({
        'Date': df_dates,
        'Close': df_raw['Adj Close'].iloc[20:].values,  # actual Adj Close aligned
        'LSTM_Predicted': y_pred_inversed
    })
    # Convert Date from index to column if needed
    df_pred = df_pred.reset_index(drop=True)

    return df_pred