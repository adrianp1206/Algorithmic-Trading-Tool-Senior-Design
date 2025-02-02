import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# If you have your own data fetching & processing:
from data_processing import fetch_stock_data, calculate_technical_indicators

def xgboost_inference_df(
    ticker,
    model_path,
    start_date='2008-01-01',
    end_date='2022-12-31',
    feature_subset=None
):
    """
    Generate a DataFrame of XGBoost predictions for a given ticker
    across a specified date range.

    Returns
    -------
    df_predictions : pd.DataFrame
        Columns: [Date, Close, XGB_Pred, XGB_Prob_Up (optional), ...]
    """

    # 1. Fetch raw data (Replace fetch_stock_data with your actual data fetch function)
    df_raw = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    # Example columns in df_raw: ['Open','High','Low','Close','Adj Close','Volume', ...]

    # 2. Calculate technical indicators (or any other feature engineering)
    df_features = calculate_technical_indicators(df_raw)
    # Make sure df_features includes all columns your XGBoost expects.

    # 3. Possibly define a feature subset or keep them all
    if feature_subset is None:
        # For example, exclude target columns from your features if you had them
        # (e.g., 'Price_Change', 'Target')
        exclude_cols = ['Price_Change', 'Target', 'Date']  # or anything you don't want
        feature_subset = [col for col in df_features.columns if col not in exclude_cols]
    else:
        # user-supplied feature_subset
        pass

    # 4. Drop NaN rows if they exist (due to moving averages, etc.)
    df_features = df_features.dropna(subset=feature_subset)
    # Keep track of the final index (dates) after dropping
    valid_index = df_features.index

    # 5. Create the input matrix X for inference
    X_inference = df_features[feature_subset]

    # 6. Load your trained XGBoost model
    xgb_model = load(model_path)

    # 7. Generate predictions
    # If it's a classifier that outputs up/down:
    #   y_pred = xgb_model.predict(X_inference)
    #   y_proba = xgb_model.predict_proba(X_inference)[:, 1]  # Probability of "Up"

    # Example: let's store both predicted label and probability
    y_pred_labels = xgb_model.predict(X_inference)
    # If you only trained a binary classification, we can get probability of positive class:
    try:
        y_pred_proba = xgb_model.predict_proba(X_inference)[:, 1]  # Probability of "Up"
    except AttributeError:
        # If it's a regressor or the model doesn't have predict_proba, skip this
        y_pred_proba = None

    # 8. Re-combine predictions with the original Close prices
    df_predictions = pd.DataFrame({
        'Date': valid_index,  # index after dropping NaNs
        'Close': df_raw['Close'].loc[valid_index],
        'XGB_Pred': y_pred_labels
    })
    # If you do have probabilities, add them as well:
    if y_pred_proba is not None:
        df_predictions['XGB_Prob_Up'] = y_pred_proba

    # Reset index so Date becomes a column
    df_predictions = df_predictions.reset_index(drop=True)

    # 9. Return the final DataFrame
    return df_predictions

