import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from data_processing import fetch_tsla_data, calculate_technical_indicators

def prepare_data(start_date='2008-01-01', end_date='2024-01-01'):
    """
    Prepare the feature matrix X and target variable y for model training.
    
    Args:
    start_date: str, The start date for fetching data.
    end_date: str, The end date for fetching data.
    
    Returns:
    X: DataFrame, Feature matrix containing technical indicators.
    y: Series, Target variable representing stock price movement (1 for up, 0 for down).
    """
    # Step 1: Fetch the historical stock data
    df = fetch_tsla_data(start_date, end_date)
    
    # Step 2: Add technical indicators
    df = calculate_technical_indicators(df)
    
    # Step 3: Define the target variable y
    # Here we assume we want to predict if the price goes up (1) or down (0)
    df['Price_Change'] = df['Close'].diff()  # Calculate daily price change
    df['Target'] = (df['Price_Change'] > 0).astype(int)  # 1 if up, 0 if down
    
    # Drop the first row because diff() will produce NaN for the first day
    df = df.dropna().reset_index(drop=True)
    
    # Step 4: Prepare the feature matrix X
    X = df.drop(columns=['Price_Change', 'Target'])  # Drop non-feature columns
    y = df['Target']  # Target variable
    
    return X, y

def train_xgboost_with_timeseries_cv(X, y, feature_subset, params=None, n_splits=5):
    """
    Train an XGBoost model using TimeSeriesSplit for cross-validation.
    
    Args:
    X: DataFrame, The complete feature set.
    y: Series or array, The target variable.
    feature_subset: List of str, The selected features to train on.
    params: dict, Dictionary of hyperparameters to pass to XGBClassifier.
    n_splits: int, Number of splits for TimeSeriesSplit.
    
    Returns:
    avg_accuracy: float, The average accuracy score across splits.
    models: list, The trained XGBoost models for each split.
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 50,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    
    # Prepare the subset of features
    X_subset = X[feature_subset]
    
    # TimeSeriesSplit initialization
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies = []
    models = []
    
    # Time series cross-validation
    for train_index, test_index in tscv.split(X_subset):
        X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Set up the XGBoost model with parameters
        model = xgb.XGBClassifier(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and evaluate accuracy on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        models.append(model)
    
    # Calculate average accuracy across splits
    avg_accuracy = np.mean(accuracies)
    print(f"Average Accuracy across {n_splits} splits: {avg_accuracy:.4f}")
    
    return avg_accuracy, models
