import pandas as pd

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

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_xgboost(X, y, feature_subset, test_size=0.2, random_state=42):
    """
    Train an XGBoost model on a subset of features and evaluate its performance.
    
    Args:
    X: DataFrame, The complete feature set.
    y: Series or array, The target variable.
    feature_subset: List of str, The selected features to train on.
    test_size: float, Proportion of the dataset to include in the test split.
    random_state: int, Random seed for reproducibility.
    
    Returns:
    model: XGBClassifier, The trained XGBoost model.
    accuracy: float, The accuracy score on the test set.
    """
    # Subset the data with the selected features
    X_subset = X[feature_subset]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=test_size, random_state=random_state)
    
    # Set up the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate accuracy on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

