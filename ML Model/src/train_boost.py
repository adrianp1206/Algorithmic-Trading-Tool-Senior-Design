import xgboost as xgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from joblib import dump
from data_processing import fetch_stock_data, calculate_technical_indicators

def train_xgboost(
    ticker,
    start_date='2008-01-01',
    end_date='2021-12-31',
    feature_subset=None,
    n_splits=5,
    params=None,
    save_model=True
):
    """
    Train an XGBoost model using TimeSeriesSplit for a given stock ticker.

    Args:
    ticker: str, The stock ticker symbol.
    start_date: str, Start date for training data.
    end_date: str, End date for training data.
    feature_subset: list of str, The selected features to use for training.
    n_splits: int, Number of splits for TimeSeriesSplit.
    params: dict, Hyperparameters for XGBoost.
    save_model: bool, If True, save the final model.

    Returns:
    metrics: dict, Performance metrics across splits.
    model: Trained XGBoost model.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Default parameters if none provided
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    
    # Fetch and preprocess data
    data = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    data = calculate_technical_indicators(data)
    data['Price_Change'] = data['Close'].diff()
    data['Target'] = (data['Price_Change'] > 0).astype(int)
    data = data.dropna()

    X = data[feature_subset] if feature_subset else data.drop(columns=['Price_Change', 'Target'])
    y = data['Target']

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Metrics
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    models = []

    # Train and evaluate across splits
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred, zero_division=0))
        recall_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        roc_auc_list.append(roc_auc_score(y_test, y_pred_proba))
        models.append(model)

    metrics = {
        'accuracy': np.mean(accuracy_list),
        'precision': np.mean(precision_list),
        'recall': np.mean(recall_list),
        'f1_score': np.mean(f1_list),
        'roc_auc': np.mean(roc_auc_list),
    }

    # Save the model if flag is True
    if save_model:
        model_filename = f"xgboost_{ticker}.joblib"
        dump(models[-1], model_filename)
        print(f"Model saved to {model_filename}")

        # Display confusion matrix for the final split
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(cmap="Blues")
        plt.title(f"Confusion Matrix for {ticker}")
        plt.show()

    return metrics, models[-1]