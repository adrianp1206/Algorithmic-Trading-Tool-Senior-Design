

def split_data_xgboost(data):
    """
    Splits the data into training, validation, and test sets for XGBoost classification and regression tasks.
    """
    # Prepare features and targets
    features = data.drop(columns=['Prediction', 'Adj Close']).values
    target_classification = data['Prediction'].values  # For classification (direction)
    target_regression = data['Adj Close'].diff().shift(-1).dropna().values  # For regression (magnitude)

    # Ensure features and target arrays align properly in length (due to shifting)
    features = features[:-1]
    target_classification = target_classification[:-1]
    target_regression = target_regression

    # Split dataset as per the paper (85% train, 15% validation)
    # Train and Validation split (85% for training and 15% for validation)
    train_size = int(0.85 * len(features))
    X_train, X_val = features[:train_size], features[train_size:]
    y_class_train, y_class_val = target_classification[:train_size], target_classification[train_size:]
    y_reg_train, y_reg_val = target_regression[:train_size], target_regression[train_size:]

    # Holdout test set (last 20% of all data)
    holdout_size = int(0.2 * len(features))
    X_holdout, y_class_holdout, y_reg_holdout = features[-holdout_size:], target_classification[-holdout_size:], target_regression[-holdout_size:]

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_holdout': X_holdout,
        'y_class_train': y_class_train,
        'y_class_val': y_class_val,
        'y_class_holdout': y_class_holdout,
        'y_reg_train': y_reg_train,
        'y_reg_val': y_reg_val,
        'y_reg_holdout': y_reg_holdout
    }
