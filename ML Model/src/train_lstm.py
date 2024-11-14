from data_processing import fetch_tsla_data, preprocess_data, create_lstm_input
from lstm_model import build_lstm_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

def train_lstm_model():
    # Fetch and preprocess data
    tsla_data = fetch_tsla_data('2008-01-01', '2023-12-31')
    tsla_data, scaler = preprocess_data(tsla_data)

    # Create model input data
    X, y = create_lstm_input(tsla_data, target_column='Adj Close', lookback=20)
    
    # Calculate the split indices
    train_size = int(0.7 * len(X))
    dev_size = int(0.15 * len(X))
    
    # Perform the splits: 70% training, 15% development, 15% testing
    X_train, y_train = X[:train_size], y[:train_size]
    X_dev, y_dev = X[train_size:train_size + dev_size], y[train_size:train_size + dev_size]
    X_test, y_test = X[train_size + dev_size:], y[train_size + dev_size:]

    # Define the input shape for the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Train the model using the training and development sets
    history = model.fit(X_train, y_train, epochs=90, batch_size=32, validation_data=(X_dev, y_dev))

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    model.save("lstm_tsla_model_rework_v11.h5")

    # Evaluate the model on the test set
    test_loss, test_mse = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    y_test_padded = np.zeros((len(y_test), scaler.min_.shape[0]))
    y_pred_padded = np.zeros((len(y_pred), scaler.min_.shape[0]))
    y_test_padded[:, 0] = y_test.flatten()
    y_pred_padded[:, 0] = y_pred.flatten()
    
    y_test_rescaled = scaler.inverse_transform(y_test_padded)[:, 0]
    y_pred_rescaled = scaler.inverse_transform(y_pred_padded)[:, 0]

    # Calculate accuracy metrics
    mape_test = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
    rmse_test = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    print(f"\nMean Absolute Percentage Error (MAPE) for Test Set: {mape_test * 100:.2f}%")
    print(f"Root Mean Squared Error (RMSE) for Test Set: {rmse_test:.2f}")

    # Plot the actual vs predicted prices for the test set
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Actual Prices', color='blue', marker='o')
    plt.plot(y_pred_rescaled, label='Predicted Prices', color='red', marker='x')
    plt.title('Predicted vs Actual Adjusted Close Prices for Test Set')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()