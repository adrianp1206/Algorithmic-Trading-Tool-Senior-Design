from data_processing import fetch_tsla_data, preprocess_data, create_lstm_input
from lstm_model import build_lstm_model
import matplotlib.pyplot as plt
import numpy as np

def train_lstm_model():
    # Fetch and preprocess data
    tsla_data = fetch_tsla_data('2008-01-01', '2021-12-31')
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
    history = model.fit(X_train, y_train, epochs=100, batch_size=30, validation_data=(X_dev, y_dev))

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    model.save("lstm_tsla_model_rework_v5.h5")

    # Optional: Evaluate the model on the test set and print results
    test_loss, test_mse = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

train_lstm_model()
