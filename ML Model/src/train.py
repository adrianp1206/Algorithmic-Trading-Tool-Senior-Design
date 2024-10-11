from data_processing import fetch_tsla_data, preprocess_data, create_lstm_input
from lstm_model import build_lstm_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_lstm_model():
    tsla_data = fetch_tsla_data('2008-01-01', '2021-12-31')
    tsla_data, scaler = preprocess_data(tsla_data)

    X, y = create_lstm_input(tsla_data, target_column='Adj Close', lookback=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.save("lstm_tsla_model_v3.h5")