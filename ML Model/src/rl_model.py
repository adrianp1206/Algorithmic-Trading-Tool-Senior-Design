import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class StockPredictionEnv(gym.Env):
    def __init__(self, lstm_model, xgboost_model, historical_data, features):
        """
        Custom stock prediction environment for RL using LSTM and XGBoost predictions.

        Args:
        lstm_model: Trained LSTM model.
        xgboost_model: Trained XGBoost model.
        historical_data: DataFrame, historical stock prices with technical indicators.
        features: List of str, features to be used for predictions.
        """
        super(StockPredictionEnv, self).__init__()

        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.historical_data = historical_data
        self.features = features

        # Observation space: State features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # Matches the state vector size
            dtype=np.float32
        )


        # Action space: Buy (1) or Sell (-1)
        self.action_space = spaces.Discrete(2)  # 0: Sell, 1: Buy

        self.current_step = 0
        self.cash = 0
        self.position = 0  # +1 for long, -1 for short

    def _get_state(self):
        if self.current_step < 20:
            state = np.zeros(7)  # Matches the state vector size
            return state # Add 4 for other indicators like cash/position

        # LSTM Prediction (raw time-series data)
        X_lstm = self.historical_data['Close'].iloc[self.current_step - 19:self.current_step + 1].values
        X_lstm = X_lstm.reshape(1, 20, -1)
        lstm_prediction = self.lstm_model.predict(X_lstm)[0][0]

        # XGBoost Prediction (dynamic feature set)
        X_xgboost = self.historical_data[self.features].iloc[self.current_step].values.reshape(1, -1)
        xgboost_prediction = self.xgboost_model.predict(X_xgboost)[0]

        lstm_rolling_accuracy = calculate_rolling_accuracy(
            self.lstm_model,
            self.historical_data,
            ['Close'],  # Feature(s) used for LSTM (adjust if more features are used)
            self.current_step
        )

    # Rolling accuracy for XGBoost
        xgboost_rolling_accuracy = calculate_rolling_accuracy(
            self.xgboost_model,
            self.historical_data,
            self.features,  # Feature(s) used for XGBoost
            self.current_step
        )

        # Additional indicators
        volatility = calculate_volatility(self.historical_data['Close'][:self.current_step + 1])
        trend_slope = calculate_moving_average_slope(self.historical_data['Close'][:self.current_step + 1])

        # State vector
        state = np.array([
            xgboost_prediction,
            lstm_prediction,
            volatility,
            trend_slope,
            self.cash,
            self.position,
            (lstm_rolling_accuracy + xgboost_rolling_accuracy) / 2 
        ])

        return state

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.current_step = 0
        self.cash = 0
        self.position = 0
        return self._get_state()

    def step(self, action):
        """
        Take a step in the environment.

        Args:
        action: int, 0 for Sell, 1 for Buy.

        Returns:
        next_state: np.array, next state representation.
        reward: float, reward for the current step.
        done: bool, whether the episode is finished.
        info: dict, additional info (optional).
        """
        # Current price and next day's price
        current_price = self.historical_data['Close'].iloc[self.current_step]
        next_price = self.historical_data['Close'].iloc[self.current_step + 1]

        # Update position and cash
        if action == 1:  # Buy
            self.position += 1
            self.cash -= current_price
        elif action == 0:  # Sell
            self.position -= 1
            self.cash += current_price

        # Calculate profit/loss for this step
        profit_loss = (next_price - current_price) * self.position

        # Add rewards for closing profitable positions
        if self.position == 0 and profit_loss > 0:
            reward = profit_loss / current_price + 0.01  # Bonus for profitable exits
        else:
            reward = profit_loss / current_price

        # Add penalties for holding losing positions
        if profit_loss < 0:
            reward += -0.01 * abs(self.position)  # Penalty for holding losses

        # Normalize the reward
        reward = np.clip(reward, -10, 10)  # Avoid extreme reward values

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.historical_data) - 2
        next_state = self._get_state() if not done else None

        return next_state, reward, done, {}


def calculate_volatility(prices, window=20):
    """
    Calculate rolling volatility (standard deviation) over a given window.

    Args:
    prices: pd.Series, historical stock prices.
    window: int, the rolling window size.

    Returns:
    float, rolling volatility.
    """
    if len(prices) < window:
        return 0
    return np.std(prices[-window:])  # Rolling standard deviation for the last `window` days

def calculate_moving_average_slope(prices, window=20):
    """
    Calculate the slope of a rolling moving average over a given window.

    Args:
    prices: pd.Series, historical stock prices.
    window: int, the rolling window size.

    Returns:
    float, the slope of the moving average.
    """
    if len(prices) < window + 1:
        return 0
    moving_avg = prices.rolling(window=window).mean()
    recent_avg = moving_avg.iloc[-1]
    previous_avg = moving_avg.iloc[-2]
    return (recent_avg - previous_avg) / previous_avg if previous_avg != 0 else 0

def calculate_rolling_accuracy(model, historical_data, features, step, window=20):
    """
    Calculate rolling accuracy for a given model over a rolling window.

    Args:
    model: Trained model (e.g., LSTM or XGBoost).
    historical_data: pd.DataFrame, input data for predictions.
    features: list, feature columns used for predictions.
    step: int, current step in the environment.
    window: int, the rolling window size.

    Returns:
    float, rolling accuracy of the model.
    """
    if step < window:
        return 0  # Not enough data to calculate rolling accuracy
    X_window = historical_data[features].iloc[step - window:step].values
    y_window = historical_data['Target'].iloc[step - window:step].values
    predictions = model.predict(X_window)
    return np.mean(predictions == y_window)

def add_target_column(data):
    """
    Adds a 'Target' column to indicate the price movement direction.

    Args:
    data: pd.DataFrame, historical stock data.

    Returns:
    pd.DataFrame, updated data with 'Target' column.
    """
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    return data