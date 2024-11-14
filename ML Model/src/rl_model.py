import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class StockPredictionEnv(gym.Env):
    def __init__(self, lstm_model, xgboost_model, nlp_model, historical_data, technical_indicators):
        super(StockPredictionEnv, self).__init__()
        
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.nlp_model = nlp_model

        self.historical_data = historical_data
        self.technical_indicators = technical_indicators
        
        # Define observation space based on the number of features in the state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Action space for weighting LSTM, XGBoost, and NLP predictions
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Assuming 3 weights

        self.current_step = 0

    def _get_state(self):
        lstm_prediction = self.lstm_model.predict(self.historical_data[self.current_step])
        xgboost_prediction = self.xgboost_model.predict(self.historical_data[self.current_step])
        nlp_sentiment = self.nlp_model.get_sentiment_score()  # Assuming this function is defined in nlp_model
        
        # Technical indicators
        volatility = calculate_volatility(self.historical_data[:self.current_step + 1])
        trend_slope = calculate_moving_average_slope(self.historical_data[:self.current_step + 1])

        # Rolling accuracies
        rolling_accuracy_xgboost = calculate_rolling_accuracy(self.xgboost_model, self.historical_data[:self.current_step + 1])
        rolling_accuracy_lstm = calculate_rolling_accuracy(self.lstm_model, self.historical_data[:self.current_step + 1])
        rolling_accuracy_nlp = calculate_rolling_accuracy(self.nlp_model, self.historical_data[:self.current_step + 1])

        # Compile the state as a dictionary
        state = np.array([
            xgboost_prediction,
            lstm_prediction,
            nlp_sentiment,
            volatility,
            trend_slope,
            rolling_accuracy_xgboost,
            rolling_accuracy_lstm,
            rolling_accuracy_nlp
        ])

        return state

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # Assign weights based on action
        weights = action / np.sum(action)  # Normalize weights to sum to 1

        lstm_prediction = self.lstm_model.predict(self.historical_data[self.current_step])
        xgboost_prediction = self.xgboost_model.predict(self.historical_data[self.current_step])
        nlp_sentiment = self.nlp_model.get_sentiment_score()

        # Weighted prediction
        weighted_prediction = weights[0] * xgboost_prediction + weights[1] * lstm_prediction + weights[2] * nlp_sentiment

        actual_price = self.historical_data[self.current_step + 1]  # Assume next price is actual
        reward = -abs(weighted_prediction - actual_price)  # Reward based on minimizing error

        self.current_step += 1
        done = self.current_step >= len(self.historical_data) - 1
        next_state = self._get_state() if not done else None

        return next_state, reward, done, {}
