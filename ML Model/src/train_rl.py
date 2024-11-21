import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
from stable_baselines3 import PPO
from rl_model import StockPredictionEnv
from data_processing import fetch_stock_data, calculate_technical_indicators
from lstm_model import Attention

def evaluate_rl_model(env, model, n_episodes=10):
    """
    Evaluate the RL model on the environment.

    Args:
    env: Gym environment.
    model: Trained RL model.
    n_episodes: Number of episodes for evaluation.

    Returns:
    dict: Evaluation metrics including average reward and action counts.
    """
    total_rewards = []
    action_counts = {0: 0, 1: 0}  # Initialize counts for Buy (1) and Sell (0)

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Convert action to an integer
            action_counts[action] += 1
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    return {
        'average_reward': avg_reward,
        'action_distribution': action_counts
    }

    return metrics

def train_rl(ticker, lstm_model_path, xgboost_model_path, start_date, end_date, features, total_timesteps, save_path):
    """
    Train the RL model using PPO in the StockPredictionEnv.
    """
    # Load models
    lstm_model = load_model(lstm_model_path, custom_objects={'Attention': Attention})
    xgboost_model = load(xgboost_model_path)

    # Fetch historical data and calculate technical indicators
    historical_data = fetch_stock_data(ticker, start_date, end_date)
    historical_data = calculate_technical_indicators(historical_data)

    # Initialize environment
    env = StockPredictionEnv(
        historical_data=historical_data,
        lstm_model=lstm_model,
        xgboost_model=xgboost_model,
        features=features  # Pass the stock-specific features here
    )

    # Train PPO model
    ppo_model = PPO("MlpPolicy", env, verbose=1)
    ppo_model.learn(total_timesteps=total_timesteps)

    # Save the trained PPO model
    ppo_model.save(save_path)

    print("\nEvaluating the RL model...")
    metrics = evaluate_rl_model(env, ppo_model, n_episodes=10)

    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics