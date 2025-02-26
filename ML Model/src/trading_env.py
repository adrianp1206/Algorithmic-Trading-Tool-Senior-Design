import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class StockTradingEnv:
    """
    A simplified stock trading environment for RL.
    Actions:
        0 = Hold
        1 = Buy (if not already holding)
        2 = Sell (if holding)
    State: 
        [Open, High, Low, Close, Volume, XGB_Pred, XGB_Prob_Up, Position]
    Reward:
        - Daily unrealized gains/losses while holding
        - Realized gains/losses when we sell
    """
    def __init__(self, df, initial_balance=10000):
        """
        df: Pandas DataFrame with columns:
            ['Open', 'High', 'Low', 'Close', 'Volume', 'XGB_Pred', 'XGB_Prob_Up', ...]
        initial_balance: starting cash
        """
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        self.action_space = [0, 1, 2]  # hold, buy, sell
        
        # We define the observation space (state) shape:
        #  - 7 numeric features from the DataFrame (Open..XGB_Prob_Up)
        #  - 1 dimension for 'Position' (0 or 1)
        self.state_size = 7 + 1  # e.g., 7 columns from your DF + 1 for position

        self.reset()

    def reset(self):
        """
        Reset the environment state to the beginning.
        Returns the initial state.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0   # 0 = not holding, 1 = holding one share
        self.last_price = 0
        self.total_shares = 0  # how many shares are we holding? (0 or 1 here)
        
        # For reward calculation, track the initial portfolio value
        self.portfolio_value = self.initial_balance
        
        return self._get_state()

    def step(self, action):
        """
        Take an action:
        0 = hold,
        1 = buy (if not holding),
        2 = sell (if holding).
        
        Returns: next_state, reward, done, info
        """
        done = False
        reward = 0
        
        # Current day data
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['Close']

        # Execute the action
        if action == 1:  # buy
            if self.position == 0:  # only buy if we are flat
                self.position = 1
                self.total_shares = 1
                self.last_price = current_price
            # else do nothing if we're already holding

        elif action == 2:  # sell
            if self.position == 1:  # only sell if we are holding
                # Realize the P&L
                pnl = (current_price - self.last_price) * self.total_shares
                self.balance += pnl
                self.position = 0
                self.total_shares = 0
                reward = pnl  # realized profit/loss as reward
            # else do nothing if we're already flat

        else:  # hold
            pass

        # Update portfolio value if we are holding
        # (unrealized gains/losses can be included in reward or not, depending on design)
        if self.position == 1:
            # Unrealized P&L
            unrealized_pnl = (current_price - self.last_price) * self.total_shares
            # Optionally, you could incorporate partial daily reward:
            reward += unrealized_pnl * 0.01  # e.g., partial credit for going in the right direction

        self.current_step += 1
        
        # Check if we're at the end of the data
        if self.current_step >= self.n_steps - 1:
            done = True
        
        # Next state
        next_state = self._get_state()

        return next_state, reward, done, {}

    def _get_state(self):
        """
        Construct the state from the current step's data + position info.
        The order of state features should match your agent's expectations.
        """
        data_row = self.df.iloc[self.current_step]
        # Example: [Open, High, Low, Close, Volume, XGB_Pred, XGB_Prob_Up, Position]
        state = [
            data_row['Open'],
            data_row['High'],
            data_row['Low'],
            data_row['Close'],
            data_row['Volume'],
            data_row['XGB_Pred'],
            data_row['XGB_Prob_Up'],
            self.position
        ]
        return np.array(state, dtype=np.float32)
