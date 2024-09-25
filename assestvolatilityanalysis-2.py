# -*- coding: utf-8 -*-
"""AssestVolatilityAnalysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14DjVjw6nfTjOW4drLd8YbR27bJLqdpS2
"""

import yfinance as yf
import pandas as pd

start_date = "2020-01-01"
end_date = "2024-06-01"

tickers = ['META', 'AAPL', 'TSLA', 'MSFT']

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
data.index = data.index.tz_localize(None)

# Check first few rows of the data
print(data.head())

import numpy as np

# Calculate daily log returns
returns = np.log(data / data.shift(1))

# Calculate daily and annualized volatility (standard deviation of returns)
volatility_daily = returns.std()
volatility_annual = volatility_daily * np.sqrt(252)  # Annualized volatility

print("Daily Volatility:\n", volatility_daily)
print("Annualized Volatility:\n", volatility_annual)

def calculate_atr(df, period=14):
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = np.abs(df['High'] - df['Adj Close'].shift(1))
    df['Low-PrevClose'] = np.abs(df['Low'] - df['Adj Close'].shift(1))

    df['True Range'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['True Range'].rolling(window=period).mean()

    return df['ATR']

data_meta = yf.download('META', start=start_date, end=end_date)
atr_meta = calculate_atr(data_meta)

data_tsla = yf.download('TSLA', start=start_date, end=end_date)
atr_tsla = calculate_atr(data_tsla)

data_msft = yf.download('MSFT', start=start_date, end=end_date)
atr_msft = calculate_atr(data_msft)

data_aapl = yf.download('AAPL', start=start_date, end=end_date)
atr_aapl = calculate_atr(data_aapl)

print("\nATR for META:\n", atr_meta.tail())
print("ATR for TSLA:\n", atr_tsla.tail())
print("ATR for MSFT:\n", atr_msft.tail())
print("ATR for AAPL:\n", atr_aapl.tail())

plt.figure(figsize=(10, 6))
plt.plot(data_meta.index, atr_meta, label='ATR (META)')
plt.title('Average True Range (ATR) for META')
plt.xlabel('Date')
plt.ylabel('ATR')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data_meta.index, atr_tsla, label='ATR (META)')
plt.title('Average True Range (ATR) for TSLA')
plt.xlabel('Date')
plt.ylabel('ATR')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data_meta.index, atr_msft, label='ATR (MSFT)')
plt.title('Average True Range (ATR) for MSFT')
plt.xlabel('Date')
plt.ylabel('ATR')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data_meta.index, atr_aapl, label='ATR (AAPL)')
plt.title('Average True Range (ATR) for AAPL')
plt.xlabel('Date')
plt.ylabel('ATR')
plt.grid(True)
plt.legend()
plt.show()

risk_free_rate = 0.01  # Assuming a risk-free rate of 1%

def sharpe_ratio(returns, risk_free_rate):
    excess_return = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    sharpe = excess_return / volatility
    return sharpe

sharpe_meta = sharpe_ratio(returns['META'], risk_free_rate)
sharpe_aapl = sharpe_ratio(returns['AAPL'], risk_free_rate)
sharpe_tsla = sharpe_ratio(returns['TSLA'], risk_free_rate)
sharpe_msft = sharpe_ratio(returns['MSFT'], risk_free_rate)

print(f"Sharpe Ratio\nMETA: {sharpe_meta}\nAAPL: {sharpe_aapl}\nTSLA: {sharpe_tsla}\nMSFT: {sharpe_msft}")

sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
sp500_data.index = sp500_data.index.tz_localize(None)

returns = np.log(data / data.shift(1))
sp500_returns = np.log(sp500_data / sp500_data.shift(1))

combined_returns = pd.concat([returns[['META', 'AAPL', 'TSLA', 'MSFT']], sp500_returns.rename('S&P500')], axis=1)

combined_returns = combined_returns.dropna()

correlation_meta = combined_returns['META'].corr(combined_returns['S&P500'])
correlation_aapl = combined_returns['AAPL'].corr(combined_returns['S&P500'])
correlation_tsla = combined_returns['TSLA'].corr(combined_returns['S&P500'])
correlation_msft = combined_returns['MSFT'].corr(combined_returns['S&P500'])

print(f"\nCorrelation with S&P 500\nMETA: {correlation_meta}\nAAPL: {correlation_aapl}\nTSLA: {correlation_tsla}\nMSFT: {correlation_msft}")

data['SMA_50_META'] = data['META'].rolling(window=50).mean()
data['SMA_200_META'] = data['META'].rolling(window=200).mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(data['META'], label='META Price')
plt.plot(data['SMA_50_META'], label='50-day SMA (META)')
plt.plot(data['SMA_200_META'], label='200-day SMA (META)')
plt.title('META Price and Moving Averages')
plt.legend()
plt.show()

data['SMA_50_TSLA'] = data['TSLA'].rolling(window=50).mean()
data['SMA_200_TSLA'] = data['TSLA'].rolling(window=200).mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(data['TSLA'], label='TSLA Price')
plt.plot(data['SMA_50_TSLA'], label='50-day SMA (TSLA)')
plt.plot(data['SMA_200_TSLA'], label='200-day SMA (TSLA)')
plt.title('TSLA Price and Moving Averages')
plt.legend()
plt.show()

data['SMA_50_MSFT'] = data['MSFT'].rolling(window=50).mean()
data['SMA_200_MSFT'] = data['MSFT'].rolling(window=200).mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(data['MSFT'], label='MSFT Price')
plt.plot(data['SMA_50_MSFT'], label='50-day SMA (MSFT)')
plt.plot(data['SMA_200_MSFT'], label='200-day SMA (MSFT)')
plt.title('MSFT Price and Moving Averages')
plt.legend()
plt.show()

data['SMA_50_AAPL'] = data['AAPL'].rolling(window=50).mean()
data['SMA_200_AAPL'] = data['AAPL'].rolling(window=200).mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(data['AAPL'], label='AAPL Price')
plt.plot(data['SMA_50_AAPL'], label='50-day SMA (AAPL)')
plt.plot(data['SMA_200_AAPL'], label='200-day SMA (AAPL)')
plt.title('AAPL Price and Moving Averages')
plt.legend()
plt.show()

volume_data = yf.download(tickers, start=start_date, end=end_date)['Volume']

print(volume_data.head())

average_volume = volume_data.mean()

print("Average Daily Trading Volume:")
print(average_volume)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for ticker in tickers:
    plt.plot(volume_data.index, volume_data[ticker], label=ticker)

plt.title('Daily Trading Volume (2020-2024)')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()

price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

dollar_volume = volume_data * price_data

average_dollar_volume = dollar_volume.mean()

print("\nAverage Dollar Volume (Liquidity in $):")
print(average_dollar_volume)