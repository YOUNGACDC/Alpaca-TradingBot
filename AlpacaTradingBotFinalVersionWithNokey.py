# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 19:17:52 2024

@author: Armanis
"""

#%% Step 1: Import Required Libraries
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import pytz
from alpaca_trade_api.rest import TimeFrame

#%% Step 2: Alpaca API Authentication
API_KEY = 'xxx'
SECRET_KEY = 'xxx'
BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading URL

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
account = api.get_account()
print(f"Account Status: {account.status}")

'''#%%  Alpaca API Authentication (Check if it works pt 2)

orders = api.list_orders()
print(orders)

# Submit a test order
api.submit_order(
    symbol='AAPL',
    qty=1,
    side='buy',
    type='market',
    time_in_force='gtc'
)


#%% Alpaca API Authentication (Check if it works part 3)
order = api.get_order('b11e3e00-f85d-4bf2-b942-92b60285f19e')  # Replace with your order ID
print(order.status)'''
#%% Step 3: Define Historical Data Fetching Function
def get_historical_data(symbol, timeframe='minute', limit=1000):
    """
    Fetch historical data for a given symbol using Alpaca API.
    """
    try:
        # Get today's date and time
        now = datetime.now(pytz.timezone('America/New_York'))
        today_date = now.strftime('%Y-%m-%d')

        # Define start and end datetime for today's trading session
        start_datetime = f"{today_date}T09:30:00-05:00"  # Market opens at 9:30 AM EST
        end_datetime = f"{today_date}T16:00:00-05:00"    # Market closes at 4:00 PM EST

        timeframes = {
            'minute': TimeFrame.Minute,
            'hour': TimeFrame.Hour,
            'day': TimeFrame.Day,
        }
        alpaca_timeframe = timeframes.get(timeframe.lower(), TimeFrame.Minute)

        bars = api.get_bars(
            symbol,
            timeframe=alpaca_timeframe,
            start=start_datetime,
            end=end_datetime,
            feed='iex'  # Use IEX feed for free tier
        ).df

        # Convert time zone to Eastern Time
        bars = bars.tz_convert('America/New_York')
        bars = bars[['close']].copy()
        bars.index.name = 'time'

        # Filter for regular trading hours (redundant safety measure)
        bars = bars.between_time('09:30', '16:00')

        return bars
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()



#%% Step 4: Define Moving Average Crossover Strategy and Define trading hours


# Define trading hours (Eastern Time)
TRADING_START_TIME = datetime.strptime("09:30:00", "%H:%M:%S").time()
TRADING_END_TIME = datetime.strptime("16:00:00", "%H:%M:%S").time()
EASTERN_TIMEZONE = pytz.timezone("US/Eastern")


def moving_average_crossover_strategy(symbol):
    """
    Implement a moving average crossover strategy with trading hours validation.
    """
    df = get_historical_data(symbol, timeframe='minute', limit=1000)
    if df.empty:
        return "hold"
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Get the current time in Eastern Time
    current_time = datetime.now(EASTERN_TIMEZONE).time()

    # Check if the current time is within trading hours
    if current_time < TRADING_START_TIME or current_time > TRADING_END_TIME:
        print("Outside of trading hours. No trades executed.")
        return "hold"

    # Determine the trading signal
    if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
        return "buy"
    elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1]:
        return "sell"
    return "hold"
#%% Step 5: Summarize Strategy Performance
def summarize_strategy(df):
    """
    Summarize the performance of the moving average crossover strategy.
    """
    buy_signals = df[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))]
    sell_signals = df[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))]

    buy_prices = buy_signals['close'].tolist()
    sell_prices = sell_signals['close'].tolist()

    trades = []
    while buy_prices or sell_prices:
        if buy_prices and sell_prices:
            buy_price = buy_prices.pop(0)
            sell_price = sell_prices.pop(0)
            trades.append({'Buy': buy_price, 'Sell': sell_price, 'Profit/Loss': sell_price - buy_price})
        elif buy_prices:
            trades.append({'Buy': buy_prices.pop(0), 'Sell': None, 'Profit/Loss': None})
        elif sell_prices:
            sell_price = sell_prices.pop(0)
            trades.append({'Buy': None, 'Sell': sell_price, 'Profit/Loss': None})

    trades_df = pd.DataFrame(trades)
    completed_trades = trades_df.dropna(subset=['Buy', 'Sell'])
    total_profit_loss = completed_trades['Profit/Loss'].sum()

    print(f"Trades Summary for Today:\n{trades_df}")
    print(f"Total Profit/Loss from Completed Trades: {total_profit_loss}")
    print(f"Unpaired Trades: {len(trades_df) - len(completed_trades)}")

    return {
        'Total Profit/Loss for today': total_profit_loss,
        'Number of Buy Signals today': len(buy_signals),
        'Number of Sell Signals today': len(sell_signals),
        'Unpaired Trades': len(trades_df) - len(completed_trades),
    }

#%% Step 6: Visualization of Strategy
def visualize_strategy(symbol, df):
    """
    Visualize the moving average crossover strategy.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['close'], label=f'{symbol} Price', color='black', alpha=0.8, linewidth=1.5)
    plt.plot(df.index, df['SMA_20'], label='20-Minute SMA', color='orange', linestyle='--', linewidth=2)
    plt.plot(df.index, df['SMA_50'], label='50-Minute SMA', color='green', linestyle='--', linewidth=2)

    buy_signals = df[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))]
    sell_signals = df[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))]

    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)

    plt.title(f'Moving Average Crossover Strategy for {symbol}', fontsize=16)
    plt.xlabel(' Military Time', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.show()



# def visualize_strategy2(symbol, df):
#     """
#     Visualize the moving average crossover strategy.

#     Args:
#         symbol: Stock symbol.
#         df: DataFrame containing 'close', 'SMA_20', and 'SMA_50'.
#     """
#     plt.clf()  # Clear the previous figure
    
#     plt.figure(figsize=(14, 8))
#     plt.plot(df.index, df['close'], label=f'{symbol} Price', color='black', alpha=0.8, linewidth=1.5)
#     plt.plot(df.index, df['SMA_20'], label='20-Minute SMA', color='orange', linestyle='--', linewidth=2)
#     plt.plot(df.index, df['SMA_50'], label='50-Minute SMA', color='green', linestyle='--', linewidth=2)

#     # Highlight buy/sell signals
#     buy_signals = df[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))]
#     sell_signals = df[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))]

#     plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
#     plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)

#     # Grid and title
#     plt.title(f' Daily Moving Average Crossover Strategy for {symbol}', fontsize=16)
#     plt.xlabel('EST Time', fontsize=12)
#     plt.ylabel('Price (USD)', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(fontsize=12, loc='upper left')
#     plt.show()
    
    
#%% Step 6: Visualization of Strategy (Zoomed View)

def visualize_strategy_zoomed2(symbol, df, start_date=None, end_date=None):
    """
    Visualize the moving average crossover strategy with a zoomed view.

    Args:
        symbol: Stock symbol.
        df: DataFrame containing 'close', 'SMA_20', and 'SMA_50'.
        start_date: Start of the zoomed range (datetime object or None).
        end_date: End of the zoomed range (datetime object or None).
    """
    try:
        # Ensure the data index is in datetime format
        df.index = pd.to_datetime(df.index)

        # Set timezone for start and end dates if provided
        tz = pytz.timezone('America/New_York')  # Eastern Time
        if start_date and end_date:
            start_date = tz.localize(start_date)
            end_date = tz.localize(end_date)
            # Filter the DataFrame for the zoomed range
            df_filtered = df.loc[start_date:end_date]
        else:
            # Default to the last 100 rows if no range is specified
            df_filtered = df.iloc[-100:]

        # Handle empty DataFrame after filtering
        if df_filtered.empty:
            print("No data available in the specified zoom range. Adjusting to available data.")
            df_filtered = df.iloc[-100:]

        # Clear the previous figure
        plt.clf()

        # Plot the zoomed data
        plt.figure(figsize=(14, 8))
        plt.plot(df_filtered.index, df_filtered['close'], label=f'{symbol} Price', color='black', alpha=0.8, linewidth=1.5)
        plt.plot(df_filtered.index, df_filtered['SMA_20'], label='20-Minute SMA', color='orange', linestyle='--', linewidth=2)
        plt.plot(df_filtered.index, df_filtered['SMA_50'], label='50-Minute SMA', color='green', linestyle='--', linewidth=2)

        # Highlight buy/sell signals
        buy_signals = df_filtered[(df_filtered['SMA_20'] > df_filtered['SMA_50']) & (df_filtered['SMA_20'].shift(1) <= df_filtered['SMA_50'].shift(1))]
        sell_signals = df_filtered[(df_filtered['SMA_20'] < df_filtered['SMA_50']) & (df_filtered['SMA_20'].shift(1) >= df_filtered['SMA_50'].shift(1))]

        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)

        # Grid and title
        plt.title(f'Moving Average Crossover Strategy for {symbol} (Zoomed View)', fontsize=16)
        plt.xlabel('Time (EST)', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12, loc='upper left')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in visualize_strategy_zoomed2: {e}")


#%% Step 7: Test the Strategy and Print Results
if __name__ == "__main__":
    symbol = 'goog'
    
    # Assuming get_historical_data is defined elsewhere
    data = get_historical_data(symbol, timeframe='minute', limit=1000)
    
    if not data.empty:
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()

        print("Displaying full strategy view...")
        visualize_strategy(symbol, data)  # Ensure visualize_strategy is defined

        print("Summarizing strategy performance...")
        summary = summarize_strategy(data)  # Ensure summarize_strategy is defined
        print(f"Summary: {summary}")

        # Specify the zoom range for today (adjust time ranges as necessary)
        zoom_start = datetime(2024,12,9,9,30)
        zoom_end = datetime(2024,12,9,12,30)


        # Display zoomed strategy view for today
        print("Displaying zoomed strategy view...")
        visualize_strategy_zoomed2(symbol, data, start_date=zoom_start, end_date=zoom_end)
    else:
        print("No data fetched. Exiting.")
