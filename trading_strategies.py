import numpy as np
import pandas as pd
import streamlit as st
from visualizations import final_balance_plotting, waiting_statement

def strategy_logic(strategy_name):
    # Placeholder function to explain strategy logic
    if strategy_name == "BOS Trading":
        return "BOS Trading logic explanation"
    elif strategy_name == "Quantile-based Strategy":
        return "Quantile-based Strategy logic explanation"

def backtest(strategy_name, data, window_size=14, thresholds=0.05, return_trades=False):
    # Placeholder function for backtesting
    results = pd.DataFrame({
        "Date": pd.date_range(start='1/1/2020', periods=len(data)),
        "Equity": np.random.randn(len(data)).cumsum()
    })
    if return_trades:
        trades = pd.DataFrame({
            "Date": pd.date_range(start='1/1/2020', periods=10),
            "Entry": np.random.randn(10),
            "Exit": np.random.randn(10),
            "Profit/Loss": np.random.randn(10)
        })
        return trades
    return results


def calculate_percentiles(data, window_size, percentile_20, percentile_80):
    data['Percentile_20'] = data['Electricity: Wtd Avg Price $/MWh'].rolling(window=window_size).apply(lambda x: np.percentile(x, percentile_20), raw=True)
    data['Percentile_80'] = data['Electricity: Wtd Avg Price $/MWh'].rolling(window=window_size).apply(lambda x: np.percentile(x, percentile_80), raw=True)
    return data

def run_percentile_strategy(starting_amount, data):
    st.write("Adjust Parameters:")
    window_size = st.slider("Window Size for Percentile-based Strategy", 1, 30, 14)
    percentile_20 = st.slider("Lower Percentile (Buy Signal)", 0, 50, 20)
    percentile_80 = st.slider("Upper Percentile (Sell Signal)", 50, 100, 80)

    start_date = st.date_input("Start Date for Plot", data['Trade Date'].min())
    end_date = st.date_input("End Date for Plot", data['Trade Date'].max())

    start_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(start_date)].index[0])
    end_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(end_date)].index[0])
    
    if st.button("Run Backtest"):
        waiting_statement()
        data = calculate_percentiles(data.iloc[start_idx:end_idx], window_size, percentile_20, percentile_80)

        data['Signal'] = 0
        data['Position'] = 0

        for i in range(window_size, len(data)):
            if data['Electricity: Wtd Avg Price $/MWh'].iloc[i] <= data['Percentile_20'].iloc[i]:
                data['Signal'].iloc[i] = 1  # Buy signal
            elif data['Electricity: Wtd Avg Price $/MWh'].iloc[i] >= data['Percentile_80'].iloc[i]:
                data['Signal'].iloc[i] = -1  # Sell signal
        
        data['Position'] = data['Signal'].replace(to_replace=0, method='ffill')
        data.fillna(method='ffill', inplace=True)

        total_roi = calculate_ROI(data)
        final_balance_plotting(starting_amount, total_roi, data, start_idx, end_idx)
    return data

def run_BOS_strategy(starting_amount, data):
    start_date = st.date_input("Start Date for Plot", data['Trade Date'].min())
    end_date = st.date_input("End Date for Plot", data['Trade Date'].max())

    start_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(start_date)].index[0])
    end_idx = data.index.get_loc(data[data['Trade Date'] == pd.to_datetime(end_date)].index[0])
    
    if st.button("Run Backtest"):
        waiting_statement()
        data = BOS_logic(data.iloc[start_idx:end_idx], starting_amount)
        total_roi = calculate_ROI(data)
        final_balance_plotting(starting_amount, total_roi, data, start_idx, end_idx)
    return data

def BOS_logic(data, initial_capital):
    # Store the Trade Date column
    trade_dates = data['Trade Date'].values
    
    # Initialize parameters
    trend = True  # Assume an initial trend
    high = -np.inf
    low = np.inf
    close = data['Electricity: Wtd Avg Price $/MWh'].iloc[0]
    extrems_date = data.index[0]
    start_date = data.index[0]
    capital = initial_capital
    position = None

    results = []

    # Iterate through the data
    for current_date, current_row in data.iterrows():
        current_price = current_row['Electricity: Wtd Avg Price $/MWh']
        data_up_to_current_date = data.loc[:current_date, 'Electricity: Wtd Avg Price $/MWh']

        # Detect trend
        new_trend, relevant_data = detect_trend(data_up_to_current_date, extrems_date, trend, close)

        # Get latest high, low, and close
        high, low, close, start_date, extrems_date = get_latest_high_and_low(relevant_data, start_date, extrems_date, trend, new_trend, high, low, close)

        # Strategy: Buy or sell based on trend change (simulated)
        if position is None:
            if new_trend:
                position = 1
            else:
                position = -1
        elif new_trend and position == -1:
            capital += current_price
            position = 1
        elif not new_trend and position == 1:
            capital -= current_price
            position = -1

        # Store results for analysis
        results.append((current_date, current_price, trend, new_trend, high, low, close, capital, position))

        # Update trend
        trend = new_trend
    
    # Create a DataFrame to analyze results
    results = pd.DataFrame(results, columns=['Trade Date', 'Electricity: Wtd Avg Price $/MWh', 'Initial Trend', 'New Trend', 'High', 'Low', 'Close', 'Capital', 'Position']).set_index('Trade Date')
    
    # Attach the stored trade dates back to the DataFrame
    results['Trade Date'] = trade_dates
    return results

def detect_trend(data, extrems_date, trend, close_readfiles):
        latest_close = data.iloc[-1]
        if trend and latest_close < close_readfiles:
            trend = False
            data = data[data.index >= extrems_date]
        elif not trend and latest_close > close_readfiles:
            trend = True
            data = data[data.index >= extrems_date]
        return trend, data
def calculate_ROI(data):
    buy_price = None
    total_return = 0.0

    for i in range(1, len(data)):
        if data['Position'].iloc[i] == 1 and data['Position'].iloc[i - 1] == -1:
            buy_price = data['Electricity: Wtd Avg Price $/MWh'].iloc[i]
        elif data['Position'].iloc[i] == -1 and data['Position'].iloc[i - 1] == 1:
            if buy_price is not None:
                sell_price = data['Electricity: Wtd Avg Price $/MWh'].iloc[i]
                total_return += (sell_price - buy_price) / buy_price
                buy_price = None

    return total_return

def get_latest_high_and_low(data, start_date, extrems_date, initial_trend, new_trend, high, low, close):
        if initial_trend and not new_trend:
            low = np.inf
            close = np.inf
            high = None
            start_date = extrems_date
        elif not initial_trend and new_trend:
            high = -np.inf
            close = -np.inf
            low = None
            start_date = extrems_date

        if new_trend:
            for i in range(len(data)):
                if high <= data.iloc[i]:
                    high = data.iloc[i]
                    extrems_date = data.index[i]
                    temp = i
            for temp in range(temp, -1, -1):
                if close == data.iloc[temp]:
                    break
                if data.iloc[temp - 1] > data.iloc[temp] and data.iloc[temp + 1] > data.iloc[temp]:
                    close = data.iloc[temp]
                    break
        else:
            for i in range(len(data)):
                if low >= data.iloc[i]:
                    low = data.iloc[i]
                    extrems_date = data.index[i]
                    temp = i
            for temp in range(temp, -1, -1):
                if close == data.iloc[temp]:
                    break
                if data.iloc[temp - 1] < data.iloc[temp] and data.iloc[temp + 1] < data.iloc[temp]:
                    close = data.iloc[temp]
                    break

        if close in [None, np.inf, -np.inf]:
            close = data.iloc[0]

        return high, low, close, start_date, extrems_date

