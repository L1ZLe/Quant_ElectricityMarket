import matplotlib.pyplot as plt
import streamlit as st

def plot_predictions(model_name, data):
    st.write(f"Plotting predictions for {model_name}")
    # Placeholder code to plot predictions
    plt.plot(data)
    plt.title(f"Predictions vs Actuals for {model_name}")
    st.pyplot()

def plot_equity_curve(strategy_name, data):
    st.write(f"Plotting equity curve for {strategy_name}")
    # Placeholder code to plot equity curve
    plt.plot(data)
    plt.title(f"Equity Curve for {strategy_name}")
    st.pyplot()

def plot_trades(trades):
    st.write("Plotting individual trades")
    # Placeholder code to plot trades
    plt.plot(trades['Date'], trades['Profit/Loss'])
    plt.title("Individual Trades")
    st.pyplot()

def waiting_statement():
    st.write("Running backtest, please wait...")
    
def final_balance_plotting(starting_amount, total_roi, data, start_idx, end_idx):
    final_balance = starting_amount * (1 + total_roi)
    
    st.write(f"Final balance starting with ${starting_amount} and buying/selling 1 MWh of electricity: ${final_balance:.2f}")
    st.write(f"Total ROI: {total_roi*100:.2f}%")

    st.write("Price Plot with Position Indicator (Zoomed In):")
    dates = data.loc[start_idx:end_idx, 'Trade Date'].values
    prices = data.loc[start_idx:end_idx, 'Electricity: Wtd Avg Price $/MWh'].values
    positions = data.loc[start_idx:end_idx, 'Position'].values
    plt.figure(figsize=(15, 7))
    for i in range(1, len(dates)):
        if positions[i] == 1:
            plt.plot([dates[i-1], dates[i]], [prices[i-1], prices[i]], color='green')
        elif positions[i] == -1:
            plt.plot([dates[i-1], dates[i]], [prices[i-1], prices[i]], color='red')
    plt.title('Price Plot with Position Indicator (Zoomed In)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(plt)