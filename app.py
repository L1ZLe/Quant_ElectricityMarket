import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from model_module import predict, calculate_metrics
from trading_strategies import strategy_logic, backtest, run_percentile_strategy, run_BOS_strategy,descriptions
from visualizations import plot_predictions, plot_equity_curve,waiting_statement
import numpy as np




# Load data
def load_dataset():
    AllInOne_Data = pd.read_csv(r"C:\Users\Sami El yaagoubi\Desktop\capstone\datasets\Data_cleaned_Dataset.csv", parse_dates=['Trade Date', 'Electricity: Delivery Start Date', 'Electricity: Delivery End Date'])

# Interpolate missing data
    AllInOne_Data = AllInOne_Data.interpolate()

    # Replace zero values with the mean of non-zero values
    mean_non_zero = AllInOne_Data[AllInOne_Data['Electricity: Wtd Avg Price $/MWh'] != 0]['Electricity: Wtd Avg Price $/MWh'].mean()
    AllInOne_Data.loc[AllInOne_Data['Electricity: Wtd Avg Price $/MWh'] == 0, 'Electricity: Wtd Avg Price $/MWh'] = mean_non_zero

    return AllInOne_Data

def home():
    st.title("Electricity Trading Strategy Project")
    
    st.write("""
    Welcome to the Electricity Trading Strategy Project website! This platform showcases the work done in analyzing and developing trading strategies for the electricity market. Here's what you can explore:
    """)

    # Project Overview
    st.header("Project Overview")
    st.write("""
    This project aims to forecast electricity prices in the USA, specifically the PJM Interconnection, and develop trading strategies based on these forecasts and other known strategies. We employ advanced machine learning models such as SARIMA and GRU to predict price movements and create a robust trading strategy. The models will be evaluated based on their accuracy in predicting the direction of the next day's price and the accuracy of the predicted prices.
    """)

    # Key Features
    st.header("Key Features")
    st.write("""
    - **Model Overview**: Detailed explanations of the models used for price predictions.
    - **Data Exploration**: Interactive visualizations of historical electricity prices and other relevant data, such as natural gas prices, and how these variables impact electricity prices.
    - **Predictions**: Live predictions based on the latest data, allowing users to see next-day price forecasts based on the chosen model.
    - **Trading Strategy**: Comprehensive description of the trading logic and strategy implementation.
    - **Performance Metrics**: Evaluation of the trading strategy’s performance through various metrics such as Sharpe ratio, win rate, and ROI.
    - **Backtesting**: Tools to backtest the trading strategy on historical data to assess its viability.
    - **Risk Management**: Discussion on risk management techniques and tools to adjust strategy parameters.
    """)

    # Model Evaluation
    st.header("Model Evaluation")
    st.write("""
    We evaluate our models using two main criteria:
    - **Direction Accuracy**: The accuracy of predicting the direction of the next day's price movement.
    - **Price Accuracy**: The accuracy of the actual predicted prices compared to the real prices.
    """)

def data_exploration():
    # First dataset: Net electricity generation by source
    data_1 = pd.read_csv('datasets/Net_generation_United_States_all_sectors_monthly.csv')
    st.title("Data Exploration")
    
    # Display the raw data
    st.write("### Net electricity generation by source")
    st.write(data_1)

    # Convert 'Month' column to datetime format
    try:
        data_1['Month'] = pd.to_datetime(data_1['Month'], format='%b-%y', errors='coerce')
    except ValueError:
        st.error("Error parsing dates in the first dataset. Please check the date format in the CSV file.")
        st.stop()

    # Drop rows with invalid dates
    data_1 = data_1.dropna(subset=['Month'])

    # Select multiple sources to plot
    sources_1 = list(data_1.columns[1:])  # Assuming the first column is the date or time
    selected_sources_1 = st.multiselect("Select sources to plot from the first dataset", options=sources_1, default=sources_1)

    # Filter data for the selected sources
    plot_data_1 = data_1[['Month'] + selected_sources_1].dropna()

    # Plot the data
    fig_1, ax_1 = plt.subplots()
    for source in selected_sources_1:
        ax_1.plot(plot_data_1['Month'], plot_data_1[source], label=source)
    
    ax_1.set_title("Net Electricity Generation by Source")
    ax_1.set_xlabel("Month")
    ax_1.set_ylabel("Net Generation (thousand megawatthours)")
    ax_1.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig_1)
    
    # Second dataset: Net generation by places
    data_2 = pd.read_csv(r"C:\Users\Sami El yaagoubi\Desktop\capstone\datasets\Net_generation_by places.csv")
    
    # Display the raw data
    st.write("### Net electricity generation by places")
    st.write(data_2)

    # Convert 'Month' column to datetime format
    try:
        data_2['Month'] = pd.to_datetime(data_2['Month'], format='%y-%b', errors='coerce')
    except ValueError:
        st.error("Error parsing dates in the second dataset. Please check the date format in the CSV file.")
        st.stop()

    # Drop rows with invalid dates
    data_2 = data_2.dropna(subset=['Month'])

    # Select multiple regions to plot
    regions = list(data_2.columns[1:])  # Assuming the first column is the date or time
    selected_regions = st.multiselect("Select regions to plot from the second dataset", options=regions, default=regions)

    # Filter data for the selected regions
    plot_data_2 = data_2[['Month'] + selected_regions].dropna()

    # Plot the data
    fig_2, ax_2 = plt.subplots()
    for region in selected_regions:
        ax_2.plot(plot_data_2['Month'], plot_data_2[region], label=region)
    
    ax_2.set_title("Net Electricity Generation by Places")
    ax_2.set_xlabel("Month")
    ax_2.set_ylabel("Net Generation (thousand megawatthours)")
    ax_2.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig_2)

    file_path_3 = "datasets\Retail_sales_of_electricity_United_States_monthly.csv"
    data_3 = pd.read_csv(file_path_3)
    
    # Display the raw data
    st.write("### Retail sales of electricity")
    st.write(data_3)

    # Convert 'Month' column to datetime format
    try:
        data_3['Month'] = pd.to_datetime(data_3['Month'], format='%b-%y', errors='coerce')
    except ValueError:
        st.error("Error parsing dates in the third dataset. Please check the date format in the CSV file.")
        st.stop()

    # Drop rows with invalid dates
    data_3 = data_3.dropna(subset=['Month'])

    # Select multiple regions to plot
    types = list(data_3.columns[1:])  # Assuming the first column is the date or time
    selected_types = st.multiselect("Select types to plot from the third dataset", options=types, default=types)

    # Filter data for the selected regions
    plot_data_3 = data_3[['Month'] + selected_types].dropna()

    # Plot the data
    fig_3, ax_3 = plt.subplots()
    for type in selected_types:
        ax_3.plot(plot_data_3['Month'], plot_data_3[type], label=type)
    
    ax_3.set_title("Retail Sales of Electricity")
    ax_3.set_xlabel("Month")
    ax_3.set_ylabel("Sales (thousand megawatthours)")
    ax_3.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig_3)

    st.subheader('Correlations between variables:')
    st.image("assets\Correlations between variables1.png")
    st.image("assets\Correlations between variables2.png")

    st.subheader("Relation between Electricity price and Temperature")
    st.image("assets\Relation between Electricity price and Temperature.png")

    st.subheader("Net_generated electricity in United States")
    st.image(r"assets\Net_generated electricity and Temperature.png")

    st.subheader("Average Electricity price by Month")
    st.image("assets\Average Electricity price by Month.png")

    st.subheader("Electricity seasonal decomposition")
    st.image("assets\Electricity seasonal decomposition.png")

    st.subheader("Natural Gas seasonal decomposition")
    st.image(r"assets\Natural Gas seasonal decomposition.png")

    st.title("Electricity and Natural Gas Data")
    st.write("Here is a preview of the dataset:")
    AllInOne_Data = load_dataset()
    st.dataframe(AllInOne_Data.head())
    # Date range slider
    st.write("Select the date range to display:")
    min_date = AllInOne_Data['Trade Date'].min().to_pydatetime()
    max_date = AllInOne_Data['Trade Date'].max().to_pydatetime()
    date_range = st.slider("Date", min_date, max_date, (min_date, max_date))
    # Filter data based on selected date range
    filtered_data = AllInOne_Data[(AllInOne_Data['Trade Date'] >= date_range[0]) & (AllInOne_Data['Trade Date'] <= date_range[1])]
    # Interactive graph for Electricity
    st.write("Electricity Prices Over Time:")
    fig_electricity = px.line(filtered_data, x='Trade Date', y='Electricity: Wtd Avg Price $/MWh', title='Electricity Prices Over Time')
    st.plotly_chart(fig_electricity)
    # Interactive graph for Natural Gas
    st.write("Natural Gas Prices Over Time:")
    fig_natural_gas = px.line(filtered_data, x='Trade Date', y='Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)', title='Natural Gas Prices Over Time')
    st.plotly_chart(fig_natural_gas)

    from sklearn.preprocessing import MinMaxScaler
    st.write("### All in One Graph")
    # Drop rows with NaN values in the specified columns
    AllInOne_Data = AllInOne_Data.dropna(subset=['Trade Date', 'Electricity: Delivery Start Date', 'Electricity: Delivery End Date'])
    # Get the list of types (assuming the first three columns are dates or time related)
    types = list(AllInOne_Data.columns[3:])
    # Multiselect widget for selecting types to plot
    selected_types = st.multiselect("Select variables to plot", options=types, default=[])
    # Prepare the data for plotting
    if selected_types:
        plot_AllInOne_Data = AllInOne_Data[['Trade Date'] + selected_types].dropna()
        # Normalize the selected columns
        scaler = MinMaxScaler()
        plot_AllInOne_Data[selected_types] = scaler.fit_transform(plot_AllInOne_Data[selected_types])
        # Plot the data
        fig_4, ax_4 = plt.subplots()
        for type in selected_types:
            ax_4.plot(plot_AllInOne_Data['Trade Date'], plot_AllInOne_Data[type], label=type)
        # Enhancements
        ax_4.set_title("Normalized Electricity Data Over Time")
        ax_4.set_xlabel("Trade Date")
        ax_4.set_ylabel("Normalized Value")
        ax_4.legend()
        plt.xticks(rotation=45)
        # Display the plot
        st.pyplot(fig_4)
    else:
        st.write("Please select at least one type to plot.")

    # New graph for moving average
    st.write("### Moving Average Graph")

    # User selects variable and moving average window
    selected_variable = st.selectbox("Select variable for moving average", options=types)
    moving_average_window = st.slider("Select moving average window (weeks)", min_value=20, max_value=100, value=20)

    # Prepare the data for the moving average plot
    if selected_variable:
        AllInOne_Data['Trade Date'] = pd.to_datetime(AllInOne_Data['Trade Date'])
        AllInOne_Data.set_index('Trade Date', inplace=True)
        plot_data_ma = AllInOne_Data[[selected_variable]].dropna()

        # Calculate weekly mean
        weekly_mean = plot_data_ma[selected_variable].resample('W').mean()

        # Broadcast the weekly mean to all values in each week
        filled_df = plot_data_ma.copy()
        filled_df['Week'] = filled_df.index.strftime('%U')  # Add a new column to store the week number
        filled_df[selected_variable] = filled_df.groupby('Week')[selected_variable].transform(lambda x: x.fillna(x.mean()))

        # Calculate the rolling mean with the selected window
        filled_df[f'rolling_mean_{moving_average_window}'] = filled_df[selected_variable].rolling(window=moving_average_window).mean()

        # Plot the data
        fig_ma, ax_ma = plt.subplots()
        ax_ma.plot(filled_df.index, filled_df[selected_variable], label=selected_variable)
        ax_ma.plot(filled_df.index, filled_df[f'rolling_mean_{moving_average_window}'], label=f'Rolling Mean ({moving_average_window} weeks)', linestyle='--')

        # Enhancements
        ax_ma.set_title(f"{selected_variable} with {moving_average_window}-Week Rolling Mean")
        ax_ma.set_xlabel("Trade Date")
        ax_ma.set_ylabel(selected_variable)
        ax_ma.legend()
        plt.xticks(rotation=45)

        # Display the plot
        st.pyplot(fig_ma)
    else:
        st.write("Please select a variable to plot.")

def models_overview():
    st.title("Models Overview")
    
    st.write("""
    In this project, various models are employed to predict electricity prices and develop trading strategies. The models are categorized into those used for predicting the sign (direction) of price changes and those used for predicting the actual price. Below is a detailed description of each model used:
    """)

    # Predicting the Sign
    st.header("Predicting the Sign")
    st.subheader("Using Deep Learning:")
    st.write("""
    - **GRU Sign Detection**: 
      A Gated Recurrent Unit (GRU) model used to predict whether the next day's price will go up or down. 

      **Components**:
        - Update gate
        - Reset gate
        - Current memory content
        - Final memory at current time step

      **Special Features**:
        - Simplified architecture compared to LSTM
        - Faster training due to fewer parameters

      **Use Case in Predicting the Sign**:
        - To capture the sequential dependencies in electricity price changes and predict the direction.

      ![GRU Structure](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Gated_Recurrent_Unit%2C_base_type.svg/1920px-Gated_Recurrent_Unit%2C_base_type.svg.png)
    """)
    st.write("""
    - **LSTM Sign Detection**: 
      A Long Short-Term Memory (LSTM) model used for the same purpose as the GRU model.

      **Components**:
        - Forget gate
        - Input gate
        - Output gate
        - Cell state

      **Special Features**:
        - Ability to capture long-term dependencies
        - Effective in handling vanishing gradient problems

      **Use Case in Predicting the Sign**:
        - To utilize its memory capabilities for more accurate prediction of price direction over longer periods.

      ![LSTM Structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
    """)

    st.subheader("Using Regression Models:")
    st.write("""
    - **Linear Regression**: 
      A basic regression model to predict the direction of the price change.

      **Components**:
        - Dependent variable
        - Independent variables
        - Coefficients
        - Intercept

      **Special Features**:
        - Simple implementation
        - Provides a baseline for comparison

      **Use Case in Predicting the Sign**:
        - To offer a straightforward approach to predicting price direction based on linear relationships.
        - Used the direction of lagged prices as a variable to predict the direction of the next day's price.

      ![Linear Regression](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA/QAAAIjCAYAAACtaVBBAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADSWklEQVR4nOzdeXgT5doG8DsttKWUFlpKWVooFlR2FUWRrwKCgKIWC7Irm4ALQtlcjggU8aCyCspRXMCjlLWV6vEom0URXFAWFxABi0BlKVvLDk3f7485kyZpJplJJskkvX/XlQs6mcy8mUwm87zL85qEEAJEREREREREFFBC/F0AIiIiIiIiItKOAT0RERERERFRAGJAT0RERERERBSAGNATERERERERBSAG9EREREREREQBiAE9ERERERERUQBiQE9EREREREQUgBjQExEREREREQUgBvREREREREREAYgBPREZ0sGDB2EymbBkyRJ/F4U8MHjwYCQnJ/u7GD7VoUMHdOjQwa3XJicnY/DgwbqWxxumTp0Kk8lks0zvslfEcydQafmsSktL0bx5c7z88sveLZQdT84nR+c7AadOnULVqlXx3//+199FIarQGNATkc8tWbIEJpMJP/74o7+L4jXyDaD8qFy5MpKTkzF69GicPXvW38ULenKFkMlkwvTp0x2uM2DAAJhMJkRFRfm4dJ6zPrdCQkJQt25ddOnSBZs2bfJ30TT5+++/MXXqVOzcudPfRTGMs2fPIiIiAiaTCXv27HF7OwsXLjRkheiyZctw+PBhjBo1CoDtuezsEWjntl4GDx5scxyioqJw3XXXoVevXsjOzkZpaanb287KysK8efPcfn1cXBwee+wxvPjii25vg4g8V8nfBSAicqRBgwa4dOkSKleu7O+ieORf//oXoqKicOHCBWzcuBELFizA9u3b8c033/i7aD7xzjvveHTD6amIiAgsW7YMkyZNsll+4cIF5ObmIiIiwk8l89w999yDRx99FEII5OfnY+HChbj77rvx2Wef4d577/V5efbu3YuQEG3tBH///TcyMzORnJyMm266yeY5f587/rJq1SqYTCbUrl0bS5cuVayQcmXhwoWoWbOm4Xp8zJw5E3379kVMTAwA4MMPP7R5/t///jfWr19fbnmTJk082q8n59OkSZPw3HPPebR/T4SHh+Pdd98FAFy6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ly6YN28enn76aURFReGFF16wvDettFwf1di6dSuaN2/uVmu50vXX0zJmZWXh3LlzGDlyJEwmE1577TWkp6fjzz//dFlONeeY1uu2Vo888gjWrVuH9evX4/rrrweg7pi88MILKCoqwpEjRyzfHTmfSHFxMd59913069cPw4cPx7lz5/Dee++ha9eu+OGHH8oNj2ndujXmzp2L3377Dc2bN9flfRGRBoKIyMcWL14sAIht27YprpOfny8AiMWLF1uWDRo0SAAQ06ZNs1n35ptvFq1bt7b8vXnzZgFALF261Ga9L774otzyixcvltv3yJEjRWRkpLh8+bJlWfv27QUA8dZbb6l6j1OmTBEAxN69e0VhYaE4ePCgeP/990WVKlVEfHy8uHDhghBCiNLSUtG4cWPRtWtXUVpaalOuhg0binvuucey7IEHHhCRkZGioKDAsmzfvn2iUqVKwv5yDkCEhISI3377zWb5sGHDRJ06dcTJkydtlvft21fExMRYjkdaWppo1qyZ0/cYExMjnnrqKafrDBo0SDRo0MDy95o1awQAMX36dJv1evXqJUwmk9i/f7/NewgLC7NZtmvXLgFALFiwwOl+5fNn5syZ4tdffxUAxObNm4UQQrz55psiKipKXLhwQQwaNEhUrVrV8roTJ06IsLAw0aVLF2E2my3L33jjDQFAvP/++0IIIa5evSpq1aolbrrpJnHlyhXLeosWLRIARPv27S3LPvzwQxESEmLZv+ytt94SAMSWLVssyxo0aCAGDRrk9L3Jx2bYsGGisLBQnDhxQnz//feiU6dOAoCYPXu2EEKIvLw8AUBcd911Nue5lnOuR48eIiIiQvz111+WZbt37xahoaHlzjn7sk+ePFkAEDk5OeXKL+9327Zt5b7nMn+dO2+//bYAIH755Reb5U2bNhV333235e9WrVqJ7t27O92WVi1atBADBgyw/P2Pf/xD1KxZU1y7ds2ybN++fSIkJEQ89NBDNueoEMLm82zWrJnNeSiTr0325Otyfn6+ZZna66P9Z6UkMTFR9OzZ0+k6Tz31VLnyObv+ultG+RoRFxcnTp8+bVmem5srAIhPP/3UsszRMVN7jmm5bjtif42yt2PHDgFAjB071rJM7THp3r27w8+tpKTE5romhBBnzpwRCQkJYujQoeXW37p1qwAgVqxY4fL9EJH+2OWeiALO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LmrUqCFat24tMjMzLcdU7XsTwvnnK5OnrVu1apXD59Wec1999ZVo3bq1CAsLE9ddd5146623HH7nHJX91KlTYtSoUaJevXoiLCxMJCYmikGDBtlMm5ibmyuaNm1qmcZL/s7769yRFRcXiypVqggA4qOPPir3/PTp00WbNm1E9erVRZUqVcSNN94oXn75ZXH16lXLOvK1ztG0fLLs7GwBQLz33nuK62zatEkAEK+//rpl2fvvvy9uvvlmy/nUvn17sX79esvzx44dE927dxfVqlUrN5XiTz/9JG6//XYRFhYm6tevL+bMmeNw2jq110e109YJIUTLli3FsGHDFJ9XmrZO6frrbhmdXSMAiClTplj+Vpq2Tu05pva67Yg8Xav8iIyMFMnJyaJnz55i9erV5aYu1HJMzp8/L/r37y+qV68uAFiOT2lpqfjnP/8pGjRoIMLDw8XNN98s/vOf/zj8nPfs2SMAiA0bNrh8L0TkHSYhDFSNSUREmvXo0UPX6bOIiLzlww8/xFNPPYVDhw6V6/VQkQTLdTsjIwNff/01fvrpJ7bQE/kJx9ATEQWQS5cu2fy9b98+/Pe//0WHDh38UyAiIg0GDBiA+vXr48033/R3UXwmWK/bp06dwrvvvovp06czmCfyI7bQExEFkDp16mDw4MGWOdH/9a9/4cqVK9ixY4fD+cKJiMi/eN0mIm9iUjwiogDSrVs3LFu2DMeOHUN4eDjatm2Lf/7zn7wpJCIyKF63icib2EJPREREREREFIA4hp6IiIiIiIgoADGgJyIiIiIiIgpAHEPvQmlpKf7++29Uq1aNGTyJiIiIiIjI64QQOHfuHOrWrYuQEOV2eAb0Lvz9999ISkrydzGIiIiIiIiogjl8+DASExMVn2dA70K1atUASAcyOjraz6UhIiIiIiKiYFdcXIykpCRLPKqEAb0Lcjf76OhoBvRERERERETkM66GfTMpHhEREREREVEAYkBPREREREREFIAY0BMREREREREFII6hJyIiIiIiG0IIlJSUwGw2+7soREEpNDQUlSpV8nhqdAb0RERERERkcfXqVRw9ehQXL170d1GIglpkZCTq1KmDsLAwt7fBgJ6IiIiIiAAApaWlyM/PR2hoKOrWrYuwsDCPWxCJyJYQAlevXkVhYSHy8/PRuHFjhIS4NxqeAT0REREREQGQWudLS0uRlJSEyMhIfxeHKGhVqVIFlStXxl9//YWrV68iIiLCre0wKR4REREREdlwt7WQiNTT43vGbyoRERERERFRAGJAT0RERERERBSAGNATERERERE5kJycjHnz5qlef9OmTTCZTDh79qzXyqRkyZIlqF69us/3S/7FgJ6IiIiIiAKayWRy+pg6dapb2922bRtGjBihev0777wTR48eRUxMjFv78zWtFRZkPMxyT0REREREujObgc2bgaNHgTp1gNRUIDTUO/s6evSo5f8rVqzA5MmTsXfvXsuyqKgoy/+FEDCbzahUyXUoFB8fr6kcYWFhqF27tqbXEHmCLfREREREFBTMZmDTJmDZMulfs9nfJaq4cnKA5GSgY0egf3/p3+Rkabk31K5d2/KIiYmByWSy/P3777+jWrVq+Pzzz9G6dWuEh4fjm2++wYEDB5CWloaEhARERUXhtttuw4YNG2y2a9+CbTKZ8O677+Khhx5CZGQkGjdujE8++cTyvH2Xe7kb/Nq1a9GkSRNERUWhW7duNhUQJSUlGD16NKpXr464uDg8++yzGDRoEHr06OH0PS9ZsgT169dHZGQkHnroIZw6dcrmeVfvr0OHDvjrr78wduxYS08GADh16hT69euHevXqITIyEi1atMCyZcu0fBzkQwzoiYiIiCjg+TqAJGU5OUCvXsCRI7bLCwqk5f76TJ577jm88sor2LNnD1q2bInz58/jvvvuw8aNG7Fjxw5069YNDzzwAA4dOuR0O5mZmejduzd+/vln3HfffRgwYABOnz6tuP7Fixcxa9YsfPjhh/j6669x6NAhTJgwwfL8q6++iqVLl2Lx4sXYsmULiouLsWbNGqdl+P777zFs2DCMGjUKO3fuRMeOHTF9+nSbdVy9v5ycHCQmJmLatGk4evSopZLh8uXLaN26NT777DP8+uuvGDFiBB555BH88MMPTstEfiLIqaKiIgFAFBUV+bsoRERERORAdrYQJpMQgO3DZJIe2dn+LmHguHTpkti9e7e4dOmSW68vKREiMbH8Z2H9mSQlSet5y+LFi0VMTIzl77y8PAFArFmzxuVrmzVrJhYsWGD5u0GDBmLu3LmWvwGISZMmWf4+f/68ACA+//xzm32dOXPGUhYAYv/+/ZbXvPnmmyIhIcHyd0JCgpg5c6bl75KSElG/fn2RlpamWM5+/fqJ++67z2ZZnz59bN63O+9PSffu3cX48eNdrkfaOPu+qY1D2UJPRERERAHLbAbGjJHCRXvysowMdr/3lc2by7fMWxMCOHxYWs/Xbr31Vpu/z58/jwkTJqBJkyaoXr06oqKisGfPHpct9C1btrT8v2rVqoiOjsaJEycU14+MjERKSorl7zp16ljWLyoqwvHjx9GmTRvL86GhoWjdurXTMuzZswe33367zbK2bdvq8v7MZjNeeukltGjRArGxsYiKisLatWtdvo78g0nxiIiIiAKML5ONGZ2WALJDB58Vq8KyGhquy3p6qlq1qs3fEyZMwPr16zFr1iw0atQIVapUQa9evXD16lWn26lcubLN3yaTCaWlpZrWF45qoHTm7vubOXMmXn/9dcybNw8tWrRA1apVkZGR4fJ15B8M6ImIiIgCSE6O1CJtHcQmJgKvvw6kp/uvXP5i5ACyIqpTR9/1vGnLli0YPHgwHnroIQBSi/bBgwd9WoaYmBgkJCRg27ZtuOuuuwBILeTbt2/HTTfdpPi6Jk2a4Pvvv7dZ9t1339n8reb9hYWFwWzXfWXLli1IS0vDwIEDAQClpaX4448/0LRpU3feInkZu9wTERERBQijJhvzp0AKICuC1FSpgul/CdPLMZmApCRpPX9r3LgxcnJysHPnTuzatQv9+/d32tLuLU8//TRmzJiB3Nxc7N27F2PGjMGZM2csWecdGT16NL744gvMmjUL+/btwxtvvIEvvvjCZh017y85ORlff/01CgoKcPLkScvr1q9fj61bt2LPnj0YOXIkjh8/rv8bJ10EVED/9ddf44EHHkDdunVhMplcZn8EpKkjbrnlFoSHh6NRo0ZYsmSJ18tJREREpDeOFXcskALIiiA0VOotApT/TOS/580zxhCROXPmoEaNGrjzzjvxwAMPoGvXrrjlllt8Xo5nn30W/fr1w6OPPoq2bdsiKioKXbt2RUREhOJr7rjjDrzzzjt4/fXX0apVK6xbtw6TJk2yWUfN+5s2bRoOHjyIlJQUxMfHAwAmTZqEW265BV27dkWHDh1Qu3Ztl1Pokf+YhC8GcOjk888/x5YtW9C6dWukp6fj448/dnpy5efno3nz5nj88cfx2GOPYePGjcjIyMBnn32Grl27qtpncXExYmJiUFRUhOjoaJ3eCREREZE2mzZJU7G5kpdX8caKyz0XANsKDzmAXL26Yg5HcMfly5eRn5+Phg0bOg0oXXE0NCQpSQrm+Vk4V1paiiZNmqB379546aWX/F0c8iJn3ze1cWhAjaG/9957ce+996pe/6233kLDhg0xe/ZsANJYk2+++QZz585VHdATERERGQHHiitLT5eCdke5BRhA+kd6OpCWxuSNavz1119Yt24d2rdvjytXruCNN95Afn4++vfv7++iUQAIqIBeq2+//RadO3e2Wda1a1dkZGQovubKlSu4cuWK5e/i4mJvFY+IiIhIkX0m+1q11L2uoo4VZwBpPKGhFa+3iDtCQkKwZMkSTJgwAUIING/eHBs2bECTJk38XTQKAEEd0B87dgwJCQk2yxISElBcXIxLly6hSpUq5V4zY8YMZGZm+qqIREREROU46q5crx4QFwecPu14HL3JJLVIV+Sx4gwgKRAlJSVhy5Yt/i4GBaiASornC88//zyKioosj8OHD/u7SERERFSBKGWy//tv4NQpKZg3erIxIiLyjaBuoa9du3a5KRaOHz+O6Ohoh63zABAeHo7w8HBfFI+IiIjIhqtM9iYTEBsLVKnCseJERBTkAX3btm3x3//+12bZ+vXr0bZtWz+ViIiIiEjZ5s3lW+atCSG10m/YILXEc6w4EVHFFlAB/fnz57F//37L3/n5+di5cydiY2NRv359PP/88ygoKMC///1vAMDjjz+ON954A8888wyGDh2KL7/8EitXrsRnn33mr7dAREREpEhthvpNm4CpUxnEExFVdAE1hv7HH3/EzTffjJtvvhkAMG7cONx8882YPHkyAODo0aM4dOiQZf2GDRvis88+w/r169GqVSvMnj0b7777LqesIyIiIkNSm6F++nQgOVkab09ERBWXSQhHo7RIVlxcjJiYGBQVFSE6OtrfxSEiIqIgYj813Z13AikpQEGB43H01uREeKtXc+w86efy5cvIz89Hw4YNERER4e/iEAU1Z983tXFoQLXQExEREQWLnByplb1jR6B/f+nflBSgXz/peftM9vbkgD8jQ6oYICLnTCYT1qxZ4+9iVDhTp07FTTfd5O9i6ELrezl48CBMJhN27tzptTIxoCciIiLyMaWp6QoKgFmzgAkTpHnnXRECOHxYauUnqugGDx6MHj16KD5/9OhR3Hvvvb4rkEYmk8nyiI6Oxm233Ybc3Fx/F8tjEyZMwMaNG72+n+TkZJhMJixfvrzcc82aNYPJZMKSJUu8Xg5fY0BPRERE5EOupqYDgOXLgQMHgEmT1G1TbTI9ooqsdu3afp+eWgiBkpISxecXL16Mo0eP4scff0S7du3Qq1cv/PLLL14t09WrV726/aioKMTFxXl1H7KkpCQsXrzYZtl3332HY8eOoWrVqj4pg68xoCciIiLyITVT0x0+DGzdCnTqpG6bapPpEVVk1l3u5a7QOTk56NixIyIjI9GqVSt8++23Nq/55ptvkJqaiipVqiApKQmjR4/GhQsXLM9/+OGHuPXWW1GtWjXUrl0b/fv3x4kTJyzPb9q0CSaTCZ9//jlat26N8PBwfPPNN4plrF69OmrXro3rr78eL730EkpKSpCXl2d5/vDhw+jduzeqV6+O2NhYpKWl4eDBg5bnS0pKMHr0aFSvXh1xcXF49tlnMWjQIJueCx06dMCoUaOQkZGBmjVrWhKG//rrr7j33nsRFRWFhIQEPPLIIzh58qTldatXr0aLFi1QpUoVxMXFoXPnzpZjsWnTJrRp0wZVq1ZF9erV0a5dO/z1118AyndTLy0txbRp05CYmIjw8HDcdNNN+OKLLyzPq/1sHBkwYAC++uorHD582LLs/fffx4ABA1Cpku0Eb4cOHUJaWhqioqIQHR2N3r174/jx4zbrvPLKK0hISEC1atUwbNgwXL58udw+3333XTRp0gQRERG48cYbsXDhQpfl1BMDeiIiIiIfUtuafvSoNL98YqLyeHqTCUhKktYj8qpbb5VORl8/br3Vq2/rhRdewIQJE7Bz505cf/316Nevn6UF/cCBA+jWrRt69uyJn3/+GStWrMA333yDUaNGWV5/7do1vPTSS9i1axfWrFmDgwcPYvDgweX289xzz+GVV17Bnj170LJlS5flKikpwXvvvQcACAsLs+yra9euqFatGjZv3owtW7YgKioK3bp1s7Syv/rqq1i6dCkWL16MLVu2oLi42GHegA8++ABhYWHYsmUL3nrrLZw9exZ33303br75Zvz444/44osvcPz4cfTu3RuANFyhX79+GDp0KPbs2YNNmzYhPT3d0uOgR48eaN++PX7++Wd8++23GDFiBEwKF67XX38ds2fPxqxZs/Dzzz+ja9euePDBB7Fv3z7Vn42ShIQEdO3aFR988AEA4OLFi1ixYgWGDh1qs15paSnS0tJw+vRpfPXV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)
    """)
    st.write("""
    - **Random Forest**: 
      An ensemble learning method using multiple decision trees to improve prediction accuracy.

      **Components**:
        - Multiple decision trees
        - Bagging
        - Majority voting

      **Special Features**:
        - Reduces overfitting by averaging multiple decision trees
        - Handles large datasets with higher dimensionality

      **Use Case in Predicting the Sign**:
        - To enhance the accuracy of direction prediction by leveraging ensemble methods.
        - Used the direction of lagged prices as a variable to predict the direction of the next day's price.

      ![Random Forest](https://miro.medium.com/v2/resize:fit:1010/1*R3oJiyaQwyLUyLZL-scDpw.png)
    """)

    # Predicting the Price
    st.header("Predicting the Price")
    st.subheader("Naive Forecast:")
    st.write("""
    A simple model that uses the previous day's price as the forecast for the next day. 

    **Components**:
      - Previous day's price

    **Special Features**:
      - Minimal computation
      - Serves as a baseline for more complex models

    **Use Case in This Project**:
      - To provide a simple benchmark for evaluating the performance of other models.
    """)

    st.subheader("Random Forest:")
    st.write("""
    An ensemble learning method using multiple decision trees to predict the actual price.

    **Components**:
      - Multiple decision trees
      - Bagging
      - Aggregation of results

    **Special Features**:
      - Captures complex interactions between features
      - Provides robust and accurate predictions

    **Use Case in This Project**:
      - To predict actual electricity prices by capturing nonlinear relationships in the data.
      - Using features/variables such as natural gas prices and temperature to predict electricity prices.
    """)

    st.subheader("Machine Learning Models:")
    st.write("""
    - **ARIMA**: 
      An Autoregressive Integrated Moving Average model used for time series forecasting.

      **Components**:
        - **Autoregressive (AR) terms**: These represent past values of the forecast variable in a regression equation. AR terms model the dependency of the variable on its own past values, where "AR(p)" indicates "p" past values are used.
        - **Integrated (I) terms**: This reflects the differencing of raw observations in time series data to achieve stationarity. The "I(d)" term denotes the number of times differencing is applied to the series for achieving stationarity.
        - **Moving Average (MA) terms**: These involve the dependency between an observation and a residual error from a moving average model applied to lagged observations. MA terms in an "MA(q)" model use "q" past errors to forecast future values.
      
      **ACF and PACF in Time Series Modeling**:
        - **Autocorrelation Function (ACF)**: ACF measures the correlation between a time series and its lagged values. For AR models, ACF helps determine the order "p" by showing significant correlations at lags up to "p". For MA models, ACF drops off after lag "q", indicating the order of the moving average.
        - **Partial Autocorrelation Function (PACF)**: PACF measures the correlation between a time series and its lagged values while adjusting for the effects of intervening lags. PACF is useful in determining the order of AR models, as it shows direct effects of past lags on the current value, without the indirect effects of shorter lags.
            ![Autoregressive](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA+oAAAKqCAYAAACtsaQKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAB7rElEQVR4nOzde1xVVcL/8e8B5QAWF5WLJImXJqVUDJJQS0tGULvYYyaNjZfHsDHJUSyVntTUirGLY6ZFNt76paNZaXYZijBrKvM65FTqpGlewSscAQOF/fuj8eSRAwJyOFv9vF+v/ZKz9tprr709Jd99WctiGIYhAAAAAABgCh7u7gAAAAAAAPgNQR0AAAAAABMhqAMAAAAAYCIEdQAAAAAATISgDgAAAACAiRDUAQAAAAAwEYI6AAAAAAAmQlAHAAAAAMBECOoAAAAAAJgIQR0AANS7tWvXymKxaO3atXXarsVi0VNPPVWnbQIAUN8I6gAAXKRXXnlFFotFsbGxF9XORx99RMisBs4TAOByR1AHAOAiLVmyRBEREdqwYYN27txZ63Y++ugjTZ06tQ57dnmq6jydOnVKTz75ZD33CACAukVQBwDgIuzevVtff/21Zs6cqaCgIC1ZssTdXXKp4uJip+VnzpxRaWlpPfemIm9vbzVo0MDd3QAA4KIQ1AEAuAhLlixRYGCg+vbtq/vuu69CUK/sXew9e/bIYrFo0aJFkqShQ4dq7ty5kn59z/rsclZRUZHGjRun8PBwWa1WXX/99XrhhRdkGEaFPr355pvq3LmzfH19FRgYqNtuu02ffPKJQ51XXnlFN9xwg6xWq8LCwjRq1Cjl5+c71OnRo4duvPFGbd68Wbfddpt8fX31xBNP2Pv+wgsvaNasWWrdurWsVqt++OEHSdL27dt13333qXHjxvL29lZMTIxWr159wXP5z3/+UwMGDNC1114rq9Wq8PBwjR07VqdOnbLXudB5cvaO+r/+9S/17t1bfn5+uuqqq9SzZ0998803DnUWLVoki8Wir776SqmpqQoKClKjRo1077336siRIxfsOwAAdYlLzgAAXIQlS5bof/7nf+Tl5aUHHnhAr776qjZu3Kibb765Ru08/PDDOnjwoLKysvT//t//c1hnGIbuvvtuffbZZxo+fLiioqL08ccf6/HHH9eBAwf017/+1V536tSpeuqpp9SlSxdNmzZNXl5eWr9+vdasWaNevXpJkp566ilNnTpV8fHxGjlypHbs2GHv91dffaWGDRva2zt27Jh69+6tpKQkPfjggwoJCbGvW7hwoX755ReNGDFCVqtVjRs31vfff6+uXbvqmmuu0cSJE9WoUSO99dZb6tevn9555x3de++9lZ6DFStWqLi4WCNHjlSTJk20YcMGvfzyy9q/f79WrFhxwfPkzPfff69bb71Vfn5+Gj9+vBo2bKjXXntNPXr00Oeff15hXIFHH31UgYGBmjJlivbs2aNZs2YpJSVFy5cvv+C+AACoMwYAAKiVTZs2GZKMrKwswzAMo7y83GjevLnx5z//2V7ns88+MyQZn332mcO2u3fvNiQZCxcutJeNGjXKcPZP86pVqwxJxtNPP+1Qft999xkWi8XYuXOnYRiG8eOPPxoeHh7Gvffea5SVlTnULS8vNwzDMA4fPmx4eXkZvXr1cqgzZ84cQ5KxYMECe1n37t0NSUZGRobTvvv5+RmHDx92WNezZ0+jffv2xi+//OKw7y5duhjXXXddleeluLi4wrGnp6cbFovF+Pnnny94ngzDMCQZU6ZMsX/u16+f4eXlZezatctedvDgQePqq682brvtNnvZwoULDUlGfHy8/VwZhmGMHTvW8PT0NPLz853uDwAAV+DRdwAAamnJkiUKCQnR7bffLunXx64HDhyoZcuWqaysrM7289FHH8nT01OjR492KB83bpwMw9A//vEPSdKqVatUXl6uyZMny8PD8Z/4s4+Hf/rppyotLdWYMWMc6iQnJ8vPz08ffvihw3ZWq1XDhg1z2q/+/fsrKCjI/vn48eNas2aN7r//fp08eVJHjx7V0aNHdezYMSUkJOjHH3/UgQMHKj1OHx8f+89FRUU6evSounTpIsMw9K9//auqU+RUWVmZPvnkE/Xr10+tWrWylzdr1kx/+MMf9OWXX8pmszlsM2LECIdH6W+99VaVlZXp559/rvH+AQCoLYI6AAC1UFZWpmXLlun222/X7t27tXPnTu3cuVOxsbHKy8tTdnZ2ne3r559/VlhYmK6++mqH8nbt2tnXS9KuXbvk4eGhyMjIKtuSpOuvv96h3MvLS61ataoQSK+55hp5eXk5batly5YOn3fu3CnDMDRp0iQFBQU5LFOmTJEkHT58uNK+7d27V0OHDlXjxo111VVXKSgo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AZKk9PR03Xzzzbr66qsVHBysfv36aceOHe7uFlCpv/zlL/bpNQGzOHDggB588EE1adJEPj4+at++vTZt2uTubgGSpLKyMk2aNEktW7aUj4+PWrdurenTp4vxneEuX3zxhe666y6FhYXJYrFo1apVDusNw9DkyZPVrFkz+fj4KD4+Xj/++KN7OnsFueKC+vLly5WamqopU6Zoy5Yt6tixoxISEnT48GF3dw3Q559/rlGjRumbb75RVlaWTp8+rV69eqmoqMjdXQMq2Lhxo1577TV16NDB3V0B7E6cOKGuXbuqYcOG+sc//qEffvhBL774ogIDA93dNUCSNGPGDL366quaM2eOtm3bphkzZui5557Tyy+/7O6u4QpVVFSkjh07au7cuU7XP/fcc5o9e7YyMjK0fv16NWrUSAkJCfrll1/quadXlituerbY2FjdfPPNmjNnjiSpvLxc4eHhevTRRzVx4kQ39w5wdOTIEQUHB+vzzz/Xbbfd5u7uAHaFhYW66aab9Morr+jpp59WVFSUZs2a5e5uAZo4caK++uor/fOf/3R3VwCn7rzzToWEhGj+/Pn2sv79+8vHx0dvvvmmG3sGSBaLRStXrlS/fv0k/Xo3PSwsTOPGjdNjjz0mSSooKFBISIgWLVqkpKQkN/b28nZF3VEvLS3V5s2bFR8fby/z8PBQfHy81q1b58aeAc4VFBRIkho3buzmngCORo0apb59+zr8/xQwg9WrVysmJkYDBgxQcHCwOnXqpNdff93d3QLsunTpouzsbP3nP/+RJH377bf68ssv1bt3bzf3DKho9+7dys3Ndfj33t/fX7GxseQnF2vg7g7Up6NHj6qsrEwhISEO5SEhIdq+fbubegU4V15erjFjxqhr16668cYb3d0dwG7ZsmXasmWLNm7c6O6uABX89NNPevXVV5WamqonnnhCGzdu1OjRo+Xl5aUhQ4a4u3uAJk6cKJvNprZt28rT01NlZWV65plnNGjQIHd3DaggNzdXkpzmp7Pr4BpXVFAHLiWjRo3Sd999py+//NLdXQHs9u3bpz//+c/KysqSt7e3u7sDVFBeXq6YmBg9++yzkqROnTrpu+++U0ZGBkEdpvDWW29pyZIlWrp0qW644Qbl5ORozJgxCgsL4zsKwO6KevS9adOm8vT0VF5enkN5Xl6eQkND3dQroKKUlBR98MEH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iRP197//Xb1799bo0aPVuHFjLV68WLt379Y777xT4XH2mujevbsefvhhpaenKycnR7169VLDhg31448/asWKFXrppZd03333Od32zjvv1Lvvvqt7771Xffv21e7du5WRkaHIyEgVFhZW6zydr2PHjhoyZIjmzZun/Px8de/eXRs2bNDixYvVr18/3X777bU+VgDAFcCtY84DAFBHzk7P5mx6sHPNnz/fuO666wyr1Wq0bdvWWLhwodNp1SQZo0aNctrG119/bURHRxteXl4O03U5a2f16tVGhw4dDG9vbyMiIsKYMWOGsWDBAkOSsXv3bnu9C03PtnnzZkOSMWnSpErr7Nmzx5BkjB071l725ptvGq1atTK8vLyMqKgo4+OPP64wPVtVx2QYhvHpp58aXbt2NXx8fAw/Pz/jrrvuMn744YcK+//555+NwYMHG0FBQYbVajVatWpljBo1ymHqs127dhn33XefERAQYHh7exudO3c2PvjgA4d2zk7PtmLFigr7GDJkiNGoUaNKz8G8efOM6Ohow8fHx7j66quN9u3bG+PHjzcOHjxor3P+uS4vLzeeffZZo0WLFobVajU6depkfPDBBzU6T87+7k+fPm1MnTrVaNmypdGwYUMjPDzcSEtLM3755ReHei1atDD69u1b4VhqMmUfAODyYjEMRikBAAAAAMAseEcdAAAAAAATIagDAAAAAGAiBHUAAAAAAEzEpUH9iy++0F133aWwsDBZLJZqTX+zdu1a3XTTTbJarWrTpo0WLVpUoc7cuXMVEREhb29vxcbGasOGDXXfeQAAAAAA3MClQb2oqEgdO3bU3Llzq1V/9+7d6tu3r26//Xbl5ORozJgxeuihh/Txxx/b6yxfvlypqamaMmWKtmzZoo4dOyohIUGHDx921WEAAAAAAFBv6m3Ud4vFopUrV6pfv36V1pkwYYI+/PBDfffdd/aypKQk5efnKzMzU5IUGxurm2++WXPmzJEklZeXKzw8XI8++qgmTpzo0mMAAAAAAMDVGri7A+dat26d4uPjHcoSEhI0ZswYSVJpaak2b96stLQ0+3oPDw/Fx8dr3bp1lbZbUlKikpIS++fy8nIdP35cTZo0kcViqduDAAAAAADgPIZh6OTJkwoLC5OHR9UPt5sqqOfm5iokJMShLCQkRDabTadOndKJEydUVlbmtM727dsr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)
        
      **Special Features**:
        - Effective for univariate time series
        - Captures various types of temporal patterns

      **Use Case in This Project**:
        - To model and forecast the electricity price time series based on past values and past forecast errors.

      ![ARIMA Model](https://pbs.twimg.com/media/GGR3W45akAAvsaf.png)
    """)
    st.write("""
    - **SARIMA**: 
      Seasonal ARIMA model to capture seasonality in the data.

      **Components**:
        - Seasonal autoregressive (SAR) terms
        - Seasonal integrated (SI) terms
        - Seasonal moving average (SMA) terms

      **Special Features**:
        - Incorporates seasonality into ARIMA
        - Models complex seasonal patterns

      **Use Case in This Project**:
        - To predict electricity prices by capturing both seasonal and non-seasonal patterns in the data.

      ![SARIMA Model](https://pbs.twimg.com/media/GBIbeegbkAAJpjz.png)
    """)
    st.write("""
    - **Auto ARIMA (SARIMA)**: 
      Automated selection of the best SARIMA model parameters.

      **Components**:
        - Automatic parameter tuning

      **Special Features**:
        - Simplifies model selection process
        - Enhances model accuracy by choosing optimal parameters

      **Use Case in This Project**:
        - To automate the process of finding the best SARIMA model for electricity price forecasting.

    """)
    st.write("""
    - **GARCH**: 
      Generalized Autoregressive Conditional Heteroskedasticity model to forecast volatility.

      **Components**:
        - Autoregressive terms for variance
        - Moving average terms for variance

      **Special Features**:
        - Models volatility clustering
        - Useful for financial time series

      **Use Case in This Project**:
        - To forecast the volatility of electricity prices, which can inform risk management strategies.

      ![GARCH Model](https://www.investopedia.com/thmb/NAR9L8kawoJ41JwUN-_gC6VEwLc=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/GARCH-9d737ade97834e6a92ebeae3b5543f22.png)
    """)


    st.subheader("Deep Learning Models:")
    st.write("""
    - **GRU**: 
      A Gated Recurrent Unit model for predicting the actual price.

      **Use Case in predicting the Price**:
        - To predict electricity prices by capturing temporal dependencies in the data.

    """)
    st.write("""
    - **LSTM**: 
      A Long Short-Term Memory model for price prediction.

      **Use Case in predicting the Price**:
        - To predict electricity prices by leveraging its long-term memory capabilities.

    """)
    st.write("""
    - **LSTM using Normalized Prices**: 
      LSTM model applied to normalized price data.

      **Components**:
        - Normalized input data

      **Special Features**:
        - Improved training efficiency
        - Enhanced model performance

      **Use Case in This Project**:
        - To achieve better model performance by normalizing the input price data.

    """)
    st.write("""
    - **LSTM Regression**: 
      LSTM model used in a regression setting for price prediction.

      **Components**:
        - Regression output layer

      **Special Features**:
        - Directly predicts continuous price values
        - Suitable for precise price forecasting

      **Use Case in This Project**:
        - To provide accurate price predictions by directly modeling the continuous price values.

    """)

    st.subheader("Model Creation, Training, and Prediction")
    st.write("""
    Here's a basic example of how a model is created, trained, and used for predictions:
    """)

    code = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample data
data = pd.read_csv('electricity_prices.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
    '''
    st.code(code, language='python')

def model_selection():
    AllInOne_Data = load_dataset()
    models_names_sign = {
        "Select a model": "",
        "GRU": "sign_gru_model.keras",
        "LSTM": "sign_LSTM_model.keras",
        "Random Forest": "sign_randomForest_model.pkl",
        "Linear Regression": "sign_linearRegression_model.pkl"
    }

    st.title("Model Selection for direction Prediction")
    st.write("Choose and compare different models for direction prediction.")

    direction_models = list(models_names_sign.keys())
    selected_direction_model = st.selectbox("Select a model", direction_models)

    # Show key metrics for the selected model
    if selected_direction_model:
        model_name = models_names_sign[selected_direction_model]
        metrics,descriptions = calculate_metrics(model_name, AllInOne_Data)
        st.write(f"Metrics for {selected_direction_model}:")
        st.write(pd.DataFrame({'Metric': descriptions.keys(), 'Description': descriptions.values()}))
        st.write(metrics)

    models_names_direction = {
      "Select a model": "",
      "ARIMA":"price_ARIMA_model.pkl",
      "GRU": "price_gru_model.h5",
      "LSTM": "price_lstm_model.h5",
    }

    st.title("Model Selection for price Prediction")
    st.write("Choose and compare different models for price prediction.")

    direction_models = list(models_names_direction.keys())
    selected_direction_model = st.selectbox("Select a model", direction_models)

    # Show key metrics for the selected model
    if selected_direction_model:
        model_name = models_names_direction[selected_direction_model]
        metrics,descriptions = calculate_metrics(model_name, AllInOne_Data)
        st.write(f"Metrics for {selected_direction_model}:")
        st.write(pd.DataFrame({'Metric': descriptions.keys(), 'Description': descriptions.values()}))
        st.write(metrics)

# def model_selection_Price&Direction_forEachOne():
#     models = {
#         "GRU": {
#             "Direction": "sign_gru_model.keras",
#             "Price": "price_gru_model.h5"
#         },
#         "LSTM": {
#             "Direction": "sign_LSTM_model.keras",
#             "Price": ""
#         },
#         "Random Forest": {
#             "Direction": "sign_randomForest_model.pkl",
#             "Price": ""
#         },
#         "Linear Regression": {
#             "Direction": "sign_linearRegression_model.pkl",
#             "Price": ""
#         },
#         "ARIMA": {
#             "Direction": "",
#             "Price": "price_ARIMA_model.pkl"
#         }
#     }

#     for model_type, selection in models.items():
#         st.title(f"Model Selection for {model_type} Prediction")
#         st.write(f"Choose and compare different models for {model_type} prediction.")

#         direction_models = list(selection.keys())
#         selected_direction_model = st.selectbox(f"Select a {model_type} model", direction_models)

#         if selected_direction_model:
#             model_name = selection[selected_direction_model]
#             metrics, descriptions = calculate_metrics(model_name, AllInOne_Data)
#             st.write(f"Metrics for {selected_direction_model}:")
#             st.write(pd.DataFrame({'Metric': descriptions.keys(), 'Description': descriptions.values()}))
#             st.write(metrics)  


def predictions():
    AllInOne_Data = load_dataset()
    st.title("Predictions")
    st.write("Visualize predictions and performance metrics.")

    selected_model = st.selectbox("Select a model for prediction", ["GRU", "LSTM", "ARIMA"])

    if selected_model:
        predictions, confidence_intervals = predict(selected_model, AllInOne_Data)
        st.write("Predictions:")
        st.write(predictions)

        if confidence_intervals:
            st.write("Confidence Intervals:")
            st.write(confidence_intervals)

        # Visualization
        plot_predictions(selected_model, AllInOne_Data)

        # Show key metrics
        metrics = calculate_metrics(selected_model, AllInOne_Data)
        st.write(f"Metrics for {selected_model}:")
        st.write(metrics)

    st.write("Input new data for prediction:")
    recent_data = st.text_area("Enter recent electricity prices (comma separated)", "")
    if st.button("Predict"):
        if recent_data:
            new_data = np.array(recent_data.split(",")).astype(float)
            new_predictions = predict(selected_model, new_data)
            st.write("New Predictions:")
            st.write(new_predictions)

def trading_strategies():
    AllInOne_Data = load_dataset()
    st.title("Trading Strategies")
    st.write("Explain and visualize trading strategies.")

    strategies = ["", "Quantile-based Strategy"]
    selected_strategy = st.selectbox("Select a strategy", strategies)

    if selected_strategy:
        st.write(f"Logic for {selected_strategy}:")
        strategy_logic(selected_strategy)

        st.write("Flowchart/Diagrams:")
        # You can add code to visualize flowcharts or diagrams

        st.write("Performance Metrics:")
        performance_metrics = backtest(selected_strategy, AllInOne_Data)
        st.write(performance_metrics)

        st.write("Equity Curve:")
        plot_equity_curve(selected_strategy, AllInOne_Data)

        st.write("Individual Trades:")
        trades = backtest(selected_strategy, AllInOne_Data, return_trades=True)
        st.write(trades)

def backtesting():
    AllInOne_Data = load_dataset()
    
    st.title("Backtesting")
    st.subheader("Interactive backtesting tool.")
    starting_amount = st.number_input("Starting Amount", value=1000, min_value=1000)
    strategies = ["Percentile Channel Breakout (Mean Reversion)", "Break of Structure"]
    selected_strategy = st.selectbox("Select a strategy for backtesting", strategies)
    
    descriptions(selected_strategy)
    if selected_strategy == "Percentile Channel Breakout (Mean Reversion)":
        backtest_results = run_percentile_strategy(starting_amount, AllInOne_Data)
        
    elif selected_strategy == "Break of Structure":
        st.image("assets\BOS.png")
        backtest_results = run_BOS_strategy(starting_amount, AllInOne_Data)
        
    
    if backtest_results is not None:
        st.write("Backtest completed.")
        st.session_state['backtest_results'] = backtest_results
        st.session_state['strategy_name'] = selected_strategy
        export_results()



def export_results():
    if 'backtest_results' in st.session_state and 'strategy_name' in st.session_state:
        backtest_results = st.session_state['backtest_results']
        strategy_name = st.session_state['strategy_name']
        
        st.write(backtest_results)
        st.title("Export Results")
        st.write("Provide options to export backtesting results.")
        
        export_format = st.radio("Choose export format", ("CSV", "Excel"))

        if st.button("Export"):
            if export_format == "CSV":
                backtest_results.to_csv(f'{strategy_name}_backtest_results.csv')
                st.write("Results exported to CSV.")
            elif export_format == "Excel":
                backtest_results.to_excel(f'{strategy_name}_backtest_results.xlsx')
                st.write("Results exported to Excel.")
    else:
        st.title("No results available for export. Please run a backtest first.")
def contact():
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from dotenv import find_dotenv, load_dotenv
    import os
    import openpyxl

    st.title("Contact Us")
    st.write("User feedback form.")
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    def send_email(name, email, message):
        # Email details
        
        from_email = os.getenv('EMAIL')
        from_password = os.getenv('EMAIL_PASSWORD')
        to_email = os.getenv('EMAIL')


        # Create the email content
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = "New Contact Form Submission"

        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_email, from_password)
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()
            return True
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return False

    st.write("Please fill out the form below to provide your feedback.")

    with st.form(key='contact_form'):
        name = st.text_input(label="Name")
        email = st.text_input(label="Email")
        message = st.text_area(label="Message")

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if send_email(name, email, message):
            st.success("Thank you for your feedback! Your message has been sent.")
        else:
            st.error("Failed to send your message. Please try again later.")

PAGES = {
    "Home": home,
    "Data Exploration": data_exploration,
    "Models Overview": models_overview,
    "Model Selection": model_selection,
    "Predictions": predictions,
    "Trading Strategies": trading_strategies,
    "Backtesting": backtesting,
    "Export Results": export_results,
    'Contact Us': contact
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()
