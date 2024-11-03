# Main Streamlit app file
# Displays UI, handles user inputs, and runs the model

import streamlit as st
import numpy as np
import pandas as pd
from model import model
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pytz

# App Title
st.title("High-Frequency Trading Backtesting")

# User input: Investment amount
input_value_amount = st.number_input("Enter amount of value you would like to invest:", min_value=0.0, max_value=100.0, value=50.0)

# User input: Stock selection
stock_option = st.selectbox("Which stock would you like to select?", ("Tesla", "Ford", "Both"))
ticker_tesla = "TSLA"
ticker_ford = "F"
# Determine ticker based on user selection
ticker = ticker_tesla if stock_option == "Tesla" else (ticker_ford if stock_option == "Ford" else False)

# Set fixed start date for investment
fixed_start_date = datetime(2024, 1, 1)

# User input: Investment duration in days
max_investment_days = 300  # Maximum allowed investment period
investment_days = st.number_input("How many days would you like to invest?", min_value=1, max_value=max_investment_days)

# Calculate the investment end date based on the selected duration
investment_end_date = (fixed_start_date + timedelta(days=investment_days)).replace(tzinfo=None)

# Load stock data for the selected ticker(s)
if ticker:  # Single stock selected
    stock_data = yf.download(ticker, start="2024-1-1", end=investment_end_date.strftime('%Y-%m-%d'), interval='1d')
    df = pd.DataFrame(stock_data)
    st.write(df.columns)
    stock_data = stock_data['(Adj Close, TSLA)']

    stock_data["Tesla_Log_Return"] = np.log(stock_data['Adj Close']/stock_data['Adj Close'].shift(1))
    st.write(stock_data["Tesla_Log_Return"])
else:  # Both stocks selected
    stock_data = yf.download(ticker_tesla, start="2024-1-1", end=investment_end_date.strftime('%Y-%m-%d'), interval='1d')
    stock_data2 = yf.download(ticker_ford, start="2024-1-1", end=investment_end_date.strftime('%Y-%m-%d'), interval='1d')

# Check if the data is available
if ticker or (not ticker and not stock_data.empty and not stock_data2.empty):

    # Handle case where data is not found for the selected stock
    if stock_data.empty and (ticker == ticker_tesla or ticker == ticker_ford):
        st.write(f"No data found for {stock_option} stock. Please check the date range.")
    
    elif not ticker:  # Both stocks selected
        # Flatten multi-level columns if needed for both stocks
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
        if isinstance(stock_data2.columns, pd.MultiIndex):
            stock_data2.columns = ['_'.join(col).strip() for col in stock_data2.columns.values]
        
        # Ensure indices are in datetime format for both stocks
        stock_data.index = stock_data.index.tz_localize(None)
        stock_data2.index = stock_data2.index.tz_localize(None)
        
        # Filter data to the specified investment period
        stock_data_filtered = stock_data.loc[fixed_start_date:investment_end_date]
        stock_data2_filtered = stock_data2.loc[fixed_start_date:investment_end_date]
        
        # Check if filtered data is empty (e.g., due to weekends or holidays)
        if stock_data_filtered.empty or stock_data2_filtered.empty:
            st.write("No trading data available for the selected date range. This may be due to weekends or holidays.")
        else:
            # Detect the 'Close' columns for each stock
            close_column_tesla = next((col for col in stock_data.columns if 'Close' in col), None)
            close_column_ford = next((col for col in stock_data2.columns if 'Close' in col), None)

            # Display data for both stocks if 'Close' columns are available
            if close_column_tesla and close_column_ford:
                st.subheader("Tesla and Ford Stock Data")
                st.write("Tesla Data:")
                st.write(stock_data_filtered)
                st.write("Ford Data:")
                st.write(stock_data2_filtered)

                # Create a Plotly line chart combining Tesla and Ford stock prices
                fig_both = go.Figure()

                # Add Tesla's closing price line
                fig_both.add_trace(go.Scatter(
                    x=stock_data_filtered.index,
                    y=stock_data_filtered[close_column_tesla],
                    mode='lines',
                    name='Tesla Close Price'
                ))

                # Add Ford's closing price line
                fig_both.add_trace(go.Scatter(
                    x=stock_data2_filtered.index,
                    y=stock_data2_filtered[close_column_ford],
                    mode='lines',
                    name='Ford Close Price'
                ))

                # Customize the layout of the combined chart
                fig_both.update_layout(
                    title="Tesla and Ford Stock Prices from {} to {}".format(
                        fixed_start_date.strftime('%Y-%m-%d'), investment_end_date.strftime('%Y-%m-%d')),
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_white"
                )

                # Display the combined chart in Streamlit
                st.plotly_chart(fig_both, use_container_width=True)

            # Display investment summary for both stocks
            st.subheader("Investment Summary")
            st.write(f"Investment Start Date: {fixed_start_date.strftime('%Y-%m-%d')}")
            st.write(f"Investment End Date: {investment_end_date.strftime('%Y-%m-%d')}")
            st.write(f"Duration: {investment_days} days")

    else:  # Single stock selected
        # Flatten multi-level columns if needed
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
        
        # Ensure the index is in datetime format
        stock_data.index = stock_data.index.tz_localize(None)
        
        # Filter data to the specified investment period
        stock_data_filtered = stock_data.loc[fixed_start_date:investment_end_date]
        
        # Check if filtered data is empty (e.g., due to weekends or holidays)
        if stock_data_filtered.empty:
            st.write("No trading data available for the selected date range. This may be due to weekends or holidays.")
        else:
            # Detect the 'Close' column
            close_column = next((col for col in stock_data.columns if 'Close' in col), None)
            
            if close_column:
                st.subheader(f"{stock_option} Stock Data")
                st.write(stock_data_filtered)

                # Create a Plotly line chart for the selected stock
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_data_filtered.index,
                    y=stock_data_filtered[close_column],
                    mode='lines',
                    name='Close Price'
                ))

                # Customize the layout of the chart
                fig.update_layout(
                    title=f"{stock_option} Stock Price from {fixed_start_date.strftime('%Y-%m-%d')} to {investment_end_date.strftime('%Y-%m-%d')}",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_white"
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

            # Display investment summary
            st.subheader("Investment Summary")
            st.write(f"Investment Start Date: {fixed_start_date.strftime('%Y-%m-%d')}")
            st.write(f"Investment End Date: {investment_end_date.strftime('%Y-%m-%d')}")
            st.write(f"Duration: {investment_days} days")
