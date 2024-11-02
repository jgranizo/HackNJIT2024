# main Streamlit app file
# display UI, handle user inputs and run model
import streamlit as st
import pandas as pd
from model import model
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pytz

st.title("High-Frequency Trading Backtesting")

# User input for investment amount
input_value_amount = st.number_input("Enter amount of value you would like to invest:", min_value=0.0, max_value=100.0, value=50.0)

# Fixed start date
fixed_start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
st.write(f"Fixed Start Date: {fixed_start_date.strftime('%Y-%m-%d')}")

# User input for investment duration in days
max_investment_days = 300  # Or another suitable maximum
investment_days = st.number_input("How many days would you like to invest?", min_value=1, max_value=max_investment_days)

# Calculate investment_end_date based on investment_days
investment_end_date = fixed_start_date + timedelta(days=investment_days)

# Load Tesla stock data for the selected date range
ticker = "TSLA"
stock_data = yf.download(ticker, start=fixed_start_date.strftime('%Y-%m-%d'), end=investment_end_date.strftime('%Y-%m-%d'))

# Check if data is available
if stock_data.empty:
    st.write("No data found for Tesla stock. Please check the date range.")
else:
    # Flatten multi-level columns if needed
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

    # Display column names for debugging
    st.write("Columns:", stock_data.columns)

    # Ensure the index is in datetime format
    stock_data.index = pd.to_datetime(stock_data.index)

    # Filter data to show only the specified investment period
    stock_data_filtered = stock_data.loc[fixed_start_date:investment_end_date]

    # If `stock_data_filtered` is empty, inform the user about weekends or holidays
    if stock_data_filtered.empty:
        st.write("No trading data available for the selected date range. This may be due to weekends or holidays.")

    else:
        # Detect the correct 'Close' column
        close_column = next((col for col in stock_data.columns if 'Close' in col), None)
        if close_column is None:
            st.write("Expected 'Close' column not found. Available columns:", stock_data.columns)
        else:
            # Display data as a table (optional)
            st.subheader("Tesla Stock Data")
            st.write(stock_data_filtered)  # Display only the filtered rows

            # Create a Plotly line chart for stock prices
            fig = go.Figure()

            # Add a line trace for the closing price
            fig.add_trace(go.Scatter(
                x=stock_data_filtered.index, 
                y=stock_data_filtered[close_column], 
                mode='lines', 
                name='Close Price'
            ))

            # Customize the layout of the chart
            fig.update_layout(
                title=f"Tesla Stock Price from {fixed_start_date.strftime('%Y-%m-%d')} to {investment_end_date.strftime('%Y-%m-%d')}",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                height=600,
                width=1000
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Display investment summary
            st.subheader("Investment Summary")
            st.write(f"Investment Start Date: {fixed_start_date.strftime('%Y-%m-%d')}")
            st.write(f"Investment End Date: {investment_end_date.strftime('%Y-%m-%d')}")
            st.write(f"Duration: {investment_days} days")
