# Main Streamlit app file
# Displays UI, handles user inputs, and runs the model

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# App Title
st.title("High-Frequency Trading Backtesting")

# User input: Investment amount
input_value_amount = st.number_input("Enter amount of value you would like to invest:", min_value=0.0, max_value=100.0, value=50.0)

# User input: Stock selection
stock_option = st.selectbox("Which stock would you like to select?", ("Tesla", "Ford", "Both"))

# Set fixed start date for investment
fixed_start_date = datetime(2024, 1, 1)

# User input: Investment duration in days
max_investment_days = 300  # Maximum allowed investment period
investment_days = st.number_input("How many days would you like to invest?", min_value=1, max_value=max_investment_days)

# Calculate the investment end date based on the selected duration
investment_end_date = (fixed_start_date + timedelta(days=investment_days)).replace(tzinfo=None)

# Load stock data from CSVs if available
if stock_option == "Tesla":
    # Load Tesla data from CSV
    stock_data = pd.read_csv("final_df_app.csv")
    st.write(stock_data.columns)  # Display columns to ensure correct loading

    # Use 'dates' as x-axis and 'Tesla_Log_Return' as y-axis
    test_dates = pd.to_datetime(stock_data['date'])  # Convert 'dates' to datetime format if not already
    data = stock_data['Tesla_Log_Return']

    # Create a Plotly line chart with test_dates and data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=data,
        mode='lines',
        name='Tesla Log Return'
    ))

    # Customize the layout of the chart
    fig.update_layout(
        title="Tesla Log Return Over Time",
        xaxis_title="Date",
        yaxis_title="Log Return",
        template="plotly_white"
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display investment summary
    st.subheader("Investment Summary")
    st.write(f"Investment Start Date: {fixed_start_date.strftime('%Y-%m-%d')}")
    st.write(f"Investment End Date: {investment_end_date.strftime('%Y-%m-%d')}")
    st.write(f"Duration: {investment_days} days")

else:
    st.write("Please select 'Tesla' to view the log return plot.")
