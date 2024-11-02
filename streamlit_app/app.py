#main Streamlit app file
#display UI, handle user inputs and run model
import streamlit as st
import pandas as pd
from model import model

st.title("High-Frequency Trading Backtesting")

# User input
input_value = st.number_input("Enter a parameter value:", min_value=0.0, max_value=100.0, value=50.0)

# Run model and display results
if st.button("Run Backtest"):
    result = model.run_backtest(input_value)
    st.write("Backtest Result:", result)
