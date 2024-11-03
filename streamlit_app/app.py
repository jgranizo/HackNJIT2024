import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
from xgboost import XGBClassifier, XGBRegressor
import pickle
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
# App Title
st.title("High-Frequency Trading Backtesting")

# User input: Investment amount
input_value_amount = st.number_input(
    "Enter amount of value you would like to invest:",
    min_value=0.0,
    max_value=1000000.0,
    value=50000.0,
)

# User input: Investment duration in days
max_investment_days = 150  # Maximum allowed investment period
investment_days = st.number_input(
    "How many days would you like to trade?", min_value=1, max_value=max_investment_days, value=30
)

# Set fixed start date for trading
# Load stock data from CSV
stock_data = pd.read_csv("final_df_app.csv")
# Convert 'date' column to datetime format
stock_data['date'] = pd.to_datetime(stock_data['date'], format='%Y-%m-%d')

start_date = stock_data['date'].min()
fixed_start_date = start_date


# Calculate the desired investment end date based on the selected duration
desired_end_date = fixed_start_date + timedelta(days=investment_days)

# Adjust the end date if it exceeds available data
available_dates = stock_data['date'].sort_values()

start_date = fixed_start_date
end_date = desired_end_date

if desired_end_date in set(available_dates):
    end_date = desired_end_date
else:
    # Find the closest date after the desired end date
    future_dates = available_dates[available_dates >= desired_end_date]
    if not future_dates.empty:
        end_date = future_dates.iloc[0]
        st.write(f"The desired end date {desired_end_date.strftime('%Y-%m-%d')} is not available in the data.")
        st.write(f"Adjusting end date to the next available date: {end_date.strftime('%Y-%m-%d')}.")
    else:
        # If no future dates are available, use the last available date
        end_date = available_dates.max()
        st.write(f"The desired end date {desired_end_date.strftime('%Y-%m-%d')} exceeds available data.")
        st.write(f"Adjusting end date to the last available date: {end_date.strftime('%Y-%m-%d')}.")
        
actual_investment_days = (end_date - start_date).days

# Filter data within the date range
mask = (stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)
data_in_range = stock_data.loc[mask]

if data_in_range.empty:
    st.error("No data available for the selected date range. Please adjust the investment duration.")
    st.stop()

# Exclude 'Tesla_Log_Return', 'Tesla_Direction', and 'date' from features
excluded_features = ['Tesla_Log_Return', 'Tesla_Direction', 'date']
features = [col for col in stock_data.columns if col not in excluded_features]




# Features and targets
X = data_in_range[features]
y_reg = data_in_range['Tesla_Log_Return']
y_clf = data_in_range['Tesla_Direction']

# Reorder columns in X to match the trained model

correct_feature_order = [
    'Tesla', 'Ford', 'Ford_Log_Return', 'Tesla_MA_5', 'Tesla_MA_50', 
    'Tesla_Volatility', 'polarity', 'Tesla_MA_100', 'Tesla_MA_200', 
    'Tesla_Bollinger_Upper_5', 'Tesla_Bollinger_Lower_5', 'Tesla_Bollinger_Upper_50',
    'Tesla_Bollinger_Lower_50', 'Tesla_RSI_5', 'Tesla_RSI_50', 
    'Tesla_MACD', 'Tesla_Log_Return_Lag1', 'Tesla_Log_Return_Lag5'
]

# Reorder columns in X to match the trained model
X = X[correct_feature_order]

try:
    xgb_classifier = joblib.load('../models/best_xgb_classifier.pkl')
    # xgb_classifier.set_params(predictor="gpu_predictor")  # Optional: Set GPU predictor if applicable
except FileNotFoundError:
    st.error("Classifier model file not found. Please ensure 'best_xgb_classifier.pkl' exists in the 'models' directory.")
    xgb_classifier = None
except Exception as e:
    st.error(f"Error loading classifier: {e}")
    xgb_classifier = None

# Load the regressor model
try:
    xgb_regressor = XGBRegressor()
    xgb_regressor.load_model('../models/best_xgb_regressor.model')

    # xgb_regressor.set_params(predictor="gpu_predictor")  # Optional: Set GPU predictor if applicable
    st.write("Regressor model loaded successfully.")
except FileNotFoundError:
    st.error("Regressor model file not found. Please ensure 'best_xgb_regressor.model' exists in the 'models' directory.")
    xgb_regressor = None
except Exception as e:
    st.error(f"Error loading regressor: {e}")
    xgb_regressor = None
# Load the models

# Make predictions
y_pred_clf = xgb_classifier.predict(X)
y_pred_reg = xgb_regressor.predict(X)

accuracy = accuracy_score(y_clf, y_pred_clf)

# Create a dataframe to display actual vs predicted values
results_df = pd.DataFrame({
    'Date': data_in_range['date'],
    'Actual Log Return': y_reg,
    'Predicted Log Return': y_pred_reg,
    'Actual Direction': y_clf,
    'Predicted Direction': y_pred_clf
})

# Compute cumulative returns
results_df['Actual Cumulative Return'] = (1 + results_df['Actual Log Return']).cumprod() - 1
results_df['Predicted Cumulative Return'] = (1 + results_df['Predicted Log Return']).cumprod() - 1

# Calculate investment outcomes
investment_amount = input_value_amount
actual_return = results_df['Actual Cumulative Return'].iloc[-1]
predicted_return = results_df['Predicted Cumulative Return'].iloc[-1]

actual_profit = investment_amount * actual_return
predicted_profit = investment_amount * predicted_return

# Display investment summary
st.subheader("Investment Summary")
st.write(f"Investment Start Date: {start_date.strftime('%Y-%m-%d')}")
st.write(f"Investment End Date: {end_date.strftime('%Y-%m-%d')}")
st.write(f"Duration: {investment_days} days")
st.write(f"Initial Investment Amount: ${investment_amount:,.2f}")
st.write(f"Actual Return: {actual_return*100:.2f}%")
st.write(f"Predicted Return: {predicted_return*100:.2f}%")
st.write(f"Actual Profit: ${actual_profit:,.2f}")
st.write(f"Predicted Profit: ${predicted_profit:,.2f}")

# Display the results
st.subheader("Predicted vs Actual Values")
st.write(results_df)

# Display classification accuracy
st.subheader("Classification Model Accuracy")
st.write(f"Accuracy: {accuracy*100:.2f}%")

# Display classification report
report = classification_report(y_clf, y_pred_clf, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.subheader("Classification Report")
st.write(report_df)

# Plot Actual vs Predicted Log Returns
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results_df['Date'],
    y=results_df['Actual Log Return'],
    mode='lines',
    name='Actual Log Return'
))
fig.add_trace(go.Scatter(
    x=results_df['Date'],
    y=results_df['Predicted Log Return'],
    mode='lines',
    name='Predicted Log Return'
))
fig.update_layout(
    title="Actual vs Predicted Tesla Log Return Over Time",
    xaxis_title="Date",
    yaxis_title="Log Return",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Plot Actual vs Predicted Cumulative Returns
fig_cumulative = go.Figure()
fig_cumulative.add_trace(go.Scatter(
    x=results_df['Date'],
    y=results_df['Actual Cumulative Return'],
    mode='lines',
    name='Actual Cumulative Return'
))
fig_cumulative.add_trace(go.Scatter(
    x=results_df['Date'],
    y=results_df['Predicted Cumulative Return'],
    mode='lines',
    name='Predicted Cumulative Return'
))
fig_cumulative.update_layout(
    title="Actual vs Predicted Cumulative Return Over Time",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    template="plotly_white"
)
st.plotly_chart(fig_cumulative, use_container_width=True)