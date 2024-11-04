import joblib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import shap
from datetime import datetime, timedelta
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report

# Set page configuration
st.set_page_config(page_title="High-Frequency Trading Backtesting", layout="wide")

# App Title and Description
st.title("📈 High-Frequency Trading Backtesting")
st.markdown("""
Welcome to the High-Frequency Trading Backtesting App! This interactive tool allows you to simulate trading strategies based on machine learning models. Adjust the parameters on the sidebar and explore the results.
""")

# Sidebar Inputs
st.sidebar.header("Configure Your Backtest")
input_value_amount = st.sidebar.number_input(
    "💰 Investment Amount ($):",
    min_value=0.0,
    max_value=1_000_000.0,
    value=50_000.0,
    step=1_000.0,
    help="Enter the amount you would like to invest."
)

max_investment_days = 200
investment_days = st.sidebar.slider(
    "⏳ Investment Duration (Days):",
    min_value=1,
    max_value=max_investment_days,
    value=30,
    help="Select the number of days you would like to trade."
)

stock_options = ["Tesla"]
selected_stock = st.sidebar.selectbox(
    "📊 Select Stock:",
    options=stock_options,
    index=0,
    help="Choose which stock's data you want to analyze."
)

# Button to run backtest
run_backtest = st.sidebar.button("Run Backtest")

# Load data function
@st.cache_data
def load_data():
    stock_data = pd.read_csv("https://raw.githubusercontent.com/jgranizo/HackNJIT2024/main/streamlit_app/final_data.csv")
    stock_data['date'] = pd.to_datetime(stock_data['date'], format='%Y-%m-%d')
    return stock_data

# Load models function
@st.cache_resource
def load_models():
    try:
        xgb_regressor = XGBRegressor()
        xgb_regressor.load_model('./models/best_xgb_regressor.model')
    except Exception as e:
        st.error(f"Error loading regressor: {e}")
        xgb_regressor = None

    try:
        xgb_classifier = joblib.load('./models/best_xgb_classifier.pkl')
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        xgb_classifier = None

    return xgb_regressor, xgb_classifier

# Main logic
if run_backtest:
    stock_data = load_data()
    xgb_regressor, xgb_classifier = load_models()

    # Date handling logic
    start_date = stock_data['date'].min()
    desired_end_date = start_date + timedelta(days=investment_days)
    available_dates = stock_data['date'].sort_values()
    end_date = min(desired_end_date, available_dates.max())
    actual_investment_days = (end_date - start_date).days

    # Filter data within the date range
    mask = (stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)
    data_in_range = stock_data.loc[mask]

    if data_in_range.empty:
        st.error("No data available for the selected date range. Please adjust the investment duration.")
    else:
        # Feature and target selection
        excluded_features = ['Tesla_Log_Return', 'Tesla_Direction', 'date']
        features = [col for col in stock_data.columns if col not in excluded_features]
        X = data_in_range[features]
        y_reg = data_in_range['Tesla_Log_Return']
        y_clf = data_in_range['Tesla_Direction']

        if xgb_regressor and xgb_classifier:
            y_pred_reg = xgb_regressor.predict(X)
            y_pred_clf = xgb_classifier.predict(X)
            accuracy = accuracy_score(y_clf, y_pred_clf)

            # Results DataFrame
            results_df = pd.DataFrame({
                'Date': data_in_range['date'],
                'Actual Log Return': y_reg,
                'Predicted Log Return': y_pred_reg,
                'Actual Direction': y_clf,
                'Predicted Direction': y_pred_clf
            })
            results_df['Actual Cumulative Return'] = (1 + results_df['Actual Log Return']).cumprod() - 1
            results_df['Predicted Cumulative Return'] = (1 + results_df['Predicted Log Return']).cumprod() - 1

            # Investment summary
            investment_amount = input_value_amount
            actual_profit = investment_amount * results_df['Actual Cumulative Return'].iloc[-1]
            predicted_profit = investment_amount * results_df['Predicted Cumulative Return'].iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Start Date", start_date.strftime('%Y-%m-%d'))
            col2.metric("End Date", end_date.strftime('%Y-%m-%d'))
            col3.metric("Duration (Days)", actual_investment_days)

            col1, col2 = st.columns(2)
            col1.metric("Initial Investment", f"${investment_amount:,.2f}")
            col1.metric("Actual Profit", f"${actual_profit:,.2f}")
            col1.metric("Actual Return", f"{results_df['Actual Cumulative Return'].iloc[-1] * 100:.2f}%")
            col2.metric("Predicted Profit", f"${predicted_profit:,.2f}")
            col2.metric("Predicted Return", f"{results_df['Predicted Cumulative Return'].iloc[-1] * 100:.2f}%")
            col2.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

            # SHAP Feature Importance
            shap_explainer = shap.TreeExplainer(xgb_classifier)
            shap_values = shap_explainer.shap_values(X)
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(shap_values).mean(axis=0)
            }).sort_values(by="Importance", ascending=False)

            fig_shap = px.bar(feature_importance, x="Importance", y="Feature", orientation="h",
                              title="🔍 SHAP Feature Importance")
            st.plotly_chart(fig_shap, use_container_width=True, key="shap_chart")

            # Plot Actual vs Predicted Log Returns
            fig_log_returns = go.Figure()
            fig_log_returns.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Actual Log Return'], mode='lines', name='Actual Log Return'))
            fig_log_returns.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Predicted Log Return'], mode='lines', name='Predicted Log Return'))
            fig_log_returns.update_layout(title="Actual vs Predicted Tesla Log Return Over Time", xaxis_title="Date", yaxis_title="Log Return")
            st.plotly_chart(fig_log_returns, use_container_width=True, key="log_returns_chart")

            # Plot Actual vs Predicted Cumulative Returns
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Actual Cumulative Return'], mode='lines', name='Actual Cumulative Return'))
            fig_cumulative.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Predicted Cumulative Return'], mode='lines', name='Predicted Cumulative Return'))
            fig_cumulative.update_layout(title="Actual vs Predicted Cumulative Return Over Time", xaxis_title="Date", yaxis_title="Cumulative Return")
            st.plotly_chart(fig_cumulative, use_container_width=True, key="cumulative_chart")

            # Classification Report
            with st.expander("View Classification Report"):
                report_df = pd.DataFrame(classification_report(y_clf, y_pred_clf, output_dict=True)).transpose()
                st.write(report_df)
else:
    st.info("Adjust the parameters on the sidebar and click **Run Backtest** to start.")
