
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

max_investment_days = 200  # Maximum allowed investment period
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
    # Load the regressor model
    try:
        xgb_regressor = XGBRegressor()
        xgb_regressor.load_model('./models/best_xgb_regressor.model')
        st.write("Regressor model loaded successfully.")

    except Exception as e:
        st.error(f"Error loading regressor: {e}")
        xgb_regressor = None

    # Load the classifier model
    try:
        xgb_classifier = joblib.load('./models/best_xgb_classifier.pkl')        
        st.write("Classifier model loaded successfully.")

    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        xgb_classifier = None

    return xgb_regressor, xgb_classifier

# Main logic
if run_backtest:
    st.subheader("🔄 Running Backtest...")
    stock_data = load_data()
    xgb_regressor, xgb_classifier = load_models()

    start_date = stock_data['date'].min()
    fixed_start_date = start_date

    # Calculate the desired investment end date based on the selected duration
    desired_end_date = fixed_start_date + timedelta(days=int(investment_days))

    # Adjust the end date if it exceeds available data
    available_dates = stock_data['date'].sort_values()

    if desired_end_date in set(available_dates):
        end_date = desired_end_date
    else:
        # Find the closest date after the desired end date
        future_dates = available_dates[available_dates >= desired_end_date]
        if not future_dates.empty:
            end_date = future_dates.iloc[0]
            st.warning(f"The desired end date {desired_end_date.strftime('%Y-%m-%d')} is not available. Adjusted to {end_date.strftime('%Y-%m-%d')}.")
        else:
            # If no future dates are available, use the last available date
            end_date = available_dates.max()
            st.warning(f"The desired end date {desired_end_date.strftime('%Y-%m-%d')} exceeds available data. Adjusted to {end_date.strftime('%Y-%m-%d')}.")

    actual_investment_days = (end_date - start_date).days

    # Filter data within the date range
    mask = (stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)
    data_in_range = stock_data.loc[mask]

    if data_in_range.empty:
        st.error("No data available for the selected date range. Please adjust the investment duration.")
        st.stop()

    # Select features based on the selected stock
    if selected_stock == "Tesla":
        excluded_features = ['Tesla_Log_Return', 'Tesla_Direction', 'date']
    
    features = [col for col in stock_data.columns if col not in excluded_features]

    # Features and targets
    X = data_in_range[features]
    y_reg = data_in_range['Tesla_Log_Return']
    y_clf = data_in_range['Tesla_Direction']

    # Reorder columns in X to match the trained model
    # (Assuming correct_feature_order is adjusted based on selected_stock)
    correct_feature_order = X.columns.tolist()
    X = X[correct_feature_order]

    if xgb_regressor is not None and xgb_classifier is not None:
        # Make predictions
        y_pred_reg = xgb_regressor.predict(X)
        y_pred_clf = xgb_classifier.predict(X)

        # Compute accuracy
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

        # Display investment summary using columns
        st.subheader("💼 Investment Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Start Date", start_date.strftime('%Y-%m-%d'))
        col2.metric("End Date", end_date.strftime('%Y-%m-%d'))
        col3.metric("Duration (Days)", actual_investment_days)

        col1, col2 = st.columns(2)
        col1.metric("Initial Investment", f"${investment_amount:,.2f}")
        col1.metric("Actual Profit", f"${actual_profit:,.2f}")
        col1.metric("Actual Return", f"{actual_return*100:.2f}%")
        col2.metric("Predicted Profit", f"${predicted_profit:,.2f}")
        col2.metric("Predicted Return", f"{predicted_return*100:.2f}%")
        col2.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        # Display the results
        st.subheader("📊 Predicted vs Actual Values")
        st.dataframe(results_df)


        #feature importance graph         # Initialize SHAP explainer for the classifier
        shap_explainer = shap.TreeExplainer(xgb_classifier)
        shap_values = shap_explainer.shap_values(X)

        # Convert SHAP values to DataFrame for better handling with Plotly
        shap_df = pd.DataFrame(shap_values, columns=X.columns)

        # Calculate mean absolute SHAP values for feature importance
        feature_importance = shap_df.abs().mean().sort_values(ascending=False)
        feature_importance_df = feature_importance.reset_index()
        feature_importance_df.columns = ['Feature', 'Importance']

        # Plot with Plotly
        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='🔍 SHAP Feature Importance',
            width=800,  # Set the width
            height=600,  # Set the height
        )

        # Customize the dark theme layout
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            font_color='white',             # White font for dark theme
            title_font_size=20
        )
        fig.update_traces(marker_color='cyan')  # Customize bar color for dark theme

        # Display in Streamlit
        st.plotly_chart(fig)

        # Display classification report in an expandable section
        with st.expander("View Classification Report"):
            report = classification_report(y_clf, y_pred_clf, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write(report_df)

        # Plot Actual vs Predicted Log Returns
        st.subheader("📈 Log Returns Over Time")
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
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    ###new
       
        # Plot Actual vs Predicted Cumulative Returns
        st.subheader("📈 Cumulative Returns Over Time")
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
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)
    else:
        st.error("Models could not be loaded. Please check the model files and try again.")
else:
    st.info("Adjust the parameters on the sidebar and click **Run Backtest** to start.")
