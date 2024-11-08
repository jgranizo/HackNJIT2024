# Predicting Stock Market Movements Using High-Frequency Trading Data and Machine Learning

## Project Overview

This project explores the potential of machine learning in predicting stock market movements within a high-frequency trading (HFT) environment. Leveraging Tesla and Ford stock data (from October 31, 2018, to October 31, 2023), the study examines the predictive capabilities of technical indicators and sentiment analysis, using Random Forest and XGBoost models to improve forecast accuracy. Check out our StreamLit App to try the models https://steampulse.streamlit.app/

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
  - [Stock Data Collection](#stock-data-collection)
  - [Technical Indicators](#technical-indicators)
  - [Sentiment Analysis](#sentiment-analysis)
- [Machine Learning Models](#machine-learning-models)
  - [Random Forest Regressor](#random-forest-regressor)
  - [XGBoost Model](#xgboost-model)
    - [Data Preparation and Train-Test Split](#data-preparation-and-train-test-split)
    - [Classification with XGBoost and GridSearch](#classification-with-xgboost-and-gridsearch)
    - [RandomSearch for Hyperparameter Tuning](#randomsearch-for-hyperparameter-tuning)
    - [Regression with XGBoost](#regression-with-xgboost)
- [LSTM Model](#lstm-model)
- [Results](#results)
- [Technologies Used](#technologies-used)
  - [Programming Languages](#programming-languages)
  - [Libraries and Frameworks](#libraries-and-frameworks)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Deployment](#deployment)
  - [Deployment Steps](#deployment-steps)

## Introduction

Stock market prediction is challenging, especially in high-frequency contexts, due to market dynamics and data volume. This project evaluates the ability of machine learning to provide insights into future price changes, emphasizing reducing prediction error. The Random Forest Regressor was used as a baseline, with additional experimentation using XGBoost for improved classification and regression tasks.

## Data Preprocessing

### Stock Data Collection

Historical stock data for Tesla and Ford was obtained from Yahoo Finance, including daily prices over a five-year period. This dataset provided a foundation for calculating technical indicators to identify patterns and inform the models.

### Technical Indicators

Several indicators were computed to capture stock behavior:

- **Tesla_Log_Return** and **Ford_Log_Return** for logarithmic returns
- **Tesla_MA_5** and **Tesla_MA_50** for 5-day and 50-day moving averages
- **Tesla_Volatility** to assess price volatility over time
- **Bollinger Bands**, **Relative Strength Index (RSI)**, and **MACD** for additional momentum and trend insights
- **Lagged Returns** with 1-day and 5-day lags for historical context

### Sentiment Analysis

Using a Financial News API, approximately 1,000 news articles on Tesla were collected to gauge the impact of sentiment on stock price movement. Sentiment scores revealed a positive bias, reflecting public optimism around Tesla.

![Sentiment Analysis](assets/sentiment_analysis.png)

## Machine Learning Models

### Random Forest Regressor

The Random Forest Regressor was trained with the processed data to predict stock price changes. By incorporating technical indicators and sentiment scores, the model achieved an improved R² score, demonstrating better alignment with stock movements compared to baseline attempts.

![Random Forest Results](assets/random_forest_results.png)

### XGBoost Model

To enhance predictions, an XGBoost model was implemented for both classification and regression tasks, with hyperparameter tuning via GridSearch and RandomSearch.

#### Data Preparation and Train-Test Split

Data was split into 80% training and 20% testing sets to facilitate effective model tuning and evaluation.

#### Classification with XGBoost and GridSearch

For classification, the best hyperparameters identified using GridSearch were:

- `colsample_bytree`: 1.0
- `learning_rate`: 0.1
- `max_depth`: 7
- `n_estimators`: 50
- `subsample`: 0.8

With these settings, the model achieved a validation accuracy of **67.61%** and improved to **72%** on testing.

#### RandomSearch for Hyperparameter Tuning

RandomSearch further optimized performance, achieving:

- **Test Accuracy**: 74.63%
- **Precision, Recall, and F1 Score**: 0.75

![RandomSearch Tuning Result](assets/randomsearch_tuning_result.png)

#### Regression with XGBoost

For continuous price prediction, the XGBoost regression model was evaluated with Root Mean Square Error (RMSE) and R² scores:

- **Test RMSE**: 0.0282
- **Test R² Score**: 0.3203

##### GridSearch Tuning Result

![GridSearch Tuning Result](assets/gridsearch_tuning_result.png)

##### RandomSearch Tuning Result

![RandomSearch Tuning Result](assets/randomsearch_regression_tuning_result.png)

## LSTM Model

This project uses an LSTM model to predict Tesla's logarithmic returns (**Tesla_Log_Return**) based on time-series data, capturing sequential patterns crucial for financial forecasting. The model includes two LSTM layers with dropout for regularization and a dense output layer, trained over 5 folds of time series cross-validation to ensure robustness. Each fold shows improved performance, with Mean Squared Error (MSE) and Mean Absolute Error (MAE) decreasing, and R² scores rising from 0.34 in Fold 1 to 0.66 in Fold 5. This trend indicates that the model benefits from more data, effectively generalizing to unseen data. With further tuning or additional data, this LSTM model can be valuable for real-world prediction tasks in finance.

## Results

Feature importance analysis highlighted the top predictive indicators:

- **Tesla_RSI_5** and **Ford_Log_Return** as key indicators for price movements
- Other impactful features included **Tesla_Log_Return_Lag1**, **Tesla_RSI_50**, and **Tesla_Log_Return_Lag5**

![Feature Importance](assets/feature_importance.png)

## Technologies Used

### Programming Languages

- **Python**: The primary language used for data analysis, machine learning model development, and deployment.

### Libraries and Frameworks

- **NumPy**: For numerical operations and handling multi-dimensional arrays.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn (sklearn)**: For machine learning algorithms, model evaluation, and preprocessing utilities.
- **XGBoost**: For gradient boosting algorithms used in classification and regression tasks.
- **RandomForest**: From scikit-learn, used as a baseline regression model.
- **Joblib**: For model serialization and saving trained models.
- **Plotly**: For creating interactive visualizations and plots.
- **Streamlit**: For building and deploying the interactive web application.

## Conclusion

This project demonstrates the value of machine learning in high-frequency trading environments. XGBoost, optimized with RandomSearch, achieved the best classification accuracy at **74.63%** and a refined fit in regression predictions. Technical indicators provided substantial predictive insight, while sentiment analysis was less impactful, possibly due to Tesla's general positive trend.

## Future Work

To build on these findings, future work could:

- Incorporate diverse sentiment sources to mitigate positive bias
- Experiment with alternative feature engineering for improved model robustness across different stocks and market periods

---

## Deployment

The developed machine learning models and data visualization tools were deployed as an interactive web application using **Streamlit**. This deployment allows users to input parameters, visualize predictions, and explore model performance in real-time.

To access the deployed application, visit: [Streamlit App URL](https://your-streamlit-app-url)

### Deployment Steps

1. **Set Up Environment**: Ensure all required libraries are installed using `requirements.txt`.
2. **Run Streamlit App**: Execute the command `streamlit run app.py` to launch the application locally.
3. **Deploy on Streamlit Sharing**: Push the code to a GitHub repository and connect it to Streamlit Sharing for cloud deployment.

---

