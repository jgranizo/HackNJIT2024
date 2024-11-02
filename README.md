# Predicting Stock Market Movements Using High-Frequency Trading Data and Machine Learning

## Project Overview

This project explores the potential of machine learning in predicting stock market movements within a high-frequency trading (HFT) environment. Leveraging Tesla and Ford stock data (from October 31, 2018, to October 31, 2023), the study examines the predictive capabilities of technical indicators and sentiment analysis, using Random Forest and XGBoost models to improve forecast accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
  - [Stock Data Collection](#stock-data-collection)
  - [Technical Indicators](#technical-indicators)
  - [Sentiment Analysis](#sentiment-analysis)
- [Machine Learning Models](#machine-learning-models)
  - [Random Forest Regressor](#random-forest-regressor)
  - [XGBoost Model](#xgboost-model)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction

Stock market prediction is challenging, especially in high-frequency contexts, due to market dynamics and data volume. This project evaluates the ability of machine learning to provide insights into future price changes, with an emphasis on reducing prediction error. The Random Forest Regressor was used as a baseline, with additional experimentation using XGBoost for improved classification and regression tasks.

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

![image](https://github.com/user-attachments/assets/45748c93-47d2-47d7-8c23-28df93aa3ef6)

## Machine Learning Models

### Random Forest Regressor

The Random Forest Regressor was trained with the processed data to predict stock price changes. By incorporating technical indicators and sentiment scores, the model achieved an improved R² score, demonstrating better alignment with stock movements compared to baseline attempts.

![image](https://github.com/user-attachments/assets/0cec7464-a2fd-4ef5-baff-925822b4a31a)

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

With these settings, the model achieved a validation accuracy of 67.61% and improved to 72% on testing.

#### RandomSearch for Hyperparameter Tuning

RandomSearch further optimized performance, achieving:
- **Test Accuracy**: 74.63%
- **Precision, Recall, and F1 Score**: 0.75



RandomSearch Tuning result
![image](https://github.com/user-attachments/assets/eceae90d-f416-492e-b45c-c4a241ba422b) 



#### Regression with XGBoost

For continuous price prediction, the XGBoost regression model was evaluated with Root Mean Square Error (RMSE) and R² scores:

- **Test RMSE**: 0.0282
- **Test R² Score**: 0.3203


GridSearch Tuning result 
![image](https://github.com/user-attachments/assets/43990359-d550-4226-9051-155c0d5cc6c4)


RandomSearch Tuning result
![image](https://github.com/user-attachments/assets/8947b176-3ddd-405f-85d9-6906b825c772) 



## Results

Feature importance analysis highlighted the top predictive indicators:

- **Tesla_RSI_5** and **Ford_Log_Return** as key indicators for price movements
- Other impactful features included **Tesla_Log_Return_Lag1**, **Tesla_RSI_50**, and **Tesla_Log_Return_Lag5**

![image](https://github.com/user-attachments/assets/76442249-018a-42cb-8508-8e55b21e141b)

## Conclusion

This project demonstrates the value of machine learning in high-frequency trading environments. XGBoost, optimized with RandomSearch, achieved the best classification accuracy at 74.63% and a refined fit in regression predictions. Technical indicators provided substantial predictive insight, while sentiment analysis was less impactful, possibly due to Tesla's general positive trend.

## Future Work

To build on these findings, future work could:

- Incorporate diverse sentiment sources to mitigate positive bias
- Experiment with alternative feature engineering for improved model robustness across different stocks and market periods


---

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jgranizo/HackNJIT2024.git
