# Bitcoin Price Forecasting

## Overview
This Python notebook is designed to forecast Bitcoin prices using historical data and machine learning algorithms. It utilizes the N-BEATS algorithm to predict future Bitcoin values based on 8 years of past price data. The project aims to provide a robust tool for financial analysts and enthusiasts to understand potential future trends in Bitcoin pricing.

## Features
- **Data Analysis:** Exploration and visualization of historical Bitcoin price data.
- **Model Training:** Utilizes an ensemble of models, in particular N-BEATS algorithm and RNN architectures for accurate time-series forecasting.
- **Error Metrics:** Calculation of Mean Absolute Percentage Error (MAPE) to evaluate the model performance.
- **Prediction:** Outputs future price predictions based on the trained model.

## Technical Details

### Data Preprocessing
- **Data Loading:** Historical Bitcoin price data is loaded for analysis.
- **Cleaning:** Cleanses the data to address any missing or inconsistent data points.
- **Normalization:** Normalizes the data to ensure efficient and effective model training.

### Model Details
- **N-BEATS Algorithm:** A deep learning model specialized for time-series forecasting, chosen for its robustness and capability to handle complex data patterns.
- **Model Configuration:** Configured with blocks focusing on trend and seasonality to accurately capture the cyclic nature of Bitcoin prices.
- **Training:** Parameters are meticulously adjusted to minimize prediction error during the model training phase.

### Prediction and Evaluation
- **Forecasting:** Forecasts future Bitcoin prices using the trained model.
- **Evaluation:** Uses the Mean Absolute Percentage Error (MAPE) to evaluate model performance.
- **Visualization:** Visualizes predictions alongside actual data to intuitively assess model performance.

## Requirements
To run this notebook, you will need the following:
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required Python libraries:
  ```bash
  pip install numpy pandas matplotlib tensorflow
